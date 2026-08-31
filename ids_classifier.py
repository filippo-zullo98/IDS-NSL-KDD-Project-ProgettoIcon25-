# =============================================================================
# ids_classifier.py  –  IDS ibrido NSL-KDD  
# =============================================================================

# ── Fase 1: Importazioni ─────────────────────────────────────────────────────

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from owlready2 import get_ontology
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (classification_report, confusion_matrix,
                             accuracy_score, f1_score)
from sklearn.feature_selection import VarianceThreshold

# ── Fase 2: Caricamento dati ──────────────────────────────────────────────────

FEATURE_NAMES = [
    'duration', 'protocol_type', 'service', 'flag', 'src_bytes', 'dst_bytes',
    'land', 'wrong_fragment', 'urgent', 'hot', 'num_failed_logins',
    'logged_in', 'num_compromised', 'root_shell', 'su_attempted', 'num_root',
    'num_file_creations', 'num_shells', 'num_access_files', 'num_outbound_cmds',
    'is_host_login', 'is_guest_login', 'count', 'srv_count', 'serror_rate',
    'srv_serror_rate', 'rerror_rate', 'srv_rerror_rate', 'same_srv_rate',
    'diff_srv_rate', 'srv_diff_host_rate', 'dst_host_count',
    'dst_host_srv_count', 'dst_host_same_srv_rate', 'dst_host_diff_srv_rate',
    'dst_host_same_src_port_rate', 'dst_host_srv_diff_host_rate',
    'dst_host_serror_rate', 'dst_host_srv_serror_rate', 'dst_host_rerror_rate',
    'dst_host_srv_rerror_rate', 'attack_type', 'difficulty'
]

try:
    df_train = pd.read_csv('KDDTrain+.txt', names=FEATURE_NAMES)
    df_test  = pd.read_csv('KDDTest+.txt',  names=FEATURE_NAMES)
    print("Dataset caricato: train", df_train.shape, "| test", df_test.shape)
except FileNotFoundError as e:
    print(f"Errore: {e}. Posiziona KDDTrain+.txt e KDDTest+.txt nella stessa directory.")
    exit(1)

# ── Fase 3: Mapping attack_type → categoria (letto dall'ontologia OWL) ────────

def build_category_map_from_ontology(owl_path: str) -> dict:
    onto = get_ontology(owl_path).load()
    ns = onto.get_namespace("http://www.semanticweb.org/nsl-kdd-ids#")

    CATEGORY_CLASS_NAMES = {
        'NormalTraffic': 'Normal',
        'DoSAttack':     'DoS',
        'ProbeAttack':   'Probe',
        'R2LAttack':     'R2L',
        'U2RAttack':     'U2R',
    }

    category_map = {}

    for individual in onto.individuals():
        # Controlla se ha la data property hasAttackName
        names = getattr(individual, 'hasAttackName', [])
        if not names:
            continue
        attack_name = names[0].lower()

        # Risali la categoria guardando il tipo diretto dell'individuo
        for cls in individual.is_a:
            cls_name = getattr(cls, 'name', '')
            if cls_name in CATEGORY_CLASS_NAMES:
                category_map[attack_name] = CATEGORY_CLASS_NAMES[cls_name]
                break
            # Controlla anche le superclassi (es. DoS_Category è istanza di DoSAttack)
            for parent in getattr(cls, 'is_a', []):
                parent_name = getattr(parent, 'name', '')
                if parent_name in CATEGORY_CLASS_NAMES:
                    category_map[attack_name] = CATEGORY_CLASS_NAMES[parent_name]
                    break

    print(f"  Ontologia caricata: {len(category_map)} tipi di attacco mappati.")
    return category_map

print("\n── Caricamento ontologia ─────────────────────────────────────────────")
CATEGORY_MAP = build_category_map_from_ontology("file://nsl_kdd_ontology.owl")

# Fallback per eventuali attacchi non presenti nell'ontologia (dataset NSL-KDD completo)
_FALLBACK_MAP = {
    # DoS
    'back':'DoS','land':'DoS','neptune':'DoS','pod':'DoS','smurf':'DoS',
    'teardrop':'DoS','apache2':'DoS','udpstorm':'DoS','processtable':'DoS',
    'worm':'DoS','mailbomb':'DoS',
    # Probe
    'satan':'Probe','ipsweep':'Probe','nmap':'Probe','portsweep':'Probe',
    'mscan':'Probe','saint':'Probe',
    # R2L
    'guess_passwd':'R2L','ftp_write':'R2L','imap':'R2L','phf':'R2L',
    'multihop':'R2L','warezmaster':'R2L','warezclient':'R2L','spy':'R2L',
    'xlock':'R2L','xsnoop':'R2L','snmpguess':'R2L','snmpgetattack':'R2L',
    'httptunnel':'R2L','sendmail':'R2L','named':'R2L',
    # U2R
    'buffer_overflow':'U2R','loadmodule':'U2R','perl':'U2R','rootkit':'U2R',
    'sqlattack':'U2R','xterm':'U2R','ps':'U2R',
    # Normal
    'normal':'Normal',
}
# Integra il fallback solo per le chiavi mancanti
for k, v in _FALLBACK_MAP.items():
    CATEGORY_MAP.setdefault(k, v)


def map_to_category(attack_type: str) -> str:
    """Converte un attack_type raw nel label multi-classe."""
    return CATEGORY_MAP.get(attack_type.lower().strip(), 'DoS')  # default DoS se sconosciuto


# Costruzione del target multi-classe (usa solo la colonna attack_type, mai come feature)
for df in [df_train, df_test]:
    df['target'] = df['attack_type'].apply(map_to_category)

print("\nDistribuzione classi (train):")
print(df_train['target'].value_counts())
print("\nDistribuzione classi (test):")
print(df_test['target'].value_counts())

# ── Fase 4: Feature Engineering – inferred_attack_category (SENZA leakage) ───
#
# NOTA: questa feature è derivata ESCLUSIVAMENTE da feature di rete osservabili.
# NON usa attack_type né target in alcun modo.
# Le regole implementano in Python la semantica delle SWRL rule dell'ontologia.

R2L_SERVICES = {'ftp', 'ftp_data', 'telnet', 'ssh', 'smtp', 'pop_3',
                'imap4', 'login', 'shell', 'exec', 'finger'}

def infer_category_from_features(row: pd.Series) -> str:
    """
    Inferisce una categoria di traffico usando solo feature di rete.
    Corrisponde alle SWRL rule definite nell'ontologia OWL.

    SWRL Rule 1 (DoS SYN flood):  serror_rate > 0.5 AND count > 10
    SWRL Rule 7 (DoS Smurf):      protocol_type == icmp AND src_bytes > 1000 AND dst_bytes == 0
    SWRL Rule 2 (Probe ICMP):     protocol_type == icmp AND dst_bytes == 0
    SWRL Rule 3 (Probe port scan):diff_srv_rate > 0.5 AND count > 30
    SWRL Rule 5 (U2R root_shell): root_shell == 1
    SWRL Rule 6 (U2R su attempt): su_attempted > 0
    SWRL Rule 4 (R2L failed login): num_failed_logins > 0 AND logged_in == 0
    SWRL Rule 8 (R2L slow brute): service in R2L_SERVICES AND duration > 0 AND logged_in == 0
    """
    protocol  = row['protocol_type']
    service   = row['service']
    src_bytes = row['src_bytes']
    dst_bytes = row['dst_bytes']

    # U2R – priorità alta: segnali di privilege escalation molto specifici
    if row['root_shell'] == 1:
        return 'U2R'
    if row['su_attempted'] > 0:
        return 'U2R'

    # DoS – attacchi volumetrici / SYN flood
    if row['serror_rate'] > 0.5 and row['count'] > 10:
        return 'DoS'
    if protocol == 'icmp' and src_bytes > 1000 and dst_bytes == 0:
        return 'DoS'

    # Probe – scanning / surveillance
    if protocol == 'icmp' and dst_bytes == 0:
        return 'Probe'
    if row['diff_srv_rate'] > 0.5 and row['count'] > 30:
        return 'Probe'

    # R2L – tentativi di accesso remoto falliti
    if row['num_failed_logins'] > 0 and row['logged_in'] == 0:
        return 'R2L'
    if service in R2L_SERVICES and row['duration'] > 0 and row['logged_in'] == 0 and src_bytes < 500:
        return 'R2L'

    return 'Unknown'


print("\n── Feature Engineering (inferred_attack_category) ───────────────────")
df_train['inferred_attack_category'] = df_train.apply(infer_category_from_features, axis=1)
df_test['inferred_attack_category']  = df_test.apply(infer_category_from_features, axis=1)

print("Distribuzione inferred_attack_category (train):")
print(df_train['inferred_attack_category'].value_counts())

# ── Fase 5: Preparazione X / y ────────────────────────────────────────────────

COLS_TO_DROP = ['attack_type', 'difficulty', 'target']

X_train_raw = df_train.drop(columns=COLS_TO_DROP)
X_test_raw  = df_test.drop(columns=COLS_TO_DROP)
y_train_raw = df_train['target']
y_test_raw  = df_test['target']

# Rimozione colonne a varianza zero PRIMA del preprocessore
#   (fit sulla sola X_train per evitare leakage dal test set)
num_cols_tmp = X_train_raw.select_dtypes(include=np.number).columns.tolist()
zero_var_cols = [c for c in num_cols_tmp if X_train_raw[c].nunique() == 1]
if zero_var_cols:
    print(f"\nRimossa/e colonna/e a varianza zero: {zero_var_cols}")
    X_train_raw = X_train_raw.drop(columns=zero_var_cols)
    X_test_raw  = X_test_raw.drop(columns=[c for c in zero_var_cols if c in X_test_raw.columns])

# Colonne finali
numerical_cols   = X_train_raw.select_dtypes(include=np.number).columns.tolist()
categorical_cols = X_train_raw.select_dtypes(include='object').columns.tolist()

print(f"\nFeature numeriche ({len(numerical_cols)}): {numerical_cols[:5]} ...")
print(f"Feature categoriche ({len(categorical_cols)}): {categorical_cols}")

# Codifica del target
label_encoder = LabelEncoder()
y_train = label_encoder.fit_transform(y_train_raw)
y_test  = label_encoder.transform(y_test_raw)
CLASS_NAMES = label_encoder.classes_
print(f"\nClassi (ordine LabelEncoder): {CLASS_NAMES}")

# ── Fase 6: Preprocessore ────────────────────────────────────────────────────

preprocessor = ColumnTransformer(transformers=[
    ('num', StandardScaler(),                  numerical_cols),
    ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols),
])

# ── Fase 7: SMOTE multi-classe ───────────────────────────────────────────────
#
# U2R e R2L hanno pochissimi esempi; k_neighbors deve essere < min_class_count - 1.
# Usiamo k_neighbors=1 che funziona sempre, anche con classi da 2 campioni.

min_class_count = pd.Series(y_train).value_counts().min()
k_neighbors_safe = max(1, min(5, min_class_count - 1))
print(f"\nSMOTE k_neighbors scelto automaticamente: {k_neighbors_safe} "
      f"(classe più piccola: {min_class_count} campioni)")

smote = SMOTE(random_state=42, k_neighbors=k_neighbors_safe)

# ── Fase 8: Funzioni di supporto ─────────────────────────────────────────────

def plot_confusion_matrix(y_true, y_pred, title: str, filename: str):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
    plt.title(title)
    plt.xlabel('Predetto')
    plt.ylabel('Vero')
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()
    print(f"  Matrice di confusione salvata: {filename}")


def evaluate(name: str, y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    f1  = f1_score(y_true, y_pred, average='macro')
    print(f"\n{'─'*60}")
    print(f"  {name}")
    print(f"  Accuracy: {acc:.4f}  |  Macro F1: {f1:.4f}")
    print(classification_report(y_true, y_pred, target_names=CLASS_NAMES))
    return acc, f1


results = {}   # { nome_modello: (accuracy, macro_f1) }

# ── Fase 9: Decision Tree SENZA SMOTE ────────────────────────────────────────

print("\n\n══ Decision Tree SENZA SMOTE ══════════════════════════════════════════")
dt_no_smote = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', DecisionTreeClassifier(random_state=42)),
])
dt_no_smote.fit(X_train_raw, y_train)
y_pred = dt_no_smote.predict(X_test_raw)
acc, f1 = evaluate("Decision Tree – NO SMOTE", y_test, y_pred)
results['DT no SMOTE'] = (acc, f1)
plot_confusion_matrix(y_test, y_pred,
                      'Confusion Matrix – Decision Tree (no SMOTE)',
                      'confusion_matrix_dt_no_smote.png')

# ── Fase 10: Random Forest SENZA SMOTE ───────────────────────────────────────

print("\n\n══ Random Forest SENZA SMOTE ══════════════════════════════════════════")
rf_no_smote = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)),
])
rf_no_smote.fit(X_train_raw, y_train)
y_pred = rf_no_smote.predict(X_test_raw)
acc, f1 = evaluate("Random Forest – NO SMOTE", y_test, y_pred)
results['RF no SMOTE'] = (acc, f1)
plot_confusion_matrix(y_test, y_pred,
                      'Confusion Matrix – Random Forest (no SMOTE)',
                      'confusion_matrix_rf_no_smote.png')

# ── Fase 11: Decision Tree CON SMOTE ─────────────────────────────────────────

print("\n\n══ Decision Tree CON SMOTE ════════════════════════════════════════════")
dt_smote = ImbPipeline([
    ('preprocessor', preprocessor),
    ('smote', smote),
    ('classifier', DecisionTreeClassifier(random_state=42)),
])
dt_smote.fit(X_train_raw, y_train)
y_pred = dt_smote.predict(X_test_raw)
acc, f1 = evaluate("Decision Tree – CON SMOTE", y_test, y_pred)
results['DT SMOTE'] = (acc, f1)
plot_confusion_matrix(y_test, y_pred,
                      'Confusion Matrix – Decision Tree (SMOTE)',
                      'confusion_matrix_dt_smote.png')

# ── Fase 12: Random Forest CON SMOTE ─────────────────────────────────────────

print("\n\n══ Random Forest CON SMOTE ════════════════════════════════════════════")
rf_smote = ImbPipeline([
    ('preprocessor', preprocessor),
    ('smote', smote),
    ('classifier', RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)),
])
rf_smote.fit(X_train_raw, y_train)
y_pred = rf_smote.predict(X_test_raw)
acc, f1 = evaluate("Random Forest – CON SMOTE", y_test, y_pred)
results['RF SMOTE'] = (acc, f1)
plot_confusion_matrix(y_test, y_pred,
                      'Confusion Matrix – Random Forest (SMOTE)',
                      'confusion_matrix_rf_smote.png')

# ── Fase 13: GridSearchCV – Random Forest CON SMOTE ──────────────────────────

print("\n\n══ GridSearchCV – Random Forest CON SMOTE ═════════════════════════════")
param_grid = {
    'classifier__n_estimators': [50, 100, 200],
    'classifier__max_depth':    [10, 20, None],
    'classifier__min_samples_split': [2, 5],
}
gs_pipeline = ImbPipeline([
    ('preprocessor', preprocessor),
    ('smote', smote),
    ('classifier', RandomForestClassifier(random_state=42, n_jobs=-1)),
])
grid_search = GridSearchCV(gs_pipeline, param_grid, cv=3,
                           scoring='f1_macro', n_jobs=-1, verbose=2)
print("Avvio GridSearchCV (scoring=f1_macro) ...")
grid_search.fit(X_train_raw, y_train)

print(f"\nMigliori parametri: {grid_search.best_params_}")
print(f"Miglior F1 macro (CV): {grid_search.best_score_:.4f}")

y_pred = grid_search.best_estimator_.predict(X_test_raw)
acc, f1 = evaluate("Random Forest GridSearchCV (best)", y_test, y_pred)
results['RF GridSearch'] = (acc, f1)
plot_confusion_matrix(y_test, y_pred,
                      'Confusion Matrix – RF GridSearchCV (best)',
                      'confusion_matrix_best_rf_gridsearch.png')

# ── Fase 14: Confronto finale ─────────────────────────────────────────────────

print("\n\n══ CONFRONTO FINALE ═══════════════════════════════════════════════════")
print(f"{'Modello':<25} {'Accuracy':>10} {'Macro F1':>10}")
print("─" * 48)
for name, (acc, f1) in results.items():
    print(f"{name:<25} {acc:>10.4f} {f1:>10.4f}")

# Grafico confronto F1
fig, ax = plt.subplots(figsize=(9, 5))
names = list(results.keys())
f1s   = [v[1] for v in results.values()]
accs  = [v[0] for v in results.values()]
x = np.arange(len(names))
w = 0.35
ax.bar(x - w/2, accs, w, label='Accuracy', color='steelblue')
ax.bar(x + w/2, f1s,  w, label='Macro F1', color='darkorange')
ax.set_xticks(x)
ax.set_xticklabels(names, rotation=15, ha='right')
ax.set_ylim(0, 1.05)
ax.set_title('Confronto modelli – Classificazione multi-classe NSL-KDD')
ax.legend()
plt.tight_layout()
plt.savefig('model_comparison.png')
plt.close()
print("\nGrafico confronto salvato: model_comparison.png")
print("\nDone.")