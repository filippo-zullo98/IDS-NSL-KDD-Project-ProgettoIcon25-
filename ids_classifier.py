# ids_classifier.py - Codice completo e aggiornato

# Fase 1: Importazioni e Caricamento Dati

import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.model_selection import train_test_split, GridSearchCV # Assicurati che GridSearchCV sia importato
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline # Usiamo questo Pipeline per i modelli SENZA SMOTE
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Definiamo i nomi delle colonne per il dataset NSL-KDD
feature_names = [
    'duration', 'protocol_type', 'service', 'flag', 'src_bytes', 'dst_bytes',
    'land', 'wrong_fragment', 'urgent', 'hot', 'num_failed_logins',
    'logged_in', 'num_compromised', 'root_shell', 'su_attempted', 'num_root',
    'num_file_creations', 'num_shells', 'num_access_files', 'num_outbound_cmds',
    'is_host_login', 'is_guest_login', 'count', 'srv_count', 'serror_rate',
    'srv_serror_rate', 'rerror_rate', 'srv_rerror_rate', 'same_srv_rate',
    'diff_srv_rate', 'srv_diff_host_rate', 'dst_host_count', 'dst_host_srv_count',
    'dst_host_same_srv_rate', 'dst_host_diff_srv_rate', 'dst_host_same_src_port_rate',
    'dst_host_srv_diff_host_rate', 'dst_host_serror_rate', 'dst_host_srv_serror_rate',
    'dst_host_rerror_rate', 'dst_host_srv_rerror_rate', 'attack_type', 'difficulty'
]

# Carichiamo il dataset di training
try:
    df_train = pd.read_csv('KDDTrain+.txt', names=feature_names)
    print("Dataset di training caricato con successo.")
except FileNotFoundError:
    print("Errore: KDDTrain+.txt non trovato. Assicurati che i file del dataset siano nella stessa directory dello script.")
    exit()

# Carichiamo il dataset di test
try:
    df_test = pd.read_csv('KDDTest+.txt', names=feature_names)
    print("Dataset di test caricato con successo.")
except FileNotFoundError:
    print("Errore: KDDTest+.txt non trovato. Assicurati che i file del dataset siano nella stessa directory dello script.")
    exit()

print("\n--- Informazioni sul Dataset di Training ---")
print(df_train.head())
print("\nShape del dataset di training:", df_train.shape)
print("\nStatistiche descrittive:")
print(df_train.describe())

print("\n--- Distribuzione dei Tipi di Attacco nel Training Set ---")
print(df_train['attack_type'].value_counts())

# Mappa tutti gli attacchi alla categoria 'attack' e 'normal' a 'normal'.
# Questo rende il problema una classificazione binaria.
df_train['target'] = df_train['attack_type'].apply(lambda x: 'normal' if x == 'normal' else 'attack')
df_test['target'] = df_test['attack_type'].apply(lambda x: 'normal' if x == 'normal' else 'attack')

print("\n--- Distribuzione delle Classi (Normal/Attack) nel Training Set ---")
print(df_train['target'].value_counts())

# Fase 2: Pre-elaborazione Dati (Feature Engineering e Trasformazioni)

# Identifica le colonne numeriche e categoriche
numerical_cols = df_train.select_dtypes(include=np.number).columns.tolist()
categorical_cols = df_train.select_dtypes(include='object').columns.tolist()
# Rimuovi le colonne target originali e quelle inutilizzate dal set di feature
if 'attack_type' in categorical_cols:
    categorical_cols.remove('attack_type')
if 'target' in categorical_cols: # Assicurati che 'target' non sia tra le features
    categorical_cols.remove('target')

# Identifica le feature (X) e la variabile target (y)
X_train = df_train.drop(['attack_type', 'difficulty', 'target'], axis=1)
y_train = df_train['target']
X_test = df_test.drop(['attack_type', 'difficulty', 'target'], axis=1)
y_test = df_test['target']

# Rimuovi colonne con varianza zero (se presenti)
cols_to_drop_zero_variance = []
for col in X_train.columns:
    if X_train[col].nunique() == 1:
        cols_to_drop_zero_variance.append(col)

if cols_to_drop_zero_variance:
    print(f"\nRimozione colonne con varianza zero: {cols_to_drop_zero_variance}")
    X_train = X_train.drop(columns=cols_to_drop_zero_variance)
    X_test = X_test.drop(columns=cols_to_drop_zero_variance)
    # Aggiorna le liste delle colonne dopo la rimozione
    numerical_cols = X_train.select_dtypes(include=np.number).columns.tolist()
    categorical_cols = X_train.select_dtypes(include='object').columns.tolist()


preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_cols),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
    ])

# Codifica la variabile target (y)
# 'normal' -> 0, 'attack' -> 1 (l'ordine dipende da LabelEncoder)
label_encoder = LabelEncoder()
y_train_encoded = label_encoder.fit_transform(y_train)
y_test_encoded = label_encoder.transform(y_test)

print("\n--- Anteprima delle Feature dopo la Preparazione (prima della trasformazione finale) ---")
print(X_train.head())
print("\nTarget codificato (prime 5):", y_train_encoded[:5])
print("Mapping classi:", label_encoder.classes_)


# Fase 3: Costruzione delle Pipeline e Addestramento dei Modelli

# --- Modelli SENZA SMOTE ---

print("\n--- Addestramento del Modello (Decision Tree SENZA SMOTE) ---")
model_dt_no_smote = Pipeline(steps=[('preprocessor', preprocessor),
                                    ('classifier', DecisionTreeClassifier(random_state=42))])
model_dt_no_smote.fit(X_train, y_train_encoded)
y_pred_dt_no_smote = model_dt_no_smote.predict(X_test)
print("Decision Tree SENZA SMOTE addestrato.")
print("Accuracy (DT senza SMOTE):", accuracy_score(y_test_encoded, y_pred_dt_no_smote))

# Visualizza e salva la Matrice di Confusione (Opzionale: puoi commentare se non vuoi 4 grafici)
cm_dt_no_smote = confusion_matrix(y_test_encoded, y_pred_dt_no_smote)
plt.figure(figsize=(8, 6))
sns.heatmap(cm_dt_no_smote, annot=True, fmt='d', cmap='Blues',
            xticklabels=label_encoder.classes_,
            yticklabels=label_encoder.classes_)
plt.title('Matrice di Confusione - Decision Tree SENZA SMOTE')
plt.xlabel('Predetto')
plt.ylabel('Vero')
plt.savefig('confusion_matrix_dt_no_smote.png')
plt.close()


print("\n--- Addestramento del Modello (Random Forest SENZA SMOTE) ---")
model_rf_no_smote = Pipeline(steps=[('preprocessor', preprocessor),
                                    ('classifier', RandomForestClassifier(random_state=42, n_estimators=100))])
model_rf_no_smote.fit(X_train, y_train_encoded)
y_pred_rf_no_smote = model_rf_no_smote.predict(X_test)
print("Random Forest SENZA SMOTE addestrato.")
print("Accuracy (RF senza SMOTE):", accuracy_score(y_test_encoded, y_pred_rf_no_smote))

# Visualizza e salva la Matrice di Confusione (Opzionale)
cm_rf_no_smote = confusion_matrix(y_test_encoded, y_pred_rf_no_smote)
plt.figure(figsize=(8, 6))
sns.heatmap(cm_rf_no_smote, annot=True, fmt='d', cmap='Blues',
            xticklabels=label_encoder.classes_,
            yticklabels=label_encoder.classes_)
plt.title('Matrice di Confusione - Random Forest SENZA SMOTE')
plt.xlabel('Predetto')
plt.ylabel('Vero')
plt.savefig('confusion_matrix_rf_no_smote.png')
plt.close()

# --- Modelli CON SMOTE ---

print("\n--- Addestramento del Modello (Decision Tree con SMOTE) ---")
model_dt_smote = ImbPipeline(steps=[('preprocessor', preprocessor),
                                     ('smote', SMOTE(random_state=42)),
                                     ('classifier', DecisionTreeClassifier(random_state=42))])

model_dt_smote.fit(X_train, y_train_encoded)
print("Decision Tree con SMOTE addestrato.")

y_pred_dt_smote = model_dt_smote.predict(X_test)

print("\n--- Valutazione del Modello Decision Tree con SMOTE ---")
print("Accuracy:", accuracy_score(y_test_encoded, y_pred_dt_smote))
print("\nClassification Report:")
print(classification_report(y_test_encoded, y_pred_dt_smote, target_names=label_encoder.classes_))

cm_dt_smote = confusion_matrix(y_test_encoded, y_pred_dt_smote)
plt.figure(figsize=(8, 6))
sns.heatmap(cm_dt_smote, annot=True, fmt='d', cmap='Blues',
            xticklabels=label_encoder.classes_,
            yticklabels=label_encoder.classes_)
plt.title('Matrice di Confusione - Decision Tree con SMOTE')
plt.xlabel('Predetto')
plt.ylabel('Vero')
plt.savefig('confusion_matrix_dt_smote.png')
plt.close()


print("\n--- Addestramento del Modello (Random Forest con SMOTE) ---")
model_rf_smote = ImbPipeline(steps=[('preprocessor', preprocessor),
                                    ('smote', SMOTE(random_state=42)),
                                    ('classifier', RandomForestClassifier(random_state=42, n_estimators=100))])

model_rf_smote.fit(X_train, y_train_encoded)
print("Random Forest con SMOTE addestrato.")

y_pred_rf_smote = model_rf_smote.predict(X_test) # Rinominata per chiarezza, prima era y_pred_rf

print("\n--- Valutazione del Modello Random Forest con SMOTE ---")
print("Accuracy:", accuracy_score(y_test_encoded, y_pred_rf_smote))
print("\nClassification Report:")
print(classification_report(y_test_encoded, y_pred_rf_smote, target_names=label_encoder.classes_))

cm_rf_smote = confusion_matrix(y_test_encoded, y_pred_rf_smote)
plt.figure(figsize=(8, 6))
sns.heatmap(cm_rf_smote, annot=True, fmt='d', cmap='Blues',
            xticklabels=label_encoder.classes_,
            yticklabels=label_encoder.classes_)
plt.title('Matrice di Confusione - Random Forest con SMOTE')
plt.xlabel('Predetto')
plt.ylabel('Vero')
plt.savefig('confusion_matrix_rf_smote.png')
plt.close()


# --- Ottimizzazione Iperparametri con GridSearchCV (Random Forest con SMOTE) ---

print("\n--- Ottimizzazione Iperparametri con GridSearchCV (Random Forest con SMOTE) ---")

param_grid_rf = {
    'classifier__n_estimators': [50, 100, 200],
    'classifier__max_depth': [10, 20, None],
    'classifier__min_samples_split': [2, 5]
}

pipeline_rf_gs = ImbPipeline(steps=[('preprocessor', preprocessor),
                                    ('smote', SMOTE(random_state=42)),
                                    ('classifier', RandomForestClassifier(random_state=42))])

grid_search_rf = GridSearchCV(pipeline_rf_gs, param_grid_rf, cv=3, scoring='f1', n_jobs=-1, verbose=2)

print("Avvio GridSearchCV...")
grid_search_rf.fit(X_train, y_train_encoded)
print("GridSearchCV completato.")

print("\n--- Risultati GridSearchCV per Random Forest ---")
print("Migliori parametri trovati:", grid_search_rf.best_params_)
print("Miglior F1-score con cross-validation:", grid_search_rf.best_score_)

best_rf_model = grid_search_rf.best_estimator_
y_pred_best_rf = best_rf_model.predict(X_test)

print("\n--- Valutazione del Miglior Modello Random Forest (GridSearchCV) ---")
print("Accuracy:", accuracy_score(y_test_encoded, y_pred_best_rf))
print("\nClassification Report:")
print(classification_report(y_test_encoded, y_pred_best_rf, target_names=label_encoder.classes_))

cm_best_rf = confusion_matrix(y_test_encoded, y_pred_best_rf)
plt.figure(figsize=(8, 6))
sns.heatmap(cm_best_rf, annot=True, fmt='d', cmap='Blues',
            xticklabels=label_encoder.classes_,
            yticklabels=label_encoder.classes_)
plt.title('Matrice di Confusione - Miglior Random Forest (GridSearchCV)')
plt.xlabel('Predetto')
plt.ylabel('Vero')
plt.savefig('confusion_matrix_best_rf_gridsearch.png')
plt.close()

# --- Confronto Finale dei Risultati ---

print("\n--- Confronto dei Risultati (Accuracy) ---")
print("Decision Tree Accuracy (SENZA SMOTE):", accuracy_score(y_test_encoded, y_pred_dt_no_smote))
print("Decision Tree Accuracy (CON SMOTE):", accuracy_score(y_test_encoded, y_pred_dt_smote))
print("Random Forest Accuracy (SENZA SMOTE):", accuracy_score(y_test_encoded, y_pred_rf_no_smote))
print("Random Forest Accuracy (CON SMOTE):", accuracy_score(y_test_encoded, y_pred_rf_smote))
print("Random Forest Accuracy (GridSearchCV - Ottimizzato):", accuracy_score(y_test_encoded, y_pred_best_rf))
