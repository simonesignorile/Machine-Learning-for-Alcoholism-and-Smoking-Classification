import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    recall_score,
    accuracy_score,
    precision_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE

# Caricamento del dataset
df = pd.read_csv("smoking_driking_dataset_Ver01.csv")

# Codifica delle colonne 'sex' e 'DRK_YN'
labelencoder_sex = LabelEncoder()
df['sex'] = labelencoder_sex.fit_transform(df['sex'])

labelencoder_drink = LabelEncoder()
df['DRK_YN'] = labelencoder_drink.fit_transform(df['DRK_YN'])

# Selezione delle features e dei target per il modello del fumatore
X_smoke = df.drop(columns=['DRK_YN', 'SMK_stat_type_cd'])
y_smoke = df['SMK_stat_type_cd']

# Calcolo IQR solo per il modello del fumatore
Q1_smoke = X_smoke.quantile(0.25)
Q3_smoke = X_smoke.quantile(0.75)
IQR_smoke = Q3_smoke - Q1_smoke

# Rimozione dei valori anomali solo per il modello del fumatore
X_smoke = X_smoke[~((X_smoke < (Q1_smoke - 1.5 * IQR_smoke)) | (X_smoke > (Q3_smoke + 1.5 * IQR_smoke))).any(axis=1)]
y_smoke = y_smoke[X_smoke.index]

# Applicazione di SMOTE
smote = SMOTE(random_state=42)
X_smoke_res, y_smoke_res = smote.fit_resample(X_smoke, y_smoke)

# Divisione del dataset in set di addestramento e test per il modello del fumatore
X_train_smoke, X_test_smoke, y_train_smoke, y_test_smoke = train_test_split(X_smoke_res, y_smoke_res, test_size=0.2, random_state=42, stratify=y_smoke_res)

# Creazione del modello Random Forest con i parametri ottimizzati per il fumatore
model_rf_smoke_opt = RandomForestClassifier(n_estimators=200, max_depth=None, min_samples_split=10)

# Addestramento del modello ottimizzato per il fumatore
model_rf_smoke_opt.fit(X_train_smoke, y_train_smoke)

# Predizione del modello ottimizzato sul set di test per il fumatore
y_pred_rf_test_smoke_opt = model_rf_smoke_opt.predict(X_test_smoke)

# Valutazione del modello ottimizzato del fumatore sul set di test
print("\nTest Set Evaluation for Smoker Model:")
print("\nClassification Report:\n", classification_report(y_test_smoke, y_pred_rf_test_smoke_opt))

recall_rf_test_smoke_opt = recall_score(y_test_smoke, y_pred_rf_test_smoke_opt, average='weighted')
accuracy_rf_test_smoke_opt = accuracy_score(y_test_smoke, y_pred_rf_test_smoke_opt)
precision_rf_test_smoke_opt = precision_score(y_test_smoke, y_pred_rf_test_smoke_opt, average='weighted')

print("\nRecall: ", recall_rf_test_smoke_opt)
print("Accuracy: ", accuracy_rf_test_smoke_opt)
print("Precision: ", precision_rf_test_smoke_opt)

# Matrice di Confusione Normalizzata per il modello del fumatore
cm_smoke = confusion_matrix(y_test_smoke, y_pred_rf_test_smoke_opt)
cm_norm_smoke = cm_smoke.astype('float') / cm_smoke.sum(axis=1)[:, np.newaxis]

plt.figure(figsize=(10, 7))
sns.heatmap(cm_norm_smoke, annot=True, fmt=".2f", cmap="Blues")
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.title('Normalized Confusion Matrix for Smoker Model')
plt.show()

# Selezione delle features e dei target per il modello dell'alcolista
X_drink = df.drop(columns=['SMK_stat_type_cd', 'DRK_YN'])
y_drink = df['DRK_YN']

# Divisione del dataset in set di addestramento e test per il modello dell'alcolista
X_train_drink, X_test_drink, y_train_drink, y_test_drink = train_test_split(X_drink, y_drink, test_size=0.2, random_state=42, stratify=y_drink)

# Creazione del modello Random Forest con i parametri ottimizzati per l'alcolista
model_rf_drink_opt = RandomForestClassifier(n_estimators=200, max_depth=None, min_samples_split=10)

# Addestramento del modello ottimizzato per l'alcolista
model_rf_drink_opt.fit(X_train_drink, y_train_drink)

# Predizione del modello ottimizzato sul set di test per l'alcolista
y_pred_rf_test_drink_opt = model_rf_drink_opt.predict(X_test_drink)

# Valutazione del modello ottimizzato dell'alcolista sul set di test
print("\nTest Set Evaluation for Drinker Model:")
print("\nClassification Report:\n", classification_report(y_test_drink, y_pred_rf_test_drink_opt))

recall_rf_test_drink_opt = recall_score(y_test_drink, y_pred_rf_test_drink_opt, average='weighted')
accuracy_rf_test_drink_opt = accuracy_score(y_test_drink, y_pred_rf_test_drink_opt)
precision_rf_test_drink_opt = precision_score(y_test_drink, y_pred_rf_test_drink_opt, average='weighted')

print("\nRecall: ", recall_rf_test_drink_opt)
print("Accuracy: ", accuracy_rf_test_drink_opt)
print("Precision: ", precision_rf_test_drink_opt)

# Matrice di Confusione Normalizzata per il modello dell'alcolista
cm_drink = confusion_matrix(y_test_drink, y_pred_rf_test_drink_opt)
cm_norm_drink = cm_drink.astype('float') / cm_drink.sum(axis=1)[:, np.newaxis]

plt.figure(figsize=(10, 7))
sns.heatmap(cm_norm_drink, annot=True, fmt=".2f", cmap="Blues")
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.title('Normalized Confusion Matrix for Drinker Model')
plt.show()
