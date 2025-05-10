import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, auc

# Φόρτωση του dataset
df = pd.read_csv(r'the path to your dataset.csv')

# Εξέταση των πρώτων γραμμών του dataset
print(df.head())

# Μετατροπή των κατηγορικών στηλών σε αριθμητικές τιμές χρησιμοποιώντας LabelEncoder
label_encoder = LabelEncoder()

# Ορισμός των κατηγορικών στηλών
categorical_columns = ['sex', 'cp', 'restecg', 'slope', 'thal', 'dataset']

for column in categorical_columns:
    df[column] = label_encoder.fit_transform(df[column])

# Χωρισμός των δεδομένων σε X (features) και y (στόχος)
X = df.drop(columns=['num'])  # Όλες οι στήλες εκτός από την 'num' (στόχος)
y = (df['num'] > 0).astype(int)  # Δυαδική στόχος (ασθένεια/όχι ασθένεια)

# Διαχωρισμός των δεδομένων σε εκπαιδευτικά και τεστ
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Κλίμακα των δεδομένων (Standardization)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Δημιουργία και εκπαίδευση του μοντέλου
model = LogisticRegression(random_state=42)
model.fit(X_train, y_train)

# Προβλέψεις (πιθανότητες) για την κατηγορία 1
y_pred_prob = model.predict_proba(X_test)[:, 1]  # Αποκτούμε την πιθανότητα της κατηγορίας 1

# Υπολογισμός ROC curve και AUC
fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)
roc_auc = auc(fpr, tpr)

# Σχεδίαση της ROC curve
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='blue', label=f'ROC curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='gray', linestyle='--')  # Διαγώνια γραμμή για τυχαία πρόβλεψη
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc='lower right')
plt.show()
