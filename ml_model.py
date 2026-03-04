import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import warnings
warnings.filterwarnings('ignore')

print("Loading training data...")
train_df = pd.read_csv('TRAIN.csv')
print(f"Loaded {train_df.shape[0]} training samples")

feature_columns = [f'F{str(i).zfill(2)}' for i in range(1, 48)]
X = train_df[feature_columns].values
y = train_df['Class'].values

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_val, y_train, y_val = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)

print("Training model...")
model = RandomForestClassifier(
    n_estimators=200,
    max_depth=20,
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)

y_pred = model.predict(X_val)
accuracy = accuracy_score(y_val, y_pred)

print(f"\nValidation Accuracy: {accuracy*100:.2f}%")
print("\nClassification Report:")
print(classification_report(y_val, y_pred, target_names=['Normal', 'Faulty']))

cm = confusion_matrix(y_val, y_pred)
print("Confusion Matrix:")
print(f"[[TN={cm[0,0]:4d}, FP={cm[0,1]:4d}]")
print(f" [FN={cm[1,0]:4d}, TP={cm[1,1]:4d}]]")

cv_scores = cross_val_score(model, X_scaled, y, cv=5, scoring='accuracy')
print(f"\n5-Fold CV Accuracy: {cv_scores.mean()*100:.2f}% (+/- {cv_scores.std()*100:.2f}%)")

print("\nRetraining on full data...")
model.fit(X_scaled, y)

print("Loading test data...")
test_df = pd.read_csv('TEST.csv')
test_ids = test_df['ID'].values
X_test = test_df[feature_columns].values
X_test_scaled = scaler.transform(X_test)

predictions = model.predict(X_test_scaled)

output_df = pd.DataFrame({'ID': test_ids, 'CLASS': predictions})
output_df.to_csv('FINAL.csv', index=False)

print(f"\nSaved {len(output_df)} predictions to FINAL.csv")
print(f"Normal: {(predictions == 0).sum()}, Faulty: {(predictions == 1).sum()}")
print("\nDone!")
