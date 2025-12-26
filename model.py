import pandas as pd
import numpy as np
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
from sklearn.impute import KNNImputer
from imblearn.over_sampling import SMOTE
import warnings
warnings.filterwarnings('ignore')


# Data importing
df = pd.read_csv("ai_emergency_triage_dataset.csv")
print("Dataset shape:", df.shape)
print("\nFirst few rows:")
print(df.head())

# Data preprocessing
print("\n--- Data Preprocessing ---")
print("Column names:", df.columns.tolist())
print("Data types:\n", df.dtypes)

# Select features and target (adjust column name if needed)
X = df.drop('triage_level', axis=1)  # Features
y = df['triage_level']  # Target

print(f"\nFeatures shape: {X.shape}")
print(f"Target shape: {y.shape}")
print(f"Target distribution:\n{y.value_counts().sort_index()}")

# Check for missing values
print(f"\nMissing values:\n{X.isnull().sum()}")

# Handle missing values using KNN Imputation (better for healthcare)
print("\n--- Handling Missing Values ---")
if X.isnull().sum().sum() > 0:
    print("Using KNN Imputer (5 nearest neighbors)...")
    imputer = KNNImputer(n_neighbors=5)
    X = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)
    print("Missing values handled using KNN Imputation")
else:
    print("No missing values found in data")

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"\nTrain set: {X_train.shape}, Test set: {X_test.shape}")
print(f"\nClass distribution BEFORE SMOTE:")
print(y_train.value_counts().sort_index())

# Handle class imbalance using SMOTE (Critical for healthcare!)
print("\n--- Handling Class Imbalance ---")
print("Applying SMOTE (Synthetic Minority Over-sampling Technique)...")
smote = SMOTE(random_state=42)
X_train, y_train = smote.fit_resample(X_train, y_train)
print(f"Train set after SMOTE: {X_train.shape}")
print(f"\nClass distribution AFTER SMOTE:")
print(pd.Series(y_train).value_counts().sort_index())

# Train individual models
print("\n--- Training Models ---")
print("Training Logistic Regression...")
model1 = LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced')
model1.fit(X_train, y_train)

print("Training Random Forest...")
model2 = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1, class_weight='balanced')
model2.fit(X_train, y_train)

# Create Voting Classifier (combines both models)
print("Creating Voting Classifier...")
voting_clf = VotingClassifier(
    estimators=[('lr', model1), ('rf', model2)],
    voting='soft'
)
voting_clf.fit(X_train, y_train)

# Predictions
print("\n--- Evaluating Models ---")
pred1 = model1.predict(X_test)
pred2 = model2.predict(X_test)
pred_voting = voting_clf.predict(X_test)

# Logistic Regression results
print("\n1. LOGISTIC REGRESSION:")
print(f"   Accuracy:  {accuracy_score(y_test, pred1):.4f}")
print(f"   Precision: {precision_score(y_test, pred1, average='weighted'):.4f}")
print(f"   Recall:    {recall_score(y_test, pred1, average='weighted'):.4f}")
print(f"   F1-Score:  {f1_score(y_test, pred1, average='weighted'):.4f}")
print(f"\n   Per-class metrics:")
print(classification_report(y_test, pred1, digits=4))

# Random Forest results
print("\n2. RANDOM FOREST:")
print(f"   Accuracy:  {accuracy_score(y_test, pred2):.4f}")
print(f"   Precision: {precision_score(y_test, pred2, average='weighted'):.4f}")
print(f"   Recall:    {recall_score(y_test, pred2, average='weighted'):.4f}")
print(f"   F1-Score:  {f1_score(y_test, pred2, average='weighted'):.4f}")
print(f"\n   Per-class metrics:")
print(classification_report(y_test, pred2, digits=4))

# Voting Classifier results
print("\n3. VOTING CLASSIFIER (Combined):")
print(f"   Accuracy:  {accuracy_score(y_test, pred_voting):.4f}")
print(f"   Precision: {precision_score(y_test, pred_voting, average='weighted'):.4f}")
print(f"   Recall:    {recall_score(y_test, pred_voting, average='weighted'):.4f}")
print(f"   F1-Score:  {f1_score(y_test, pred_voting, average='weighted'):.4f}")
print(f"\n   Per-class metrics:")
print(classification_report(y_test, pred_voting, digits=4))

# Summary
print("\n" + "="*50)
print("BEST MODEL:")
best_score = max(
    accuracy_score(y_test, pred1),
    accuracy_score(y_test, pred2),
    accuracy_score(y_test, pred_voting)
)
if accuracy_score(y_test, pred_voting) == best_score:
    print("✓ Voting Classifier (Combined) performs best!")
elif accuracy_score(y_test, pred2) == best_score:
    print("✓ Random Forest performs best!")
else:
    print("✓ Logistic Regression performs best!")
print("="*50)

joblib.dump(model1, "model_lr.pkl", compress=3)
joblib.dump(model2, "model_rf.pkl", compress=3)
joblib.dump(voting_clf, "model_voting.pkl", compress=3)
joblib.dump(imputer, "imputer.pkl", compress=3)
joblib.dump(list(X.columns), "feature_columns.pkl", compress=3)
print("✅ Models and preprocessing objects saved successfully!")

