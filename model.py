# ===============================
# 1. Import Required Libraries
# ===============================

import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

from sklearn.impute import KNNImputer

from xgboost import XGBClassifier

import warnings
warnings.filterwarnings("ignore")

# ===============================
# 2. Load Dataset
# ===============================

df = pd.read_csv("synthetic_500k_bias_safe.csv")

print("Dataset shape:", df.shape)
print(df.head())

# ===============================
# 3. Split Features & Target
# ===============================

X = df.drop("triage_level", axis=1)
y = df["triage_level"]

print("\nTarget distribution:")
print(y.value_counts().sort_index())

# ===============================
# 4. Handle Missing Values
# ===============================

imputer = KNNImputer(n_neighbors=5)
X = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)

# ===============================
# 5. Train–Test Split
# ===============================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# ===============================
# 6. Define XGBoost Model
# ===============================

xgb_model = XGBClassifier(
    n_estimators=300,
    max_depth=7,
    learning_rate=0.05,

    subsample=0.8,
    colsample_bytree=0.8,

    objective="multi:softmax",
    num_class=4,

    eval_metric="mlogloss",
    random_state=42,
    n_jobs=-1
)

# ===============================
# 7. Train the Model
# ===============================

print("\nTraining XGBoost model...")
xgb_model.fit(X_train, y_train)
print("Training completed.")

# ===============================
# 8. Evaluate Model
# ===============================

y_pred = xgb_model.predict(X_test)

print("\nClassification Report:")
print(classification_report(y_test, y_pred, digits=4))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# ===============================
# 9. Save Model & Preprocessing
# ===============================

joblib.dump(xgb_model, "xgboost_triage_model.pkl")
joblib.dump(imputer, "knn_imputer.pkl", compress=3)
joblib.dump(list(X.columns), "feature_columns.pkl")

print("\n✅ Model and preprocessing objects saved!")
