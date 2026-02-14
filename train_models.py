import pandas as pd
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score,
    recall_score, f1_score, matthews_corrcoef
)
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

# Create folder for models
if not os.path.exists("model"):
    os.makedirs("model")

# 1. Load Data
df = pd.read_csv("heart.csv")

# 2. Split Original Data (to save a clean test_sample.csv for the app)
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
test_df.to_csv("test_sample.csv", index=False)

# 3. Preprocessing Function
def preprocess(data):
    X = data.drop("HeartDisease", axis=1)
    y = data["HeartDisease"]
    X = pd.get_dummies(X, drop_first=True)
    return X, y

X_train_raw, y_train = preprocess(train_df)
X_test_raw, y_test = preprocess(test_df)

# Align columns
X_test_raw = X_test_raw.reindex(columns=X_train_raw.columns, fill_value=0)

# 4. Scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train_raw)
X_test = scaler.transform(X_test_raw)

# Save artifacts
joblib.dump(scaler, "model/scaler.pkl")
joblib.dump(X_train_raw.columns.tolist(), "model/features.pkl")

# 5. Train & Print 6 Metrics
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "KNN": KNeighborsClassifier(),
    "Naive Bayes": GaussianNB(),
    "Random Forest": RandomForestClassifier(random_state=42),
    "XGBoost": XGBClassifier(eval_metric="logloss")
}

print(f"{'Model':<20} | Acc   | AUC   | Prec  | Rec   | F1    | MCC")
print("-" * 70)

for name, model in models.items():
    model.fit(X_train, y_train)
    joblib.dump(model, f"model/{name.replace(' ', '_')}.pkl")
    
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else y_pred
    
    metrics = [
        accuracy_score(y_test, y_pred),
        roc_auc_score(y_test, y_prob),
        precision_score(y_test, y_pred),
        recall_score(y_test, y_pred),
        f1_score(y_test, y_pred),
        matthews_corrcoef(y_test, y_pred)
    ]
    
    print(f"{name:<20} | " + " | ".join([f"{m:.3f}" for m in metrics]))

print("\n✅ All models and preprocessing files saved in /model folder.")