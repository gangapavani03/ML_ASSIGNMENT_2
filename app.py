import streamlit as st
import pandas as pd
import joblib
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, f1_score, matthews_corrcoef, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

st.set_page_config(page_title="Heart Disease Prediction", layout="wide")

st.title("❤️ Heart Disease Prediction App")
st.caption("ML-based clinical decision support demo")
st.write("Select a model and upload a CSV file to evaluate predictions.")
st.markdown(
    "<style> .stApp { background-color: #0f172a; color: #e5e7eb; } </style>",
    unsafe_allow_html=True
)

# Load models
models = {
    "Logistic Regression": joblib.load("model/Logistic_Regression.pkl"),
    "Decision Tree": joblib.load("model/Decision_Tree.pkl"),
    "KNN": joblib.load("model/KNN.pkl"),
    "Naive Bayes": joblib.load("model/Naive_Bayes.pkl"),
    "Random Forest": joblib.load("model/Random_Forest.pkl"),
    "XGBoost": joblib.load("model/XGBoost.pkl")
}

model_name = st.selectbox("Choose a model", list(models.keys()))
model = models[model_name]

uploaded_file = st.file_uploader("Upload test CSV", type="csv")

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)

    if "HeartDisease" not in data.columns:
        st.error("CSV must contain HeartDisease column.")
    else:
        X = data.drop("HeartDisease", axis=1)
        y = data["HeartDisease"]

        X = pd.get_dummies(X, drop_first=True)

        y_pred = model.predict(X)
        y_prob = model.predict_proba(X)[:, 1] if hasattr(model, "predict_proba") else y_pred

        acc = accuracy_score(y, y_pred)
        auc = roc_auc_score(y, y_prob)
        prec = precision_score(y, y_pred)
        rec = recall_score(y, y_pred)
        f1 = f1_score(y, y_pred)
        mcc = matthews_corrcoef(y, y_pred)

        st.subheader("📊 Performance Metrics")
        st.write(f"Accuracy: {acc:.3f}")
        st.write(f"AUC: {auc:.3f}")
        st.write(f"Precision: {prec:.3f}")
        st.write(f"Recall: {rec:.3f}")
        st.write(f"F1: {f1:.3f}")
        st.write(f"MCC: {mcc:.3f}")

        cm = confusion_matrix(y, y_pred)
        st.subheader("Confusion Matrix")

        fig, ax = plt.subplots()
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        st.pyplot(fig)
