import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score, 
    recall_score, f1_score, matthews_corrcoef, confusion_matrix
)

# Page Config
st.set_page_config(page_title="Heart Disease Analysis", layout="wide")

# Colorful UI Styling
st.markdown("""
    <style>
    .metric-card {
        background-color: #1E1E1E;
        padding: 15px;
        border-radius: 10px;
        border-left: 5px solid #FF4B4B;
        margin-bottom: 10px;
    }
    .metric-value { font-size: 24px; font-weight: bold; color: #00FFCC; }
    .metric-label { font-size: 14px; color: #BBBBBB; }
    </style>
    """, unsafe_allow_html=True)

st.title("❤️ Heart Disease Prediction App")

# 1. Download Option
with open("test_sample.csv", "rb") as f:
    st.download_button("⬇️ Step 1: Download Template CSV", f, "test_sample.csv", "text/csv")

# 2. Model Selection (After Download)
try:
    scaler = joblib.load("model/scaler.pkl")
    features = joblib.load("model/features.pkl")
    model_names = ["Logistic Regression", "Decision Tree", "KNN", "Naive Bayes", "Random Forest", "XGBoost"]
    
    st.subheader("🌈 Step 2: Choose your Machine Learning Model")
    selected_name = st.selectbox("", model_names)
    model = joblib.load(f"model/{selected_name.replace(' ', '_')}.pkl")
except:
    st.error("Error: Model files not found. Run 'python train_models.py' first.")
    st.stop()

# 3. File Upload
uploaded_file = st.file_uploader("📤 Step 3: Upload Test CSV", type="csv")

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    
    # Original columns check
    required = ["Age","Sex","ChestPainType","RestingBP","Cholesterol","FastingBS",
                "RestingECG","MaxHR","ExerciseAngina","Oldpeak","ST_Slope","HeartDisease"]
    
    if not all(c in df.columns for c in required):
        st.error(f"Error: Uploaded CSV must contain original columns: {required}")
    else:
        # --- PREPROCESSING ---
        X = df.drop("HeartDisease", axis=1)
        y = df["HeartDisease"]
        X_enc = pd.get_dummies(X, drop_first=True)
        # Force columns to match training exactly
        X_final = X_enc.reindex(columns=features, fill_value=0)
        X_scaled = scaler.transform(X_final)

        # --- PREDICTION ---
        y_pred = model.predict(X_scaled)
        y_prob = model.predict_proba(X_scaled)[:, 1] if hasattr(model, "predict_proba") else y_pred

        # --- METRICS ---
        metrics = {
            "Accuracy": accuracy_score(y, y_pred),
            "AUC Score": roc_auc_score(y, y_prob),
            "Precision": precision_score(y, y_pred),
            "Recall": recall_score(y, y_pred),
            "F1 Score": f1_score(y, y_pred),
            "MCC Score": matthews_corrcoef(y, y_pred)
        }

        # --- DISPLAY RESULTS ---
        st.divider()
        st.header(f"📊 Results for {selected_name}")
        
        # 6 Metrics in 3 Columns
        cols = st.columns(3)
        m_list = list(metrics.items())
        for i in range(6):
            with cols[i % 3]:
                st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-label">{m_list[i][0]}</div>
                        <div class="metric-value">{m_list[i][1]:.3f}</div>
                    </div>
                """, unsafe_allow_html=True)

        # Visualizations
        c1, c2 = st.columns(2)
        with c1:
            st.write("### Confusion Matrix")
            fig, ax = plt.subplots()
            sns.heatmap(confusion_matrix(y, y_pred), annot=True, fmt='d', cmap='Reds', ax=ax)
            st.pyplot(fig)
        with c2:
            st.write("### Prediction vs Actual")
            res_df = pd.DataFrame({'Actual': y, 'Predicted': y_pred})
            st.bar_chart(res_df.apply(pd.Series.value_counts))