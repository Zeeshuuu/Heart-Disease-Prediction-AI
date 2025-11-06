import streamlit as st
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt
import numpy as np

# Load saved files
model = joblib.load("heart_disease_model.pkl")
scaler = joblib.load("scaler.pkl")
feature_names = joblib.load("feature_names.pkl")

st.set_page_config(page_title="❤️ Heart Disease Prediction Dashboard", layout="wide")
st.title("❤️ Heart Disease Prediction Dashboard")
st.write("An interactive AI dashboard to predict heart disease risk and explain model predictions using SHAP.")

# Sidebar input section
st.sidebar.header("🧍‍♂️ Patient Information")

age = st.sidebar.slider("Age", 20, 90, 50)
sex = st.sidebar.selectbox("Sex", ["male", "female"])
cp = st.sidebar.selectbox("Chest Pain Type", ["typical angina", "atypical angina", "non-anginal", "asymptomatic"])
trestbps = st.sidebar.slider("Resting Blood Pressure", 80, 200, 120)
chol = st.sidebar.slider("Cholesterol (mg/dl)", 100, 400, 200)
fbs = st.sidebar.selectbox("Fasting Blood Sugar > 120 mg/dl", ["True", "False"])
restecg = st.sidebar.selectbox("Resting ECG Results", ["normal", "ST-T abnormality", "LVH"])
thalach = st.sidebar.slider("Max Heart Rate Achieved", 70, 220, 150)
exang = st.sidebar.selectbox("Exercise Induced Angina", ["True", "False"])
oldpeak = st.sidebar.slider("ST Depression (oldpeak)", 0.0, 6.0, 1.0)
slope = st.sidebar.selectbox("Slope of Peak Exercise ST Segment", ["upsloping", "flat", "downsloping"])
ca = st.sidebar.slider("Major Vessels Colored by Fluoroscopy (0–3)", 0, 3, 0)
thal = st.sidebar.selectbox("Thalassemia", ["normal", "fixed defect", "reversible defect"])

# Create DataFrame
input_dict = {
    "age": [age],
    "sex": [sex],
    "cp": [cp],
    "trestbps": [trestbps],
    "chol": [chol],
    "fbs": [fbs],
    "restecg": [restecg],
    "thalach": [thalach],
    "exang": [exang],
    "oldpeak": [oldpeak],
    "slope": [slope],
    "ca": [ca],
    "thal": [thal]
}

input_df = pd.DataFrame(input_dict)

# One-hot encode categorical features
df_encoded = pd.get_dummies(input_df)
for col in feature_names:
    if col not in df_encoded.columns:
        df_encoded[col] = 0
df_encoded = df_encoded[feature_names]

# Scale the data
scaled_input = scaler.transform(df_encoded)

# Prediction and SHAP explanation
if st.button("🔍 Predict"):
    prediction = model.predict(scaled_input)[0]
    probability = model.predict_proba(scaled_input)[0][1]

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### 🩺 Prediction Result")
        if prediction == 1:
            st.error(f"⚠️ **High risk** of heart disease.\n\n**Probability:** {probability:.2f}")
        else:
            st.success(f"✅ **Low risk** of heart disease.\n\n**Probability:** {probability:.2f}")

    with col2:
        st.markdown("### 📊 SHAP Explanation")

        # Create SHAP explainer
        explainer = shap.Explainer(model)
        shap_values = explainer(df_encoded)

        # Select only class 1 (heart disease)
        shap_values_class1 = shap_values[:, 1]

        # ✅ Waterfall plot
        st.subheader("💧 SHAP Waterfall Plot (Feature Contribution)")
        fig1, ax1 = plt.subplots(figsize=(8, 6))
        sample_shap = shap_values_class1[0]

        if isinstance(sample_shap.base_values, np.ndarray):
            sample_shap.base_values = float(np.mean(sample_shap.base_values))
        if not hasattr(sample_shap, "data") or isinstance(sample_shap.data, (int, float)):
            sample_shap.data = df_encoded.iloc[0].values

        shap.plots.waterfall(sample_shap, max_display=10, show=False)
        st.pyplot(fig1)

    # --- Global feature importance ---
    st.markdown("---")
    st.subheader("📈 Global Feature Importance (Overall Impact)")

    fig2, ax2 = plt.subplots(figsize=(8, 6))
    try:
        shap.summary_plot(shap_values.values[:, :, 1], df_encoded, plot_type="bar", show=False)
    except Exception:
        shap.summary_plot(shap_values.values, df_encoded, plot_type="bar", show=False)
    st.pyplot(fig2)
