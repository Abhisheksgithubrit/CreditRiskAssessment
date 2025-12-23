import streamlit as st
import pandas as pd
import joblib
import os
from src.model import train_model

model = None
encoder = None
categorical_cols = None
numerical_cols = None

def run_app():
    global model, encoder, categorical_cols, numerical_cols

    st.title("💳 Credit Risk Assessment App")

    # ------------------ TRAINING ------------------
    st.subheader("📁 Upload Dataset")
    uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

    if uploaded_file:
        data = pd.read_csv(uploaded_file)
        st.write("### Raw Data", data)

        if "Risk" not in data.columns:
            st.error("❌ Dataset must contain a 'Risk' column.")
            return

        X = data.drop("Risk", axis=1)
        y = data["Risk"]

        if st.button("Train Model"):
            accuracy, report, trained_model, trained_encoder = train_model(X, y)

            model = trained_model
            encoder = trained_encoder

            # 🔐 load saved columns AFTER training
            categorical_cols = joblib.load("categorical_cols.pkl")
            numerical_cols = joblib.load("numerical_cols.pkl")

            st.success(f"✅ Model trained with {accuracy * 100:.2f}% accuracy.")
            st.json(report)

    # ------------------ LOAD MODEL ------------------
    if model is None:
        if all(os.path.exists(f) for f in [
            "model.pkl", "encoder.pkl",
            "categorical_cols.pkl", "numerical_cols.pkl"
        ]):
            model = joblib.load("model.pkl")
            encoder = joblib.load("encoder.pkl")
            categorical_cols = joblib.load("categorical_cols.pkl")
            numerical_cols = joblib.load("numerical_cols.pkl")
        else:
            st.warning("⚠️ Train the model first.")
            return

    st.markdown("---")

    # ------------------ PREDICTION ------------------
    st.subheader("🧠 Make a Credit Risk Prediction")

    age = st.number_input("Age", 18, 100, 30)
    income = st.number_input("Income", 0, value=30000)
    credit_score = st.number_input("Credit Score", 300, 900, 650)
    loan_amount = st.number_input("Loan Amount", 0, value=200000)

    gender = st.selectbox("Gender", ["Male", "Female"])
    married = st.selectbox("Married", ["Yes", "No"])
    education = st.selectbox("Education", ["Graduate", "Not Graduate"])
    self_employed = st.selectbox("Self Employed", ["Yes", "No"])

    if st.button("Predict Risk"):
        input_df = pd.DataFrame([{
            "Age": age,
            "Income": income,
            "CreditScore": credit_score,
            "LoanAmount": loan_amount,
            "Gender": gender,
            "Married": married,
            "Education": education,
            "Self_Employed": self_employed
        }])

        # ✅ USE SAVED TRAINING COLUMNS (THIS IS THE FIX)
        input_cat_encoded = encoder.transform(input_df[categorical_cols])

        input_cat_df = pd.DataFrame(
            input_cat_encoded,
            columns=encoder.get_feature_names_out(categorical_cols)
        )

        input_final = pd.concat(
            [
                input_cat_df.reset_index(drop=True),
                input_df[numerical_cols].reset_index(drop=True)
            ],
            axis=1
        )

        prediction = model.predict(input_final)[0]

        if prediction == 1:
            st.error("🔴 High Credit Risk")
        else:
            st.success("🟢 Low Credit Risk")
