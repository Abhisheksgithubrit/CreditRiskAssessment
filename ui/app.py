import streamlit as st
import pandas as pd
import joblib
import os
from src.model import train_model

model = None
encoder = None

def run_app():
    global model, encoder

    st.title("💳 Credit Risk Assessment App")

    # Upload CSV
    st.subheader("📁 Upload Dataset")
    uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

    if uploaded_file:
        data = pd.read_csv(uploaded_file)
        st.write("### Raw Data", data)

        # Separate features and label
        X = data.drop("Risk", axis=1)
        y = data["Risk"]

        if st.button("Train Model"):
            accuracy, report, trained_model, trained_encoder = train_model(X, y)

            # Save the trained model and encoder
            joblib.dump(trained_model, "model.pkl")
            joblib.dump(trained_encoder, "encoder.pkl")

            model = trained_model
            encoder = trained_encoder

            st.success(f"✅ Model trained successfully with {accuracy * 100:.2f}% accuracy.")
            st.json(report)

    # Load model/encoder if already saved
    if model is None or encoder is None:
        if os.path.exists("model.pkl") and os.path.exists("encoder.pkl"):
            model = joblib.load("model.pkl")
            encoder = joblib.load("encoder.pkl")
        else:
            st.warning("⚠️ No trained model found. Please upload data and train first.")
            return

    st.markdown("---")

    # Live Prediction
    st.subheader("🧠 Make a Credit Risk Prediction")

    age = st.number_input("Age", 18, 100)
    income = st.number_input("Income", 0)
    credit_score = st.number_input("Credit Score", 300, 900)
    loan_amount = st.number_input("Loan Amount", 0)
    gender = st.selectbox("Gender", ["Male", "Female"])
    married = st.selectbox("Married", ["Yes", "No"])
    education = st.selectbox("Education", ["Graduate", "Not Graduate"])
    self_employed = st.selectbox("Self Employed", ["Yes", "No"])

    if st.button("Predict Risk"):
        input_df = pd.DataFrame([[age, income, credit_score, loan_amount, gender, married, education, self_employed]],
                                columns=["Age", "Income", "CreditScore", "LoanAmount", "Gender", "Married", "Education", "Self_Employed"])

        input_encoded = encoder.transform(input_df)
        prediction = model.predict(input_encoded)

        st.success(f"🎯 Predicted Risk: {'High' if prediction[0] == 1 else 'Low'}")
