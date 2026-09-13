"""Streamlit UI for the AI Financial Risk & Credit Scoring Platform."""
from pathlib import Path
import pandas as pd
import streamlit as st
import joblib

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "models" / "credit_risk_model.joblib"

st.set_page_config(page_title="AI Credit Risk", page_icon="💳", layout="wide")
st.title("💳 AI Financial Risk & Credit Scoring")
st.caption("Educational portfolio demo using synthetic credit-risk data")

if not MODEL_PATH.exists():
    st.error("Model not found. Run: python src/generate_data.py && python src/train.py")
    st.stop()

model = joblib.load(MODEL_PATH)

with st.sidebar:
    st.header("Applicant Profile")
    age = st.slider("Age", 21, 65, 35)
    annual_income = st.number_input("Annual income", 180_000, 8_000_000, 900_000, step=25_000)
    employment_years = st.slider("Employment years", 0.0, 40.0, 7.0, 0.5)
    debt_to_income = st.slider("Debt-to-income ratio", 0.02, 0.85, 0.30, 0.01)
    credit_history_years = st.slider("Credit history years", 0.5, 35.0, 8.0, 0.5)
    num_open_accounts = st.slider("Open accounts", 1, 18, 5)
    num_delinquencies = st.slider("Delinquencies", 0, 8, 0)
    loan_amount = st.number_input("Loan amount", 25_000, 2_500_000, 400_000, step=25_000)
    loan_term_months = st.selectbox("Loan term (months)", [12, 24, 36, 48, 60, 72], index=2)
    savings_balance = st.number_input("Savings balance", 5_000, 5_000_000, 500_000, step=25_000)
    previous_defaults = st.slider("Previous defaults", 0, 4, 0)
    utilization_ratio = st.slider("Credit utilization", 0.01, 0.99, 0.30, 0.01)
    employment_type = st.selectbox("Employment type", ["Salaried", "Self-employed", "Contract", "Unemployed"])
    home_ownership = st.selectbox("Home ownership", ["Owned", "Mortgage", "Rented"])
    loan_purpose = st.selectbox("Loan purpose", ["Home", "Education", "Medical", "Business", "Personal", "Vehicle"])

row = pd.DataFrame([{
    "age": age, "annual_income": annual_income, "employment_years": employment_years,
    "debt_to_income": debt_to_income, "credit_history_years": credit_history_years,
    "num_open_accounts": num_open_accounts, "num_delinquencies": num_delinquencies,
    "loan_amount": loan_amount, "loan_term_months": loan_term_months,
    "savings_balance": savings_balance, "previous_defaults": previous_defaults,
    "utilization_ratio": utilization_ratio, "employment_type": employment_type,
    "home_ownership": home_ownership, "loan_purpose": loan_purpose,
}])

probability = float(model.predict_proba(row)[0, 1])
score = int(round(850 - 550 * probability))
if score >= 750:
    band = "Low Risk"
elif score >= 650:
    band = "Moderate Risk"
elif score >= 550:
    band = "High Risk"
else:
    band = "Very High Risk"

c1, c2, c3 = st.columns(3)
c1.metric("Probability of Default", f"{probability:.1%}")
c2.metric("Estimated Credit Score", score)
c3.metric("Risk Band", band)

st.progress(probability, text="Modeled probability of elevated default risk")

st.subheader("Decision Support")
if probability < 0.25:
    st.success("The model estimates relatively low risk for this applicant profile.")
elif probability < 0.50:
    st.warning("The model estimates moderate risk. Review affordability and credit history carefully.")
else:
    st.error("The model estimates elevated risk. Additional underwriting review is recommended.")

st.subheader("Applicant Inputs")
st.dataframe(row.T.rename(columns={0: "Value"}), use_container_width=True)

st.info("This score is a model-derived portfolio metric, not an official credit score. Do not use this demo for real lending decisions.")
