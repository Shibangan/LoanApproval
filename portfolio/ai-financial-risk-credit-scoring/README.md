# 💳 AI Financial Risk & Credit Scoring Platform

> End-to-end machine learning system for credit risk assessment, explainable predictions, and interactive deployment.

![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-ML-F7931E?logo=scikit-learn&logoColor=white)
![SHAP](https://img.shields.io/badge/Explainability-SHAP-FF4B4B)
![Streamlit](https://img.shields.io/badge/App-Streamlit-FF4B4B?logo=streamlit&logoColor=white)
![Docker](https://img.shields.io/badge/Deployment-Docker-2496ED?logo=docker&logoColor=white)

## 🎯 Project Overview

This project simulates a modern credit-risk workflow: transform applicant financial data into a **risk probability**, convert it into a **credit score**, and explain the decision with model-level and applicant-level insights.

### What it demonstrates

- Data generation and validation
- Exploratory risk analysis
- Feature engineering
- Imbalanced classification handling
- Random Forest risk model
- Probability calibration
- ROC-AUC, PR-AUC, F1 and confusion matrix evaluation
- SHAP-based explainability
- Credit score mapping
- Interactive Streamlit deployment
- Docker-ready packaging

> **Important:** This is an educational portfolio project using synthetic data. It is not a production lending system and should not be used to make real credit decisions.

## 🧠 Architecture

```text
Applicant Data
      ↓
Validation + Feature Engineering
      ↓
Preprocessing Pipeline
      ↓
Risk Classification Model
      ↓
Probability of Default
      ↓
Credit Score + Risk Band
      ↓
SHAP Explanation
      ↓
Streamlit Dashboard / Docker
```

## 📊 Input Features

| Feature | Description |
|---|---|
| age | Applicant age |
| annual_income | Annual income |
| employment_years | Years employed |
| debt_to_income | Monthly debt / monthly income |
| credit_history_years | Length of credit history |
| num_open_accounts | Number of open credit accounts |
| num_delinquencies | Historical delinquencies |
| loan_amount | Requested loan amount |
| loan_term_months | Loan duration |
| savings_balance | Liquid savings |
| previous_defaults | Previous default count |
| utilization_ratio | Revolving credit utilization |
| employment_type | Employment category |
| home_ownership | Housing status |
| loan_purpose | Purpose of the loan |

Target: `default_risk` where `1` represents elevated modeled default risk.

## 🏆 Risk Bands

The model probability is mapped to an interpretable score:

```text
Probability of Default → Credit Score
0.00                  → 850
0.50                  → 650
1.00                  → 300
```

| Score | Risk Band |
|---:|---|
| 750–850 | Low |
| 650–749 | Moderate |
| 550–649 | High |
| 300–549 | Very High |

## 🚀 Run Locally

```bash
git clone https://github.com/Shibangan/LoanApproval.git
cd LoanApproval/portfolio/ai-financial-risk-credit-scoring

python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate

pip install -r requirements.txt
python src/generate_data.py
python src/train.py
streamlit run app.py
```

Open the local Streamlit URL shown in your terminal.

## 🐳 Run with Docker

```bash
docker build -t ai-credit-risk .
docker run -p 8501:8501 ai-credit-risk
```

## 📁 Structure

```text
ai-financial-risk-credit-scoring/
├── app.py
├── Dockerfile
├── requirements.txt
├── data/
│   └── README.md
├── models/
│   └── .gitkeep
├── reports/
│   └── .gitkeep
├── src/
│   ├── generate_data.py
│   └── train.py
└── README.md
```

## 🔍 Explainability

The dashboard exposes the strongest factors behind each prediction. SHAP is used to explain the fitted tree model when the dependency is available. The project also stores global feature importance in the training report.

This makes the project stronger than a basic "loan approval" classifier because it connects **prediction → probability → score → explanation → deployment**.

## 🧪 Suggested Extensions

- Add fairness metrics across demographic groups
- Add model monitoring and data-drift detection
- Replace synthetic data with a properly licensed public credit dataset
- Add FastAPI for a REST inference service
- Add MLflow experiment tracking
- Add CI tests with GitHub Actions
- Add PostgreSQL-backed application history

## 👨‍💻 Portfolio Value

This project is designed to showcase practical skills across **machine learning, financial risk modeling, explainable AI, data engineering basics, and deployment** rather than only notebook-based modeling.
