# 💳 Loan Approval Prediction

> An end-to-end machine learning project for predicting loan approval outcomes from applicant and financial attributes.

[![Python](https://img.shields.io/badge/Python-3.x-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![Pandas](https://img.shields.io/badge/Pandas-Data%20Analysis-150458?logo=pandas&logoColor=white)](https://pandas.pydata.org/)
[![Scikit--Learn](https://img.shields.io/badge/Scikit--Learn-ML-F7931E?logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626?logo=jupyter&logoColor=white)](https://jupyter.org/)

## 📌 Overview

This project explores applicant information and builds a machine-learning workflow to predict whether a loan application is likely to be approved.

The training dataset contains **614 records and 13 columns**, covering demographic, education, employment, income, loan and credit-related information.

## 🧠 Project Workflow

```text
Raw Dataset
    │
    ▼
Data Inspection & Cleaning
    │
    ▼
Exploratory Data Analysis
    │
    ▼
Feature Preparation
    │
    ▼
Model Training
    │
    ▼
Model Evaluation
    │
    ▼
Saved Prediction Model
```

## 📂 Repository Structure

```text
LoanApproval/
│
├── LoanApproval/
│   ├── LoanApproval.ipynb
│   ├── loan_approval_model.pkl
│   ├── train_u6lujuX_CVtuZ9i.csv
│   └── test_Y3wMUE5_7gLdaTN.csv
│
└── README.md
```

## 🔍 Dataset Features

The dataset includes variables such as:

- Gender
- Marital status
- Dependents
- Education
- Self-employment status
- Applicant income
- Co-applicant income
- Loan amount
- Loan term
- Credit history
- Property area
- Loan approval outcome

## 🛠️ Tech Stack

| Area | Tools |
|---|---|
| Language | Python |
| Data Analysis | Pandas, NumPy |
| Visualization | Matplotlib, Seaborn |
| Machine Learning | Scikit-learn |
| Development | Jupyter Notebook |
| Model Persistence | Pickle |

## 🚀 Run Locally

```bash
git clone https://github.com/Shibangan/LoanApproval.git
cd LoanApproval
pip install pandas numpy matplotlib seaborn scikit-learn jupyter
jupyter notebook
```

Open `LoanApproval/LoanApproval.ipynb` and run the notebook from top to bottom.

## 🎯 What This Project Demonstrates

- Practical data cleaning
- Exploratory data analysis
- Handling categorical and numerical variables
- Machine-learning model development
- Model persistence with a `.pkl` artifact
- Reproducible notebook-based analysis

## 🔮 Planned Improvements

- [ ] Add model comparison across multiple classifiers
- [ ] Add cross-validation and hyperparameter tuning
- [ ] Add confusion matrix and ROC-AUC visualization
- [ ] Add explainable AI with SHAP
- [ ] Build an interactive Streamlit prediction app
- [ ] Add automated tests and CI
- [ ] Add a polished project dashboard

## 👨‍💻 About

I'm a Data Science student building practical projects across **data analytics, machine learning, AI and visualization**.

This repository is part of my growing data-science portfolio.

⭐ If you find the project useful, consider starring the repository.
