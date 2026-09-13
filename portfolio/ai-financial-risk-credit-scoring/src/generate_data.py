"""Generate a reproducible synthetic credit-risk dataset for portfolio/demo use."""
from pathlib import Path
import numpy as np
import pandas as pd

RNG = np.random.default_rng(42)
ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)


def generate(n: int = 6000) -> pd.DataFrame:
    age = RNG.integers(21, 66, n)
    income = np.clip(RNG.lognormal(10.9, 0.55, n), 180_000, 8_000_000)
    employment_years = np.clip(age - RNG.integers(20, 32, n) + RNG.normal(0, 2, n), 0, 40).round(1)
    dti = np.clip(RNG.beta(2.2, 5.0, n) * 0.9, 0.02, 0.85)
    credit_history = np.clip(employment_years + RNG.normal(4, 3, n), 0.5, 35).round(1)
    open_accounts = np.clip(RNG.poisson(5, n) + 1, 1, 18)
    delinquencies = np.clip(RNG.poisson(0.55, n), 0, 8)
    loan_amount = np.clip(income * RNG.uniform(0.08, 0.85, n), 25_000, 2_500_000)
    loan_term = RNG.choice([12, 24, 36, 48, 60, 72], n, p=[.05, .12, .28, .15, .30, .10])
    savings = np.clip(income * RNG.uniform(0.03, 1.4, n), 5_000, 5_000_000)
    previous_defaults = np.clip(RNG.poisson(0.18, n), 0, 4)
    utilization = np.clip(RNG.beta(2.0, 3.5, n), 0.01, 0.99)
    employment_type = RNG.choice(["Salaried", "Self-employed", "Contract", "Unemployed"], n, p=[.62, .20, .13, .05])
    home = RNG.choice(["Owned", "Mortgage", "Rented"], n, p=[.28, .43, .29])
    purpose = RNG.choice(["Home", "Education", "Medical", "Business", "Personal", "Vehicle"], n)

    income_l = np.log1p(income)
    risk_logit = (
        -4.0
        + 2.9 * dti
        + 1.8 * utilization
        + 0.70 * delinquencies
        + 1.35 * previous_defaults
        + 0.00000055 * loan_amount
        - 0.00000012 * savings
        - 0.045 * credit_history
        - 0.00000009 * income
        + 0.08 * (loan_term >= 60)
        + 0.25 * (employment_type == "Contract")
        + 0.80 * (employment_type == "Unemployed")
        + 0.18 * (home == "Rented")
        + 0.05 * (purpose == "Personal")
        + RNG.normal(0, 0.55, n)
    )
    probability = 1 / (1 + np.exp(-risk_logit))
    default_risk = RNG.binomial(1, probability)

    return pd.DataFrame({
        "age": age,
        "annual_income": income.round(2),
        "employment_years": employment_years,
        "debt_to_income": dti.round(4),
        "credit_history_years": credit_history,
        "num_open_accounts": open_accounts,
        "num_delinquencies": delinquencies,
        "loan_amount": loan_amount.round(2),
        "loan_term_months": loan_term,
        "savings_balance": savings.round(2),
        "previous_defaults": previous_defaults,
        "utilization_ratio": utilization.round(4),
        "employment_type": employment_type,
        "home_ownership": home,
        "loan_purpose": purpose,
        "default_risk": default_risk,
    })


if __name__ == "__main__":
    df = generate()
    out = DATA_DIR / "credit_risk.csv"
    df.to_csv(out, index=False)
    print(f"Saved {len(df):,} rows to {out}")
