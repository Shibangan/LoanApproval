"""Weather Trend & Forecasting starter project.

Creates a reproducible synthetic daily weather dataset when no CSV is supplied,
then engineers calendar, lag, and rolling features and trains a Random Forest
regressor to forecast temperature.
"""

from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
DATA_DIR.mkdir(exist_ok=True)


def make_sample_data(days=1200, seed=42):
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2022-01-01", periods=days, freq="D")
    day_of_year = dates.dayofyear.to_numpy()
    seasonal = 7 * np.sin(2 * np.pi * (day_of_year - 80) / 365.25)
    trend = np.linspace(0, 1.5, days)
    temperature = 26 + seasonal + trend + rng.normal(0, 1.8, days)
    humidity = np.clip(72 - 0.8 * (temperature - 26) + rng.normal(0, 5, days), 35, 98)
    rainfall = np.clip(rng.gamma(1.2, 4, days) * (rng.random(days) < 0.22), 0, None)
    return pd.DataFrame({
        "date": dates,
        "temperature_c": temperature.round(2),
        "humidity_pct": humidity.round(2),
        "rainfall_mm": rainfall.round(2),
    })


def build_features(df):
    df = df.copy().sort_values("date")
    df["day_of_year"] = df["date"].dt.dayofyear
    df["month"] = df["date"].dt.month
    df["temp_lag_1"] = df["temperature_c"].shift(1)
    df["temp_lag_7"] = df["temperature_c"].shift(7)
    df["temp_roll_7"] = df["temperature_c"].shift(1).rolling(7).mean()
    return df.dropna()


def main():
    df = make_sample_data()
    csv_path = DATA_DIR / "sample_weather.csv"
    df.to_csv(csv_path, index=False)

    featured = build_features(df)
    features = ["humidity_pct", "rainfall_mm", "day_of_year", "month", "temp_lag_1", "temp_lag_7", "temp_roll_7"]
    split = int(len(featured) * 0.8)
    train, test = featured.iloc[:split], featured.iloc[split:]

    model = RandomForestRegressor(n_estimators=250, random_state=42, n_jobs=-1)
    model.fit(train[features], train["temperature_c"])
    predictions = model.predict(test[features])

    print("Weather Trend & Forecasting")
    print(f"MAE:  {mean_absolute_error(test['temperature_c'], predictions):.2f} °C")
    print(f"RMSE: {np.sqrt(mean_squared_error(test['temperature_c'], predictions)):.2f} °C")
    print(f"R²:   {r2_score(test['temperature_c'], predictions):.3f}")
    print(f"Sample data saved to: {csv_path}")


if __name__ == "__main__":
    main()
