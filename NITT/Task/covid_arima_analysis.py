import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import warnings
warnings.filterwarnings("ignore")

# === 1. Load and Filter Dataset ===
df = pd.read_csv('owid-covid-data.csv', parse_dates=['date'])

# Check required columns
if not all(col in df.columns for col in ['location', 'date', 'new_cases']):
    raise ValueError("Dataset must include 'location', 'date', and 'new_cases' columns.")

# Filter India data
df = df[df['location'] == 'India'][['date', 'new_cases']].dropna()
df.columns = ['Date', 'Cases']
df.set_index('Date', inplace=True)

# === 2. Convert to Weekly Infected ===
weekly_cases = df['Cases'].resample('W').sum()

# === 3. Train-Test Split ===
train = weekly_cases['2020-01-01':'2023-11-30']
test = weekly_cases['2023-12-01':'2024-07-31']

# === 4. ARIMA Grid Search over (p,d,q) ===
best_model = None
best_order = None
best_r2 = float('-inf')
results = []

print("🔍 Starting ARIMA model training and evaluation...\n")

for p in range(2, 6):
    for d in range(1, 4):
        for q in range(2, 6):
            try:
                model = ARIMA(train, order=(p, d, q)).fit()
                forecast = model.forecast(steps=len(test))
                mae = mean_absolute_error(test, forecast)
                rmse = np.sqrt(mean_squared_error(test, forecast))
                r2 = r2_score(test, forecast)
                results.append(((p, d, q), mae, rmse, r2))

                if r2 > best_r2:
                    best_r2 = r2
                    best_model = model
                    best_order = (p, d, q)
            except Exception as e:
                print(f"⚠️ ARIMA({p},{d},{q}) failed: {e}")

# === 5. Report Best Model ===
if best_model:
    forecast = best_model.forecast(steps=len(test))

    print("\n✅ Best ARIMA Model:")
    print(f"Order: {best_order}")
    print(f"MAE: {mean_absolute_error(test, forecast):.2f}")
    print(f"RMSE: {np.sqrt(mean_squared_error(test, forecast)):.2f}")
    print(f"R²: {r2_score(test, forecast):.4f}")

    # === 6. Plot Forecast ===
    plt.figure(figsize=(12, 6))
    plt.plot(test.index, test.values, label="Actual", marker='o')
    plt.plot(test.index, forecast, label=f"Forecast ARIMA{best_order}", linestyle='--', marker='x')
    plt.title(f"COVID-19 Weekly Forecast (India) - ARIMA{best_order}")
    plt.xlabel("Date")
    plt.ylabel("Weekly Infected Cases")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # === 7. Top 5 Models Table ===
    print("\n📊 Top 5 ARIMA Models (Sorted by R²):")
    top_models = sorted(results, key=lambda x: x[3], reverse=True)[:5]
    for i, (order, mae, rmse, r2) in enumerate(top_models, 1):
        print(f"{i}. ARIMA{order} → MAE: {mae:.2f}, RMSE: {rmse:.2f}, R²: {r2:.4f}")
else:
    print("❌ No valid ARIMA model found.")
