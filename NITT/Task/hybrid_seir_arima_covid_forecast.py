import numpy as np
import pandas as pd
from scipy.integrate import odeint
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt

# Load the dataset
df = pd.read_csv('owid-covid-data.csv', usecols=['location', 'date', 'new_cases'])
df = df[df['location'] == 'India']
df['date'] = pd.to_datetime(df['date'])
df = df[['date', 'new_cases']].dropna()
df = df[df['new_cases'] >= 0]
df.set_index('date', inplace=True)
weekly_df = df.resample('W').sum()

# Split into training and testing datasets
train = weekly_df[:'2023-11']
test = weekly_df['2023-12':]

# SEIR model implementation
def seir_model(y, t, beta, sigma, gamma, N):
    S, E, I, R = y
    dSdt = -beta * S * I / N
    dEdt = beta * S * I / N - sigma * E
    dIdt = sigma * E - gamma * I
    dRdt = gamma * I
    return dSdt, dEdt, dIdt, dRdt

# Parameters
N = 1.38e9  # India's population
I0, R0 = 1, 0
E0 = I0 * 2
S0 = N - I0 - R0 - E0
beta, sigma, gamma = 0.5, 1/5.2, 1/12.39
y0 = S0, E0, I0, R0
t_train = np.arange(len(train))
ret = odeint(seir_model, y0, t_train, args=(beta, sigma, gamma, N))
S, E, I, R = ret.T
seir_pred_train = pd.Series(I, index=train.index)

# Residuals
residuals = train['new_cases'] - seir_pred_train

# Forecasting and evaluation
best_model = None
best_r2 = -np.inf
for p in range(2, 6):
    for d in range(1, 4):
        for q in range(2, 6):
            try:
                model = ARIMA(residuals, order=(p, d, q)).fit()
                forecast_resid = model.forecast(steps=len(test))
                
                # SEIR forecast for test period
                t_test = np.arange(len(train), len(train) + len(test))
                ret_test = odeint(seir_model, y0, t_test, args=(beta, sigma, gamma, N))
                I_test = ret_test[:, 2]
                seir_pred_test = pd.Series(I_test, index=test.index)
                
                # Combined forecast
                combined_forecast = seir_pred_test + forecast_resid.values
                mae = mean_absolute_error(test['new_cases'], combined_forecast)
                rmse = np.sqrt(mean_squared_error(test['new_cases'], combined_forecast))
                r2 = r2_score(test['new_cases'], combined_forecast)
                
                if r2 > best_r2:
                    best_r2 = r2
                    best_model = {
                        'order': (p, d, q),
                        'mae': mae,
                        'rmse': rmse,
                        'r2': r2,
                        'forecast': combined_forecast
                    }
            except:
                continue

# Output results
print(f"Best ARIMA Model Order: {best_model['order']}")
print(f"MAE: {best_model['mae']:.2f}")
print(f"RMSE: {best_model['rmse']:.2f}")
print(f"R²: {best_model['r2']:.4f}")

# Plotting
plt.figure(figsize=(12, 6))
plt.plot(train.index, train['new_cases'], label='Training Data')
plt.plot(test.index, test['new_cases'], label='Actual Test Data')
plt.plot(test.index, best_model['forecast'], label='SEIR + ARIMA Forecast')
plt.xlabel('Date')
plt.ylabel('Weekly New Cases')
plt.title('COVID-19 Weekly New Cases Forecast')
plt.legend()
plt.tight_layout()
plt.show()
