import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error, mean_absolute_error
from scipy.integrate import odeint

# Load dataset
df = pd.read_csv("owid-covid-data.csv")
df_india = df[df["location"] == "India"]

# Keep data from Dec 1, 2023
df_india["date"] = pd.to_datetime(df_india["date"])
df_india = df_india[df_india["date"] >= "2023-12-01"]

# Fill missing values
df_india = df_india.fillna(0)

# Define SEIR model
def seir_model(y, t, beta, sigma, gamma, mu):
    S, E, I, R, D = y
    N = S + E + I + R + D
    dSdt = -beta * S * I / N
    dEdt = beta * S * I / N - sigma * E
    dIdt = sigma * E - gamma * I - mu * I
    dRdt = gamma * I
    dDdt = mu * I
    return [dSdt, dEdt, dIdt, dRdt, dDdt]

# Initial conditions
N = 1.4e9  # India's population approx
I0 = df_india["new_cases"].iloc[0]
E0 = I0
R0 = 0
D0 = df_india["new_deaths"].iloc[0]
S0 = N - E0 - I0 - R0 - D0
y0 = [S0, E0, I0, R0, D0]

# SEIR parameters
beta = 0.3
sigma = 1/5.2
gamma = 1/12.0
mu = 0.005

# Time grid
t = np.arange(len(df_india))

# Integrate SEIR equations
ret = odeint(seir_model, y0, t, args=(beta, sigma, gamma, mu))
S, E, I, R, D = ret.T

# Get actual data
actual_infected = df_india["new_cases"].values
actual_deaths = df_india["new_deaths"].values
actual_recovered = np.maximum(I - actual_infected, 0)  # crude estimate

# Residuals
residual_infected = actual_infected - I
residual_deaths = actual_deaths - D
residual_recovered = actual_recovered - R

# ARIMA on residuals
def fit_predict_arima(series, steps):
    model = ARIMA(series, order=(2,1,2))
    model_fit = model.fit()
    forecast = model_fit.forecast(steps=steps)
    return forecast

forecast_steps = 7

infected_forecast = fit_predict_arima(residual_infected, forecast_steps)
deaths_forecast = fit_predict_arima(residual_deaths, forecast_steps)
recovered_forecast = fit_predict_arima(residual_recovered, forecast_steps)

# Combine SEIR + ARIMA
combined_infected = I[-forecast_steps:] + infected_forecast
combined_deaths = D[-forecast_steps:] + deaths_forecast
combined_recovered = R[-forecast_steps:] + recovered_forecast

# Ground truth for comparison
true_infected = actual_infected[-forecast_steps:]
true_deaths = actual_deaths[-forecast_steps:]

# Performance
print("MSE (Infected):", mean_squared_error(true_infected, combined_infected))
print("MAE (Infected):", mean_absolute_error(true_infected, combined_infected))
print("MSE (Deaths):", mean_squared_error(true_deaths, combined_deaths))
print("MAE (Deaths):", mean_absolute_error(true_deaths, combined_deaths))

# Plot
plt.figure(figsize=(12, 5))
plt.plot(true_infected, label="Actual Infected")
plt.plot(combined_infected, label="Predicted Infected (SEIR+ARIMA)")
plt.title("COVID-19 Infected Forecast")
plt.legend()
plt.grid(True)
plt.show()
