import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from scipy.optimize import curve_fit
import matplotlib.dates as mdates

# Load COVID-19 data
df = pd.read_csv("owid-covid-data.csv")
country = "India"
df = df[df["location"] == country].copy()
df['total_cases'] = df['total_cases'].fillna(0)
df['total_deaths'] = df['total_deaths'].fillna(0)

# Estimate active and recovered cases
df['active_estimate'] = df['total_cases'] - df['total_deaths']
df = df[df['active_estimate'] > 0].iloc[:100]  # First 100 days with valid data
N = int(df['population'].iloc[0])

df['infected'] = df['active_estimate']
df['recovered'] = df['total_cases'] - df['infected']
df['susceptible'] = N - df['infected'] - df['recovered']
df['exposed'] = df['infected'] * 0.2  # estimate for SEIR

t = np.arange(len(df))
S, I, R = df['susceptible'].values, df['infected'].values, df['recovered'].values
E = df['exposed'].values

# --- SI Model ---
def si_model(y, t, beta):
    S, I = y
    dSdt = -beta * S * I / N
    dIdt = beta * S * I / N
    return dSdt, dIdt

def fit_si(t, beta):
    y0 = S[0], I[0]
    sol = odeint(si_model, y0, t, args=(beta,))
    return sol[:, 1]

beta_si, _ = curve_fit(fit_si, t, I, bounds=(0.0001, 1))
si_result = odeint(si_model, (S[0], I[0]), t, args=(beta_si[0],))
_, I_si = si_result.T

# --- SIR Model ---
def sir_model(y, t, beta, gamma):
    S, I, R = y
    dSdt = -beta * S * I / N
    dIdt = beta * S * I / N - gamma * I
    dRdt = gamma * I
    return dSdt, dIdt, dRdt

def fit_sir(t, beta, gamma):
    y0 = S[0], I[0], R[0]
    sol = odeint(sir_model, y0, t, args=(beta, gamma))
    return sol[:, 1]

params_sir, _ = curve_fit(fit_sir, t, I, bounds=([0.0001, 0.0001], [1, 1]))
sir_result = odeint(sir_model, (S[0], I[0], R[0]), t, args=tuple(params_sir))
_, I_sir, _ = sir_result.T

# --- SEIR Model ---
def seir_model(y, t, beta, gamma, sigma):
    S, E, I, R = y
    dSdt = -beta * S * I / N
    dEdt = beta * S * I / N - sigma * E
    dIdt = sigma * E - gamma * I
    dRdt = gamma * I
    return dSdt, dEdt, dIdt, dRdt

def fit_seir(t, beta, gamma, sigma):
    y0 = S[0], E[0], I[0], R[0]
    sol = odeint(seir_model, y0, t, args=(beta, gamma, sigma))
    return sol[:, 2]

params_seir, _ = curve_fit(fit_seir, t, I, bounds=([0.0001, 0.0001, 0.0001], [1, 1, 1]))
seir_result = odeint(seir_model, (S[0], E[0], I[0], R[0]), t, args=tuple(params_seir))
_, _, I_seir, _ = seir_result.T

# Convert date strings to datetime
df['date'] = pd.to_datetime(df['date'])

# --- Enhanced Plotting ---
plt.figure(figsize=(16, 8))
plt.plot(df['date'], I, color='black', linewidth=2, label='🟡 Actual Infected')
plt.plot(df['date'], I_si, linestyle='--', color='orange', linewidth=2, label='🔶 SI Model Prediction')
plt.plot(df['date'], I_sir, linestyle='-.', color='blue', linewidth=2, label='🔷 SIR Model Prediction')
plt.plot(df['date'], I_seir, linestyle=':', color='green', linewidth=2, label='🟢 SEIR Model Prediction')

plt.title(f'📊 COVID-19 Infection Modeling in {country}', fontsize=18, fontweight='bold')
plt.xlabel('Date', fontsize=14)
plt.ylabel('Number of Infected People', fontsize=14)
plt.xticks(rotation=45, fontsize=12)
plt.yticks(fontsize=12)
plt.grid(True, linestyle='--', alpha=0.6)

# Format date ticks
plt.gca().xaxis.set_major_locator(mdates.AutoDateLocator())
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))

# Legend outside the plot
plt.legend(loc='upper left', bbox_to_anchor=(1, 1), fontsize=12, frameon=True)

plt.tight_layout()
plt.subplots_adjust(right=0.8)
plt.show()
