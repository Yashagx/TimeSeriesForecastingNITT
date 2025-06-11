import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

# Common simulation parameters
N = 1000         # Total population
I0 = 1           # Initial infected
R0 = 0           # Initial recovered
E0 = 0           # Initial exposed (only for SEIR)
S0 = N - I0 - R0 # Initial susceptible
beta = 0.3       # Transmission rate
gamma = 0.1      # Recovery rate
sigma = 0.2      # Incubation rate (for SEIR)
days = 160       # Time frame
t = np.linspace(0, days, days)

# --- SI Model ---
def si_model(y, t, beta):
    S, I = y
    dSdt = -beta * S * I / N
    dIdt = beta * S * I / N
    return [dSdt, dIdt]

# --- SIR Model ---
def sir_model(y, t, beta, gamma):
    S, I, R = y
    dSdt = -beta * S * I / N
    dIdt = beta * S * I / N - gamma * I
    dRdt = gamma * I
    return [dSdt, dIdt, dRdt]

# --- SEIR Model ---
def seir_model(y, t, beta, sigma, gamma):
    S, E, I, R = y
    dSdt = -beta * S * I / N
    dEdt = beta * S * I / N - sigma * E
    dIdt = sigma * E - gamma * I
    dRdt = gamma * I
    return [dSdt, dEdt, dIdt, dRdt]

# Simulate SI
si_result = odeint(si_model, [S0, I0], t, args=(beta,))
S_si, I_si = si_result.T

# Simulate SIR
sir_result = odeint(sir_model, [S0, I0, R0], t, args=(beta, gamma))
S_sir, I_sir, R_sir = sir_result.T

# Simulate SEIR
seir_result = odeint(seir_model, [S0, E0, I0, R0], t, args=(beta, sigma, gamma))
S_seir, E_seir, I_seir, R_seir = seir_result.T

# --- Plotting All Models ---
fig, axs = plt.subplots(3, 1, figsize=(10, 15))
plt.subplots_adjust(hspace=0.4)

# SI plot
axs[0].plot(t, S_si, 'b', label='Susceptible')
axs[0].plot(t, I_si, 'r', label='Infected')
axs[0].set_title('SI Model')
axs[0].set_xlabel('Days')
axs[0].set_ylabel('Population')
axs[0].legend()

# SIR plot
axs[1].plot(t, S_sir, 'b', label='Susceptible')
axs[1].plot(t, I_sir, 'r', label='Infected')
axs[1].plot(t, R_sir, 'g', label='Recovered')
axs[1].set_title('SIR Model')
axs[1].set_xlabel('Days')
axs[1].set_ylabel('Population')
axs[1].legend()

# SEIR plot
axs[2].plot(t, S_seir, 'b', label='Susceptible')
axs[2].plot(t, E_seir, 'orange', label='Exposed')
axs[2].plot(t, I_seir, 'r', label='Infected')
axs[2].plot(t, R_seir, 'g', label='Recovered')
axs[2].set_title('SEIR Model')
axs[2].set_xlabel('Days')
axs[2].set_ylabel('Population')
axs[2].legend()

plt.tight_layout()
plt.show()
