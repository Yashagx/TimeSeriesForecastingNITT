import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

# Initial human population
Nh = 1000
Ih0 = 1     # Initial infected humans
Eh0 = 0     # Initial exposed
Rh0 = 0     # Initial recovered
Sh0 = Nh - Ih0 - Eh0 - Rh0

# Initial vector population
Nv = 5000
Iv0 = 10    # Infected vectors
Sv0 = Nv - Iv0

# Model parameters
beta_hv = 0.001     # Infection rate: vector → human
beta_vh = 0.0005    # Infection rate: human → vector
sigma_h = 1/5       # Incubation rate (1/incubation period in humans)
gamma_h = 1/10      # Recovery rate (1/duration of infection)
mu_v = 1/14         # Vector death rate (1/lifespan)

# Time grid
days = 100
t = np.linspace(0, days, days)

# Model equations
def vector_borne_model(y, t, Nh, Nv, beta_hv, beta_vh, sigma_h, gamma_h, mu_v):
    Sh, Eh, Ih, Rh, Sv, Iv = y

    dSh = -beta_hv * Sh * Iv / Nv
    dEh = beta_hv * Sh * Iv / Nv - sigma_h * Eh
    dIh = sigma_h * Eh - gamma_h * Ih
    dRh = gamma_h * Ih

    dSv = -beta_vh * Sv * Ih / Nh - mu_v * Sv
    dIv = beta_vh * Sv * Ih / Nh - mu_v * Iv

    return [dSh, dEh, dIh, dRh, dSv, dIv]

# Initial conditions vector
y0 = [Sh0, Eh0, Ih0, Rh0, Sv0, Iv0]

# Solve ODEs
result = odeint(vector_borne_model, y0, t, args=(Nh, Nv, beta_hv, beta_vh, sigma_h, gamma_h, mu_v))
Sh, Eh, Ih, Rh, Sv, Iv = result.T

# Plot
plt.figure(figsize=(12, 8))
plt.plot(t, Sh, label='Susceptible Humans')
plt.plot(t, Eh, label='Exposed Humans')
plt.plot(t, Ih, label='Infected Humans')
plt.plot(t, Rh, label='Recovered Humans')
plt.plot(t, Sv, '--', label='Susceptible Vectors', color='brown')
plt.plot(t, Iv, '--', label='Infected Vectors', color='orange')

plt.xlabel('Days')
plt.ylabel('Population')
plt.title('Vector-Borne Disease SEIR-VSIV Model')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
