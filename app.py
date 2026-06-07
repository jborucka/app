import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

st.set_page_config(
    page_title="Symulator SIRV – odra",
    layout="wide"
)

st.title("Interaktywny symulator modelu SIRV")
st.write(
    "Model pokazuje wpływ parametrów β, γ, poziomu szczepień i liczby początkowo zakażonych na przebieg epidemii."
)

# Parametry boczne
st.sidebar.header("Parametry modelu")

N = st.sidebar.slider(
    "Populacja",
    min_value=100_000,
    max_value=40_000_000,
    value=36_500_000,
    step=100_000
)

beta = st.sidebar.slider(
    "β - tempo transmisji",
    min_value=0.1,
    max_value=3.0,
    value=1.875,
    step=0.025
)

gamma = st.sidebar.slider(
    "γ - tempo zdrowienia",
    min_value=0.01,
    max_value=0.5,
    value=0.125,
    step=0.005
)

vax = st.sidebar.slider(
    "Poziom szczepień",
    min_value=0.80,
    max_value=0.99,
    value=0.92,
    step=0.01
)

I0 = st.sidebar.slider(
    "Początkowa liczba zakażonych I₀",
    min_value=1,
    max_value=1000,
    value=10,
    step=1
)

days = st.sidebar.slider(
    "Liczba dni symulacji",
    min_value=30,
    max_value=1000,
    value=365,
    step=30
)

# Model SIRV
def sirv_simulate(I0, vax, N, beta, gamma, days):

    V0 = vax * N
    S0 = max(N - V0 - I0, 1.0)
    R0_state = 0.0

    def odes(t, y):
        S, I, R = y

        Ntot = S + I + R + V0

        dS = -beta * S * I / Ntot
        dI = beta * S * I / Ntot - gamma * I
        dR = gamma * I

        return [dS, dI, dR]

    t_eval = np.linspace(0, days, days + 1)

    sol = solve_ivp(
        odes,
        (0, days),
        [S0, I0, R0_state],
        t_eval=t_eval,
        method="RK45",
        rtol=1e-8,
        atol=1e-8
    )

    return sol, V0, S0

sol, V0, S0 = sirv_simulate(
    I0=I0,
    vax=vax,
    N=N,
    beta=beta,
    gamma=gamma,
    days=days
)

# Parametry epidemiologiczne
R0_model = beta / gamma
Reff = R0_model * (1 - vax)

S_star_fraction = gamma / beta
S_star_people = S_star_fraction * N

if Reff < 1:
    status = "Epidemia wygasa"
else:
    status = "Ryzyko rozwoju epidemii"

# Metryki
col1, col2, col3, col4 = st.columns(4)

col1.write(f"R0 = {R0_model:.2f}")
col2.write(f"Reff = {Reff:.2f}")
col3.write(f"S* = {S_star_fraction*100:.1f}%")
col4.write(status)

st.write("---")

# Wykres SIRV
fig, ax = plt.subplots(figsize=(12, 6))

ax.plot(sol.t, sol.y[0], label="S - podatni")
ax.plot(sol.t, sol.y[1], label="I - zakażeni")
ax.plot(sol.t, sol.y[2], label="R - ozdrowiali")
ax.axhline(V0, linestyle="--", label="V - zaszczepieni")

ax.set_title("Przebieg modelu SIRV")
ax.set_xlabel("Dzień")
ax.set_ylabel("Liczba osób")
ax.set_yscale("log")
ax.set_ylim(1, None)
ax.legend()
ax.grid(alpha=0.4)

st.pyplot(fig)

# Wykres zakażonych
fig2, ax2 = plt.subplots(figsize=(12, 5))

ax2.plot(sol.t, sol.y[1], label="I(t) - aktywnie zakażeni")

ax2.set_title("Liczba aktywnie zakażonych w czasie")
ax2.set_xlabel("Dzień")
ax2.set_ylabel("Liczba zakażonych")

ax2.legend()
ax2.grid(alpha=0.4)

st.pyplot(fig2)

# Interpretacja
st.subheader("Interpretacja wyników")

st.write(f"""
### Wybrane parametry

- β = **{beta:.3f}**
- γ = **{gamma:.3f}**
- poziom szczepień = **{vax*100:.1f}%**
- populacja = **{N:,.0f}**
- początkowo zakażonych = **{I0}**

### Parametry epidemiologiczne

- R₀ = **{R0_model:.2f}**
- R_eff = **{Reff:.2f}**
- S* = **{S_star_fraction:.3f}**
  (około **{S_star_fraction*100:.1f}% populacji**)

### Wniosek

Jeżeli **R_eff < 1**, epidemia stopniowo wygasa.

Jeżeli **R_eff > 1**, istnieje ryzyko rozwoju epidemii.

Aktualny status: **{status}**
""")

if Reff < 1:
    st.success("Dla tych parametrów epidemia ma tendencję do wygasania.")
else:
    st.error("Dla tych parametrów epidemia może się rozwijać.")
