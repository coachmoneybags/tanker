import numpy
import scipy
import matplotlib

print("Environment is working. Ready to build the model.")

"""
Dynamic thermal model of a jacketed nitrile dip tank.

Models how the compound temperature in the tank changes over time,
accounting for three heat flows:
  1. The jacket (hot/cold fluid) driving the tank toward a setpoint
  2. Heat carried IN by hot ceramic formers dipping through
  3. Heat lost to the surrounding air (ambient)

This is a "lumped" model: we assume the compound is well-mixed, so the
whole tank is one uniform temperature at any instant. That's the standard
first approximation for a stirred/circulated tank.
"""

import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

# ----------------------------------------------------------------------
# PARAMETERS  --  these are the knobs you'll tune to match your real line.
# I've put in reasonable placeholder values; swap in your actual numbers.
# ----------------------------------------------------------------------

# --- Tank contents (the nitrile compound) ---
V_tank = 1.514           # tank volume of compound, cubic metres (m^3)
rho = 990.0            # compound density, kg/m^3  (nitrile latex ~ water-ish)
cp = 3500.0            # compound specific heat, J/(kg*K)  (aqueous dispersion)
mass = rho * V_tank    # total mass of compound in the tank, kg
T0 = 15.0              # starting compound temperature, deg C

# --- Jacket (your temperature control) ---
T_jacket = 10.0        # jacket fluid temperature / setpoint, deg C
UA_jacket = 400.0      # jacket heat-transfer coeff * area, W/K
                       #   bigger = jacket grips the tank harder/faster

# --- Ambient losses (tank surface to room air) ---
T_ambient = 27.0       # room air temperature, deg C
UA_ambient = 25.0      # tank-surface-to-air UA, W/K
                       #   usually small vs the jacket

# --- Formers (the hot ceramic hands dipping through) ---
# Each former enters hot, dips, and carries heat into the compound.
# We approximate this as a steady average heat input based on how many
# formers per second pass through and how much heat each dumps in.
formers_per_hour = 3600.0   # throughput: formers passing through per hour
T_former = 50.0             # former temperature as it enters the dip, deg C
m_former = 0.8              # mass of compound-contacting former tip, kg
cp_former = 800.0           # former material specific heat, J/(kg*K) (ceramic)
# Heat each former gives up as it equilibrates toward the tank temp is
# handled inside the model (it depends on the current tank temp).

# --- Simulation time ---
t_end = 3600.0         # how long to simulate, seconds (here: 1 hour)

# ----------------------------------------------------------------------
# THE MODEL  --  the energy balance as a differential equation.
#
# Core idea (conservation of energy on the tank):
#
#   (mass * cp) * dT/dt  =  Q_jacket + Q_formers - Q_ambient
#
# i.e. the rate the tank's heat content changes equals heat in from the
# jacket, plus heat carried in by formers, minus heat lost to the room.
# We rearrange to solve for dT/dt (the rate of temperature change) and
# hand that to the solver.
# ----------------------------------------------------------------------

def dTdt(t, T):
    """
    Given the current time t and current tank temperature T,
    return the rate of temperature change dT/dt.
    solve_ivp calls this over and over to step the temperature forward.
    T comes in as a 1-element array, so we use T[0].
    """
    T = T[0]

    # Heat from the jacket: proportional to the temperature gap.
    # If jacket is warmer than tank, this is positive (heating).
    Q_jacket = UA_jacket * (T_jacket - T)

    # Heat lost to ambient: proportional to how much hotter the tank is
    # than the room. Positive value = heat leaving the tank.
    Q_ambient = UA_ambient * (T - T_ambient)

    # Heat carried in by formers.
    # Each former enters at T_former and leaves at ~tank temp, so it gives
    # the compound: m_former * cp_former * (T_former - T) joules.
    # Multiply by how many formers arrive per second to get watts (J/s).
    formers_per_sec = formers_per_hour / 3600.0
    Q_formers = formers_per_sec * m_former * cp_former * (T_former - T)

    # Assemble the energy balance and solve for the rate of change.
    dT = (Q_jacket + Q_formers - Q_ambient) / (mass * cp)
    return [dT]

# ----------------------------------------------------------------------
# SOLVE  --  integrate the temperature forward in time.
# ----------------------------------------------------------------------

# Times at which we want the temperature reported (for a smooth plot).
t_eval = np.linspace(0, t_end, 500)

solution = solve_ivp(
    dTdt,                 # the rate function above
    [0, t_end],           # time span: start and end (seconds)
    [T0],                 # initial condition: starting temperature
    t_eval=t_eval,        # times to record
    method="RK45",        # a good general-purpose solver
)

t_minutes = solution.t / 60.0   # convert seconds -> minutes for the plot
T_history = solution.y[0]       # the temperature at each recorded time

# ----------------------------------------------------------------------
# REPORT + PLOT
# ----------------------------------------------------------------------

T_final = T_history[-1]
print(f"Starting temperature: {T0:.2f} C")
print(f"Final temperature after {t_end/60:.0f} min: {T_final:.2f} C")

plt.figure(figsize=(9, 5))
plt.plot(t_minutes, T_history, linewidth=2, label="Compound temperature")
plt.axhline(T_jacket, color="tab:orange", linestyle="--",
            label=f"Jacket setpoint ({T_jacket} C)")
plt.xlabel("Time (minutes)")
plt.ylabel("Temperature (deg C)")
plt.title("Dynamic dip-tank temperature response")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
