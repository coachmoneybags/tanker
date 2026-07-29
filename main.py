"""
Cooling model for the jacketed nitrile dip tank.
Goal: hold the compound at 20 C against the process heat load.

Calibrated from plant data:
  With the 15 kW chiller running, the tank still warmed ~2.1 C/hr.
  net warming = load - cooling:  mass*cp*rate = load - 15 kW
  -> load ~= 18.2 kW. So 15 kW of cooling is ~3 kW short.
"""

import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

# --- Tank contents ---
V_tank = 1.514               # tank volume of compound, cubic metres (m^3)
rho = 990.0                  # compound density, kg/m^3  (nitrile latex ~ water-ish)
cp = 3650.0                  # compound specific heat, J/(kg*K)  (aqueous dispersion)
mass = rho * V_tank          # total mass of compound in the tank, kg
T0 = 20.0                    # start at the temperature we want to hold

# --- Heat load (calibrated from the measured warm-up) ---
Q_load = 18200.0             # W, net process heat into the tank (formers + room)

# --- Cooling ---
chiller_capacity = 15000.0   # W, max the real chiller can remove  <-- try 19000
T_target = 20.0              # deg C, the temperature we want to hold

def dTdt(t, T):
    T = T[0]
    # Simple chiller model: full capacity whenever the tank is above
    # target, off when at/below it. This is what "running but losing" is.
    Q_cool = chiller_capacity if T > T_target else 0.0
    dT = (Q_load - Q_cool) / (mass * cp)
    return [dT]

t_end = 3 * 3600.0                       # simulate 3 hours
t_eval = np.linspace(0, t_end, 1000)
sol = solve_ivp(dTdt, [0, t_end], [T0], t_eval=t_eval,
                method="RK45", max_step=10.0)

t_hours = sol.t / 3600.0
T_hist = sol.y[0]

print(f"Heat load:      {Q_load/1000:.1f} kW")
print(f"Chiller:        {chiller_capacity/1000:.1f} kW")
print(f"Shortfall:      {(Q_load - chiller_capacity)/1000:.1f} kW")
print(f"Tank after {t_end/3600:.0f} h: {T_hist[-1]:.1f} C (target {T_target} C)")

plt.figure(figsize=(9, 5))
plt.plot(t_hours, T_hist, lw=2, label="Tank temperature")
plt.axhline(T_target, color="tab:green", ls="--", label=f"Target ({T_target} C)")
plt.xlabel("Time (hours)")
plt.ylabel("Temperature (deg C)")
plt.title("Cooling: 15 kW chiller vs ~18 kW heat load")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
