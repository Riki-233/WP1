import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass


# ==========================
# CONSTANTES FÍSICAS
# ==========================
FT2M = 0.3048
G = 9.80665
R = 287.05287
GAMMA = 1.4

X_LIMIT_KM = 350.0
H_LIMIT_KM = 15.0


# ==========================
# MODELO ATMOSFÉRICO ISA
# ==========================
class ISAAtmosphere:

    def __init__(self):
        self.T0 = 288.15
        self.P0 = 101325.0
        self.L = -0.0065

    def properties(self, h):

        if h <= 11000:
            T = self.T0 + self.L * h
            P = self.P0 * (T / self.T0) ** (-G / (self.L * R))
        else:
            T11 = self.T0 + self.L * 11000
            P11 = self.P0 * (T11 / self.T0) ** (-G / (self.L * R))
            T = T11
            P = P11 * np.exp(-G * (h - 11000) / (R * T))

        rho = P / (R * T)
        a = np.sqrt(GAMMA * R * T)

        return {"rho": rho, "a": a, "T": T, "P": P}


# ==========================
# AERONAVE
# ==========================
@dataclass(frozen=True)
class Aircraft:

    name: str
    mtow_t: float
    S: float
    cd_data: dict
    thrust_coeff: tuple  # (CT1, CT2, CT3)

    def drag_coefficients(self, phase):
        return self.cd_data[phase]

    def thrust_available(self, altitude):

        CT1, CT2, CT3 = self.thrust_coeff
        h_ft = altitude / FT2M

        T = CT1 * (1 - h_ft / CT2 + CT3 * h_ft**2)

        return max(T, 0.0)


# ==========================
# DINÁMICA DE ASCENSO
# ==========================
class ClimbDynamics:

    def __init__(self, atmosphere):
        self.atm = atmosphere

    def configuration(self, h):
        h_ft = h / FT2M
        if h_ft < 1500:
            return "TO"
        elif h_ft < 2000:
            return "IC"
        return "CLN"

    def optimal_speed(self, mode, T, W, rho, S, cd0, cd2):

        if mode == "maxRC":
            disc = T**2 + 12 * cd0 * cd2 * W**2
            return np.sqrt((T + np.sqrt(disc)) /
                           (3 * rho * S * cd0))

        if mode == "maxAngle":
            return np.sqrt(2 * W / (rho * S)) * (cd2 / cd0)**0.25

        raise ValueError("Unknown mode")

    def drag(self, W, V, rho, S, cd0, cd2):

        q = 0.5 * rho * V**2
        CL = W / (q * S)
        CD = cd0 + cd2 * CL**2

        return q * S * CD

    def state_derivatives(self, state, aircraft, weight, mode):

        x, h = state

        atm_data = self.atm.properties(h)
        rho = atm_data["rho"]

        phase = self.configuration(h)
        cd0, cd2 = aircraft.drag_coefficients(phase)

        thrust = aircraft.thrust_available(h)

        V = self.optimal_speed(mode, thrust, weight,
                               rho, aircraft.S, cd0, cd2)

        D = self.drag(weight, V, rho, aircraft.S, cd0, cd2)

        excess = thrust - D

        if excess <= 0:
            gamma = 0.0
        else:
            gamma = np.arcsin(np.clip(excess / weight, 0, 0.25))

        dxdt = V * np.cos(gamma)
        dhdt = V * np.sin(gamma)

        return np.array([dxdt, dhdt])


# ==========================
# INTEGRADOR EXPLÍCITO
# ==========================
class ExplicitEulerIntegrator:

    def integrate(self, f, state0, dt, stop_condition):

        state = np.array(state0, dtype=float)
        trajectory = [state.copy()]

        while not stop_condition(state):
            derivative = f(state)
            state = state + dt * derivative
            trajectory.append(state.copy())

        return np.array(trajectory)


# ==========================
# FLEET
# ==========================
FLEET = {
    "B767-300ER": Aircraft(
        "B767-300ER", 204.10, 283.50,
        {"TO": (0.02070, 0.04830),
         "IC": (0.01400, 0.04900),
         "CLN": (0.01740, 0.04590)},
        (0.35167e6, 0.44673e5, 0.10129e-9)
    ),
    "B777-300": Aircraft(
        "B777-300", 299.30, 428.04,
        {"TO": (0.01750, 0.05250),
         "IC": (0.01730, 0.04840),
         "CLN": (0.01570, 0.04200)},
        (0.42577e6, 0.48987e5, 0.66146e-10)
    ),
    "B737": Aircraft(
        "B737", 70.80, 124.65,
        {"TO": (0.03330, 0.04280),
         "IC": (0.02700, 0.04410),
         "CLN": (0.02350, 0.04450)},
        (0.14573e6, 0.55638e5, 0.14200e-10)
    ),
    "A320-212": Aircraft(
        "A320-212", 77.00, 122.60,
        {"TO": (0.03930, 0.03960),
         "IC": (0.02420, 0.04690),
         "CLN": (0.02400, 0.03750)},
        (0.13605e6, 0.52238e5, 0.26637e-10)
    ),
    "A319-131": Aircraft(
        "A319-131", 70.00, 122.60,
        {"TO": (0.04450, 0.03280),
         "IC": (0.02840, 0.03760),
         "CLN": (0.02800, 0.03100)},
        (0.13900e6, 0.58900e5, 0.57200e-14)
    ),
}


# ==========================
# API EXTERNA
# ==========================
def getCCO(aircraft_name, MTOW_percent, mode):

    aircraft = FLEET[aircraft_name]
    weight = MTOW_percent / 100 * aircraft.mtow_t * 1000 * G

    atmosphere = ISAAtmosphere()
    dynamics = ClimbDynamics(atmosphere)
    integrator = ExplicitEulerIntegrator()

    state0 = [0.0, 35 * FT2M]

    x_limit = X_LIMIT_KM * 1000
    h_limit = H_LIMIT_KM * 1000

    def stop_condition(state):
        x, h = state
        return (x >= x_limit) or (h >= h_limit)

    trajectory = integrator.integrate(
        lambda s: dynamics.state_derivatives(
            s, aircraft, weight, mode),
        state0,
        dt=1.0,
        stop_condition=stop_condition
    )

    return trajectory[:, 0] / 1000, trajectory[:, 1] / 1000


# ==========================
# MAIN
# ==========================
def main():

    aircraft_order = ["B767-300ER", "B777-300",
                      "B737", "A320-212", "A319-131"]

    scenarios = [
        (100, "maxRC", "RC"),
        (80, "maxRC", "RC"),
        (100, "maxAngle", "angle"),
        (80, "maxAngle", "angle"),
    ]

    fig, ax = plt.subplots(figsize=(10, 6))

    for ac in aircraft_order:
        for w, mode, label in scenarios:
            x, y = getCCO(ac, w, mode)
            ax.plot(x, y, linewidth=1.8,
                    label=f"{ac} [{w}% MTOW, {label}]")

    ax.set_xlabel("Distance from start (km)")
    ax.set_ylabel("Altitude (km)")
    ax.set_title("CCO climb trajectories (ISA, max thrust), x_stop=350 km, h_stop=15 km")
    ax.set_xlim(0, X_LIMIT_KM)
    ax.set_ylim(0, H_LIMIT_KM)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, ncol=2)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()