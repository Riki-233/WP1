import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from cycler import cycler

# -------------------- Parámetros globales --------------------
FT2M = 0.3048
M2KM = 0.001
G0 = 9.80665
R_AIR = 287.05287
GAMMA = 1.4

# Límites en km (internamente se trabaja en metros, luego convierte)
X_STOP_KM = 350.0
H_STOP_KM = 15.0

# -------------------- Datos de avión --------------------
@dataclass(frozen=True)
class Aircraft:
    name: str
    Wmax_t: float
    S_m2: float
    CD0_TO: float
    CD2_TO: float
    CD0_IC: float
    CD2_IC: float
    CD0_CLN: float
    CD2_CLN: float
    CT1_N: float
    CT2_ft: float
    CT3_1ft2: float

    def cd_params(self, conf: str):
        if conf == "TO":
            return self.CD0_TO, self.CD2_TO
        if conf == "IC":
            return self.CD0_IC, self.CD2_IC
        return self.CD0_CLN, self.CD2_CLN

# -------------------- Simulador CCO --------------------
class CCOSim:
    def __init__(self, fleet: dict[str, Aircraft]):
        self.fleet = fleet

    # ISA (troposfera + isoterma por encima)
    def isa(self, h_m: float):
        T0, p0, L = 288.15, 101325.0, -0.0065
        if h_m <= 11000.0:
            T = T0 + L*h_m
            p = p0 * (T/T0) ** (-G0/(L*R_AIR))
        else:
            T11 = T0 + L*11000.0
            p11 = p0 * (T11/T0) ** (-G0/(L*R_AIR))
            T = T11
            p = p11 * np.exp(-G0*(h_m-11000.0)/(R_AIR*T))
        rho = p/(R_AIR*T)
        a = np.sqrt(GAMMA*R_AIR*T)
        return T, p, rho, a

    # Configuración por AGL (aprox altitud si no metes elevación)
    def config(self, h_m: float):
        h_ft = h_m / FT2M
        # SoW: 1500 ft y 2000 ft para transiciones
        if h_ft < 1500.0:
            return "TO"
        if h_ft < 2000.0:
            return "IC"
        return "CLN"

    # Thrust máximo con CT1/CT2/CT3
    def thrust_max(self, ac: Aircraft, h_m: float):
        h_ft = h_m / FT2M
        T = ac.CT1_N * (1.0 - h_ft/ac.CT2_ft + ac.CT3_1ft2*(h_ft**2))
        return max(T, 0.0)

    # Drag con CD = CD0 + CD2*CL^2 (CL por equilibrio L~W)
    def drag(self, W_N: float, V_ms: float, rho: float, ac: Aircraft, CD0: float, CD2: float):
        q = 0.5*rho*(V_ms**2)
        CL = W_N/(q*ac.S_m2)
        CD = CD0 + CD2*(CL**2)
        return q*ac.S_m2*CD

    # Velocidades óptimas
    def v_opt(self, mode: str, T_N: float, W_N: float, rho: float, ac: Aircraft, CD0: float, CD2: float):
        if mode == "maxRC":
            disc = T_N**2 + 12.0*CD0*CD2*(W_N**2)
            return np.sqrt((T_N + np.sqrt(disc)) / (3.0*rho*ac.S_m2*CD0))
        if mode == "maxAngle":
            return np.sqrt(2.0*W_N/(rho*ac.S_m2)) * (CD2/CD0)**0.25
        raise ValueError("mode must be 'maxRC' or 'maxAngle'")

    # Integra una trayectoria y devuelve (x_km, y_km)
    def getCCO(self, ac_type: str, MTOW_percent: float, speed_type: str,
               dt: float = 1.0, x_stop_km: float = X_STOP_KM, h_stop_km: float = H_STOP_KM):
        ac = self.fleet[ac_type]
        W_N = (MTOW_percent/100.0) * ac.Wmax_t * 1000.0 * G0

        # SoW: start 35 ft above airport reference elevation
        x_m, h_m = 0.0, 35.0*FT2M

        # Convierte límites a metros para el loop
        x_stop_m = x_stop_km * 1000.0
        h_stop_m = h_stop_km * 1000.0

        path = [(x_m, h_m)]
        while (x_m < x_stop_m) and (h_m < h_stop_m):
            conf = self.config(h_m)
            CD0, CD2 = ac.cd_params(conf)
            _, _, rho, _ = self.isa(h_m)
            Tmax = self.thrust_max(ac, h_m)

            V = self.v_opt(speed_type, Tmax, W_N, rho, ac, CD0, CD2)
            D = self.drag(W_N, V, rho, ac, CD0, CD2)

            excess = Tmax - D
            if excess <= 0.0:
                gamma = 0.0
            else:
                gamma = np.arcsin(np.clip(excess/W_N, 0.0, 0.25))

            x_m += (V*np.cos(gamma)) * dt
            h_m += (V*np.sin(gamma)) * dt
            path.append((x_m, h_m))

        arr = np.array(path, dtype=float)
        # Convierte a km antes de devolver
        return arr[:, 0]*M2KM, arr[:, 1]*M2KM

# -------------------- Fleet (de tu tabla) --------------------
FLEET = {
    "B767-300ER": Aircraft(
        name="B767-300ER", Wmax_t=204.10, S_m2=283.50,
        CD0_TO=0.02070, CD2_TO=0.04830,
        CD0_IC=0.01400, CD2_IC=0.04900,
        CD0_CLN=0.01740, CD2_CLN=0.04590,
        CT1_N=0.35167e6, CT2_ft=0.44673e5, CT3_1ft2=0.10129e-9
    ),
    "B777-300": Aircraft(
        name="B777-300", Wmax_t=299.30, S_m2=428.04,
        CD0_TO=0.01750, CD2_TO=0.05250,
        CD0_IC=0.01730, CD2_IC=0.04840,
        CD0_CLN=0.01570, CD2_CLN=0.04200,
        CT1_N=0.42577e6, CT2_ft=0.48987e5, CT3_1ft2=0.66146e-10
    ),
    "B737": Aircraft(
        name="B737", Wmax_t=70.80, S_m2=124.65,
        CD0_TO=0.03330, CD2_TO=0.04280,
        CD0_IC=0.02700, CD2_IC=0.04410,
        CD0_CLN=0.02350, CD2_CLN=0.04450,
        CT1_N=0.14573e6, CT2_ft=0.55638e5, CT3_1ft2=0.14200e-10
    ),
    "A320-212": Aircraft(
        name="A320-212", Wmax_t=77.00, S_m2=122.60,
        CD0_TO=0.03930, CD2_TO=0.03960,
        CD0_IC=0.02420, CD2_IC=0.04690,
        CD0_CLN=0.02400, CD2_CLN=0.03750,
        CT1_N=0.13605e6, CT2_ft=0.52238e5, CT3_1ft2=0.26637e-10
    ),
    "A319-131": Aircraft(
        name="A319-131", Wmax_t=70.00, S_m2=122.60,
        CD0_TO=0.04450, CD2_TO=0.03280,
        CD0_IC=0.02840, CD2_IC=0.03760,
        CD0_CLN=0.02800, CD2_CLN=0.03100,
        CT1_N=0.13900e6, CT2_ft=0.58900e5, CT3_1ft2=0.57200e-14
    ),
}

# -------------------- API compatible: función suelta getCCO(...) --------------------
_SIM = CCOSim(FLEET)

def getCCO(ac_type, MTOW_percent, speed_type):
    """
    Wrapper para mantener la firma simple getCCO(ac_type, MTOW_percent, speed_type).
    Devuelve (x_km, y_km).
    """
    return _SIM.getCCO(ac_type, MTOW_percent, speed_type, x_stop_km=X_STOP_KM, h_stop_km=H_STOP_KM)

# -------------------- Main: 20 curvas + colores tipo oficial --------------------
def main():
    aircraft_order = ["B767-300ER", "B777-300", "B737", "A320-212", "A319-131"]

    # Orden de casos como en el oficial: 100 RC, 80 RC, 100 angle, 80 angle
    cases = [
        (100, "maxRC", "RC"),
        (80,  "maxRC", "RC"),
        (100, "maxAngle", "angle"),
        (80,  "maxAngle", "angle"),
    ]

    fig, ax = plt.subplots(figsize=(10, 6))

    # Ciclo de colores tab10 (para que coincida el patrón de colores/leyenda)
    ax.set_prop_cycle(cycler(color=list(plt.get_cmap("tab10").colors)))

    for ac in aircraft_order:
        for w, mode, mode_label in cases:
            x_km, y_km = getCCO(ac, w, mode)
            ax.plot(x_km, y_km, linewidth=1.8, label=f"{ac} [{w}% MTOW, {mode_label}]")

    ax.set_xlabel("Distance from start (km)")
    ax.set_ylabel("Altitude (km)")
    ax.set_title("CCO climb trajectories (ISA, max thrust), x_stop=350 km, h_stop=15 km")
    ax.set_xlim(0, X_STOP_KM)
    ax.set_ylim(0, H_STOP_KM)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
