import numpy as np
import matplotlib.pyplot as plt

# ---------------------------------------------------------
# 1. DADES DELS AVIONS
# ---------------------------------------------------------
AIRCRAFT_DATA = {
    "B767-300ER": {
        "Wmax": 204.10, "S": 283.50,
        "CD0_TO": 0.02070, "CD2_TO": 0.04830,
        "CD0_IC": 0.01400, "CD2_IC": 0.04900,
        "CD0_CLN": 0.01740, "CD2_CLN": 0.04590,
        "CT1": 0.35167e6, "CT2": 44673, "CT3": 0.10129e-9
    },
    "B777-300": {
        "Wmax": 299.30, "S": 428.04,
        "CD0_TO": 0.01750, "CD2_TO": 0.05250,
        "CD0_IC": 0.01730, "CD2_IC": 0.04840,
        "CD0_CLN": 0.01570, "CD2_CLN": 0.04200,
        "CT1": 0.42577e6, "CT2": 48987, "CT3": 0.66146e-10
    },
    "B737": {
        "Wmax": 70.80, "S": 124.65,
        "CD0_TO": 0.03330, "CD2_TO": 0.04280,
        "CD0_IC": 0.02700, "CD2_IC": 0.04410,
        "CD0_CLN": 0.02350, "CD2_CLN": 0.04450,
        "CT1": 0.14573e6, "CT2": 55638, "CT3": 0.14200e-10
    },
    "A320-212": {
        "Wmax": 77.00, "S": 122.60,
        "CD0_TO": 0.03930, "CD2_TO": 0.03960,
        "CD0_IC": 0.02420, "CD2_IC": 0.04690,
        "CD0_CLN": 0.02400, "CD2_CLN": 0.03750,
        "CT1": 0.13605e6, "CT2": 52238, "CT3": 0.26637e-10
    },
    "A319-131": {
        "Wmax": 70.00, "S": 122.60,
        "CD0_TO": 0.04450, "CD2_TO": 0.03280,
        "CD0_IC": 0.02840, "CD2_IC": 0.03760,
        "CD0_CLN": 0.02800, "CD2_CLN": 0.03100,
        "CT1": 0.13900e6, "CT2": 58900, "CT3": 0.57200e-14
    }
}

G0 = 9.80665
FT2M = 0.3048

# ---------------------------------------------------------
# 2. FUNCIONS AUXILIARS
# ---------------------------------------------------------

def get_atmosphere(h_m):
    rho0 = 1.225
    T0 = 288.15
    L = -0.0065
    R = 287.05

    if h_m < 11000:
        T = T0 + L * h_m
        rho = rho0 * (T / T0) ** (-(G0 / (L * R)) - 1)
    else:
        T11 = T0 + L * 11000
        p11 = 22632  # Pa
        rho = (p11 / (R * T11)) * np.exp(-G0 * (h_m - 11000) / (R * T11))
    return rho


def getCCO(ac_type, MTOW_percent, speed_type):

    ac = AIRCRAFT_DATA[ac_type]
    W = (MTOW_percent / 100.0) * ac['Wmax'] * 1000.0 * G0

    x = 0.0
    h = 35.0 * FT2M
    dt = 1.0

    x_valors = [x]
    h_valors = [h]

    while x < 350000:
        h_ft = h / FT2M
        if h_ft < 1500:
            cd0, cd2 = ac['CD0_TO'], ac['CD2_TO']
        elif h_ft < 2000:
            cd0, cd2 = ac['CD0_IC'], ac['CD2_IC']
        else:
            cd0, cd2 = ac['CD0_CLN'], ac['CD2_CLN']

        rho = get_atmosphere(h)
        T_max = ac['CT1'] * (1 - h_ft / ac['CT2'] + ac['CT3'] * h_ft ** 2)

        if speed_type == 'max_RC':
            V = np.sqrt((T_max + np.sqrt(T_max ** 2 + 12 * cd0 * cd2 * W ** 2)) / (3 * rho * ac['S'] * cd0))
        else:
            V = np.sqrt((2 * W) / (rho * ac['S']) * np.sqrt(cd2 / cd0))

        CL = W / (0.5 * rho * V ** 2 * ac['S'])
        CD = cd0 + cd2 * CL ** 2
        D = 0.5 * rho * V ** 2 * ac['S'] * CD

        gamma = np.arcsin((T_max - D) / W)

        x += V * np.cos(gamma) * dt
        h += V * np.sin(gamma) * dt

        if h > 14000: break
        x_valors.append(x)
        h_valors.append(h)

    return np.array(x_valors), np.array(h_valors)


# ---------------------------------------------------------
# 3. SCRIPT PRINCIPAL PER GENERAR EL GRÀFIC
# ---------------------------------------------------------
def main():
    avions = ["B767-300ER", "B777-300", "B737", "A320-212", "A319-131"]
    casos = [
        (100, 'max_RC', 'RC'),
        (80, 'max_RC', 'RC'),
        (100, 'max_angle', 'angle'),
        (80, 'max_angle', 'angle')
    ]

    plt.figure(figsize=(12, 7))

    for model in avions:
        for percent, s_type, label_short in casos:
            x_m, h_m = getCCO(model, percent, s_type)
            plt.plot(x_m, h_m, label=f"{model} [{percent}% MTOW, {label_short}]")

    plt.xlabel("x [m]")
    plt.ylabel("h [m]")
    plt.title("Simulació de trajectòries d'ascens CCO")
    plt.xlim(0, 350000)
    plt.ylim(0, 14000)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(fontsize=7, ncol=2, loc='lower right')
    plt.show()


if __name__ == "__main__":
    main()