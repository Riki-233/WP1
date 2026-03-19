import numpy as np
import matplotlib.pyplot as plt

# ---------------------------------------------------------
# 1. DADES DELS AVIONS (Originals, sense tocar res)
# ---------------------------------------------------------
AIRCRAFT_DATA = {
    "B767-300ER": {
        "Wmax": 204.10, "S": 283.50,
        "CD0_TO": 0.02070, "CD2_TO": 0.04830,
        "CD0_IC": 0.01400, "CD2_IC": 0.04900,
        "CD0_CLN": 0.01740, "CD2_CLN": 0.04590,
        "CT1": 0.35167e6, "CT2": 44673, "CT3": 0.10129e-9,
        "CF1": 0.54005, "CF2": 5578.2
    },
    "B777-300": {
        "Wmax": 299.30, "S": 428.04,
        "CD0_TO": 0.01750, "CD2_TO": 0.05250,
        "CD0_IC": 0.01730, "CD2_IC": 0.04840,
        "CD0_CLN": 0.01570, "CD2_CLN": 0.04200,
        "CT1": 0.42577e6, "CT2": 48987, "CT3": 0.66146e-10,
        "CF1": 0.87843, "CF2": -36897
    },
    "B737": {
        "Wmax": 70.80, "S": 124.65,
        "CD0_TO": 0.03330, "CD2_TO": 0.04280,
        "CD0_IC": 0.02700, "CD2_IC": 0.04410,
        "CD0_CLN": 0.02350, "CD2_CLN": 0.04450,
        "CT1": 0.14573e6, "CT2": 55638, "CT3": 0.14200e-10,
        "CF1": 0.94680, "CF2": -1e15
    },
    "A320-212": {
        "Wmax": 77.00, "S": 122.60,
        "CD0_TO": 0.03930, "CD2_TO": 0.03960,
        "CD0_IC": 0.02420, "CD2_IC": 0.04690,
        "CD0_CLN": 0.02400, "CD2_CLN": 0.03750,
        "CT1": 0.13605e6, "CT2": 52238, "CT3": 0.26637e-10,
        "CF1": 0.94000, "CF2": -1e6
    },
    "A319-131": {
        "Wmax": 70.00, "S": 122.60,
        "CD0_TO": 0.04450, "CD2_TO": 0.03280,
        "CD0_IC": 0.02840, "CD2_IC": 0.03760,
        "CD0_CLN": 0.02800, "CD2_CLN": 0.03100,
        "CT1": 0.13900e6, "CT2": 58900, "CT3": 0.57200e-14,
        "CF1": 0.68800, "CF2": -16700
    }
}

G0 = 9.80665
FT2M = 0.3048


# ---------------------------------------------------------
# 2. FUNCIONS AUXILIARS
# ---------------------------------------------------------
def get_atmosphere(h_m):
    rho0, T0, L, R = 1.225, 288.15, -0.0065, 287.05
    if h_m < 11000:
        T = T0 + L * h_m
        rho = rho0 * (T / T0) ** (-(G0 / (L * R)) - 1)
    else:
        T11, p11 = T0 + L * 11000, 22632
        rho = (p11 / (R * T11)) * np.exp(-G0 * (h_m - 11000) / (R * T11))
    return rho


def getCCO(ac_type, MTOW_percent, speed_type):
    ac = AIRCRAFT_DATA[ac_type]
    W = (MTOW_percent / 100.0) * ac['Wmax'] * 1000.0 * G0
    x, h, dt = 0.0, 35.0 * FT2M, 1.0
    x_valors, h_valors = [x], [h]

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

        CF1, CF2 = ac['CF1'] / (60 * 1000), ac['CF2'] * 0.514444
        eta = CF1 * (1 + V / CF2)
        FF = eta * T_max
        m = (W / G0) - FF * dt
        W = m * G0

    return np.array(x_valors), np.array(h_valors)


# ---------------------------------------------------------
# 3. SCRIPT PRINCIPAL
# ---------------------------------------------------------
def main():
    avions = ["B767-300ER", "B777-300", "B737", "A320-212", "A319-131"]
    casos = [(100, 'max_RC'), (80, 'max_RC'), (100, 'max_angle'), (80, 'max_angle')]

    # CANVIA AQUESTS VALORS PER LES DISTÀNCIES QUE VOLGUEU (en metres)
    punts_interes = [66063, 105779, 134600]

    plt.figure(figsize=(14, 8))

    # Imprimim capçalera de la taula
    print(f"\n{'CONFIGURACIÓ AVIO':<35} | {'ALTITUD 1 (m)':>15} | {'ALTITUD 2 (m)':>15} | {'ALTITUD 3 (m)':>15}")
    print("-" * 100)

    for model in avions:
        for percent, s_type in casos:
            x_m, h_m = getCCO(model, percent, s_type)

            # Pintem la línia de la trajectòria
            plt.plot(x_m, h_m, label=f"{model} [{percent}%, {s_type}]", alpha=0.7)

            # Busquem i pintem les altituds en els punts d'interès
            altituds_trobades = []
            for p in punts_interes:
                if p <= x_m[-1]:
                    h_interp = np.interp(p, x_m, h_m)
                    altituds_trobades.append(f"{h_interp:15.2f}")
                    # AIXÒ PINTA ELS PUNTS NEGRES AL GRÀFIC
                    plt.scatter(p, h_interp, color='black', s=15, zorder=5)
                else:
                    altituds_trobades.append(f"{'>14000':>15}")

            # Mostrem la fila a la terminal
            nom_fila = f"{model} ({percent}%, {s_type})"
            print(f"{nom_fila:<35} | {altituds_trobades[0]} | {altituds_trobades[1]} | {altituds_trobades[2]}")

    plt.xlabel("Distància horitzontal recorreguda x [m]")
    plt.ylabel("Altitud assolida h [m]")
    plt.title(f"Altituds en distàncies fixes: {punts_interes} metres")
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend(fontsize=7, ncol=2, loc='upper left', bbox_to_anchor=(1, 1))
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()

