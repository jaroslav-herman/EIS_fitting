import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from glob import glob

# from hybdrt.models import DRT
from scipy.interpolate import interp1d
import wepy.basics as we


def IV_curves(files, plot=True, norm=1):
    if isinstance(files, str):
        files = [files]
    Ecells = []
    Is = []
    for file in files:
        Ecells.append([])
        Is.append([])
    print(Ecells)
    for i, file in enumerate(files):
        # file = glob(os.path.join(file + "*SV*" + ".mpt"))[0]
        df = we.read_file(file)
        unique_vals = np.unique(df["cycle number"].values)

        for val in unique_vals:
            Ecell = df.loc[(df["cycle number"] == val), "control/V"].values
            mask = np.diff(Ecell) > 0
            mask = np.append(mask, False)

            I = (df.loc[(df["cycle number"] == val), "<I>/mA"].values)[mask] / norm
            # Ewe = (df.loc[(df["cycle number"] == val), 'Ewe/V'].values)[mask]
            # Ece = (df.loc[(df["cycle number"] == val), 'Ece/V'].values)[mask]
            Ecell = Ecell[mask]

            Is[i].append(np.array(I))
            Ecells[i].append(np.array(Ecell))

            if plot:
                label = f"{we.get_sample_number(file)} day {we.get_day(file)} proc {we.get_procedure(file)} cyc {int(val)}"
                plt.plot(I, Ecell, label=label)
    if plot:
        plt.legend()
        plt.show()
    # Is = np.array(Is)
    # Ecells = np.array(Ecells)

    return Ecells, Is


def IV_curves_data(data, norm=1):

    Ecells = []
    Is = []

    groups = data.groupby("cycle number")
    for cycle, group in groups:

        Ecell = group["control/V"].values
        mask = np.diff(Ecell) > 0
        mask = np.append(mask, False)

        I = (group["<I>/mA"].values)[mask] / norm
        # Ewe = (df.loc[(df["cycle number"] == val), 'Ewe/V'].values)[mask]
        # Ece = (df.loc[(df["cycle number"] == val), 'Ece/V'].values)[mask]
        Ecell = Ecell[mask]

        Is.append(np.array(I))
        Ecells.append(np.array(Ecell))

    return Ecells, Is


def current_vs_time(
    files,
    U=1.9,
    I=[],
    E=[],
    plot=False,
    norm=1,
    Ros=[],
    ERs=[],
    cycles=[2, 10, 19],
    corr=False,
):
    if len(I) == 0:
        I, E = IV_curves(files, plot=False)
    I_at_U = []
    if isinstance(U, float):
        U = [U]
    if len(Ros) > 0 and len(Ros) == len(ERs):
        corr = True
    if corr == True and len(Ros) == 0:
        Ros, ERs = calculate_Ro_files(files, cycles)

    for u in U:
        I_at_u = []

        if corr == False:
            for I_day, E_day in zip(I, E):
                for j in range(0, len(I_day)):
                    index = np.absolute(E_day[j] - u).argmin()
                    I_at_u.append(I_day[j][index])
        elif corr == True:
            for I_day, E_day, ER_day, Ro_day in zip(I, E, ERs, Ros):
                for j in range(0, len(I_day)):
                    U_corr, _ = corrected_voltage(
                        E_day[j], I_day[j], ER_day[j], Ro_day[j]
                    )
                    index = np.absolute(U_corr - u).argmin()
                    I_at_u.append(I_day[j][index])

        I_at_U.append(I_at_u)
    I_at_U = np.array(I_at_U) / norm
    if plot == True:
        colors = get_colors(len(U))
        for i, u, color in zip(I_at_U, U, colors):
            plt.plot(i, "x", label=str(u) + " V", c=color)
            plt.plot(i, "--", alpha=0.5, c=color)
            plt.xlabel("Cycle")
            if norm == 1:
                plt.ylabel("I (mA)")
            else:
                plt.ylabel("I (ma cm-2)")
            plt.legend()
        plt.show()

    return I_at_U


def corrected_voltage(V_meas, I, V_R, R_values):
    # Interpolate resistance across full voltage range
    R_interp_func = interp1d(V_R, R_values, kind="linear", fill_value="extrapolate")
    R_interp = R_interp_func(V_meas)

    # Apply ohmic correction
    V_corr = V_meas - I * R_interp / 1000
    return V_corr, R_interp


def IV_curves_corr(files, Ro=None, plot=True, norm=1):
    if isinstance(files, str):
        files = [files]
    if isinstance(Ro, float):
        Ro = [Ro]
    Ecells = []
    Is = []
    for i, file in enumerate(files):
        skip_rows = get_header_lines(file)
        with open(file, "r", encoding="latin1") as f:
            df = pd.read_csv(f, skiprows=skip_rows - 1, delimiter="\t")
        unique_vals = np.unique(df["cycle number"].values)

        if isinstance(Ro[i], float):
            Ro[i] = [Ro[i]] * len(unique_vals)
        print(Ro[i])

        for val in unique_vals:
            Ecell = df.loc[(df["cycle number"] == val), "control/V"].values
            mask = np.diff(Ecell) > 0
            mask = np.append(mask, False)

            I = (df.loc[(df["cycle number"] == val), "<I>/mA"].values)[mask] / norm
            # Ewe = (df.loc[(df["cycle number"] == val), 'Ewe/V'].values)[mask]
            # Ece = (df.loc[(df["cycle number"] == val), 'Ece/V'].values)[mask]
            Ecell = Ecell[mask] - Ro[i][int(val) - 1] * I * norm / 1000

            Is.append(I)
            Ecells.append(Ecell)

            if plot:
                label = f"{we.get_sample_number(file)} day {we.get_day(file)} proc {we.get_procedure(file)} cyc {int(val)}"
                plt.plot(I, Ecell, label=label)
    if plot:
        plt.legend()
        plt.show()
    Is = np.array(Is)
    Ecells = np.array(Ecells)

    return Is, Ecells


def IV_curves_corrected(files, E=False, I=False, ERs=False, Ros=False, plot=True):
    if E == False or I == False:
        I, E = IV_curves(files, plot=False)
    E_corr = E
    if ERs == False or Ros == False:
        Ros, ERs = calculate_Ro_files(files, cycles=[2, 5, 10, 14])
    E_corr = []
    for file in files:
        E_corr.append([])
    for i, file in enumerate(files):
        file = glob(os.path.join(file + "*PEIS*" + ".mpt"))[0]
        for j, s in enumerate(E[i]):
            U_corr, _ = corrected_voltage(E[i][j], I[i][j], ERs[i][j], Ros[i][j])
            E_corr[i].append(U_corr)
            if plot:
                label = f"{get_sample_number(file)} day {get_day(file)} proc {get_procedure(file)} cyc {j}"
                plt.plot(I[i][j], U_corr, label=label)
    if plot:
        plt.legend()
        plt.show()
    return E_corr
