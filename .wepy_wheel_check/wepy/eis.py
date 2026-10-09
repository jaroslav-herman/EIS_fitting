import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import re
from glob import glob
from hybdrt.models import DRT
import wepy.basics as we
from impedance.models.circuits import CustomCircuit
from bayes_drt2.inversion import Inverter

def load_MEA_params(MEA):
    for mea in MEA:

#         if mea['name'] == '028':
#             mea['PEIS_files'] = glob(os.path.join('C:/Users/Herman/Desktop/MFF/Diplomka/Data z PEMWE/'+mea['name']+'*/PEIS','time*'+'.txt'))
#             mea['current_files'] = glob(os.path.join('C:/Users/Herman/Desktop/MFF/Diplomka/Data z PEMWE/'+mea['name']+'*/PEIS','current*'+'.txt'))

#             mea['cycles'] = len(mea['PEIS_files'])
#             cycles = mea['cycles']
#             print(cycles)

#             mea['U'] = np.empty((cycles,15))
#             mea['Ro'] = np.empty((cycles,15)) # Ohmic resistance
#             mea['Roe'] = np.empty((cycles,15)) # Ohmic resistance error
#             mea['Ra'] = np.empty((cycles,15)) # anode resistance
#             mea['Rae'] = np.empty((cycles,15)) # anode resistance error
#             mea['Pa'] = np.empty((cycles,15)) # Anode CPE
#             mea['Pae'] = np.empty((cycles,15)) # Anode CPE error
#             mea['aa'] = np.empty((cycles,15)) # Anode alpha parameter
#             mea['aae'] = np.empty((cycles,15)) # Anode alpha parameter error

#             mea['Cc'] = np.empty((cycles,15)) # Cathod capacity
#             mea['Ca'] = np.empty((cycles,15)) # Cathod capacity
#             mea['I'] = np.empty((cycles,15)) # Current measured

#             for cycle, file in enumerate(mea['PEIS_files']):
#                 d = np.loadtxt(file,skiprows=0)
#                 mea['U'][cycle] = d[0,:]
#                 mea['Ro'][cycle] = d[1,:]
#                 mea['Roe'][cycle] = d[2,:]

#                 mea['Ra'][cycle] = d[3,:]
#                 mea['Rae'][cycle] = d[4,:]
#                 mea['Pa'][cycle] = d[5,:]
#                 mea['Pae'][cycle] = d[6,:]
#                 mea['aa'][cycle] = d[7,:]
#                 mea['aae'][cycle] = d[8,:]

#                 mea['Ca'][cycle] = mea['Ra'][cycle]**((1-mea['aa'][cycle])/mea['aa'][cycle])*(mea['Pa'][cycle])**((1)/mea['aa'][cycle])

#         else:

        mea['PEIS_files'] = glob(os.path.join('C:/Users/Herman/Desktop/MFF/Diplomka/Data z PEMWE/'+mea['name']+'*/PEIS','time*'+'.txt'))
        mea['current_files'] = glob(os.path.join('C:/Users/Herman/Desktop/MFF/Diplomka/Data z PEMWE/'+mea['name']+'*/PEIS','current*'+'.txt'))

        mea['cycles'] = len(mea['PEIS_files'])
        cycles = mea['cycles']
        print(cycles)

        mea['U'] = np.empty((cycles,15))
        mea['Ro'] = np.empty((cycles,15)) # Ohmic resistance
        mea['Roe'] = np.empty((cycles,15)) # Ohmic resistance error
        mea['Rind'] = np.empty((cycles,15)) #  resistance of inductance element
        mea['Rinde'] = np.empty((cycles,15)) # resistance of inductance element error
        mea['Ind'] = np.empty((cycles,15)) #  Inductance
        mea['Inde'] = np.empty((cycles,15)) # Inductance error
        mea['Rc'] = np.empty((cycles,15)) # Cathod resistance
        mea['Rce'] = np.empty((cycles,15)) # Cathod resistance error
        mea['Pc'] = np.empty((cycles,15)) # Cathod CPE
        mea['Pce'] = np.empty((cycles,15)) # Cathod CPE error
        mea['ac'] = np.empty((cycles,15)) # Cathod alpha parameter
        mea['ace'] = np.empty((cycles,15)) # Cathod alpha parameter error
        mea['Ra'] = np.empty((cycles,15)) # anode resistance
        mea['Rae'] = np.empty((cycles,15)) # anode resistance error
        mea['Pa'] = np.empty((cycles,15)) # Anode CPE
        mea['Pae'] = np.empty((cycles,15)) # Anode CPE error
        mea['aa'] = np.empty((cycles,15)) # Anode alpha parameter
        mea['aae'] = np.empty((cycles,15)) # Anode alpha parameter error

        mea['Cc'] = np.empty((cycles,15)) # Cathod capacity
        mea['Ca'] = np.empty((cycles,15)) # Cathod capacity
        mea['I'] = np.empty((cycles,15)) # Current measured

        for cycle, file in enumerate(mea['PEIS_files']):
            d = np.loadtxt(file,skiprows=0)
            mea['U'][cycle] = d[0,:]
            mea['Ro'][cycle] = d[1,:]
            mea['Roe'][cycle] = d[2,:]
            mea['Rc'][cycle] = d[7,:]
            mea['Rce'][cycle] = d[8,:]
            mea['Pc'][cycle] = d[9,:]
            mea['Pce'][cycle] = d[10,:]
            mea['ac'][cycle] = d[11,:]
            mea['ace'][cycle] = d[12,:]
            mea['Ra'][cycle] = d[13,:]
            mea['Rae'][cycle] = d[14,:]
            mea['Pa'][cycle] = d[15,:]
            mea['Pae'][cycle] = d[16,:]
            mea['aa'][cycle] = d[17,:]
            mea['aae'][cycle] = d[18,:]

            mea['Ca'][cycle] = mea['Ra'][cycle]**((1-mea['aa'][cycle])/mea['aa'][cycle])*(mea['Pa'][cycle])**((1)/mea['aa'][cycle])
            mea['Cc'][cycle] = mea['Rc'][cycle]**((1-mea['ac'][cycle])/mea['ac'][cycle])*(mea['Pc'][cycle])**((1)/mea['ac'][cycle])


        for cycle, file in enumerate(mea['current_files']):
            currents = np.loadtxt(file,skiprows=0)
            mea['I'][cycle] = currents


def relaxis_fit_params(MEA, params):
    for mea in MEA:
        for param in params.keys():
            mea[param] = []
        skip_rows = 2
        file = mea["file"]
        with open(file, encoding="latin1") as f:
            df = pd.read_csv(f, skiprows=skip_rows, delimiter="\t")
        print(file)
        cycles = np.unique(
            df.loc[df["Free Variable"] == int(mea["name"]), "Free Variable 2"].values
        )
        mea["cycles"] = cycles
        for param, param_name in zip(params.keys(), params.values()):
            if param_name == "None":
                continue
            for cycle in cycles:
                mea[param].append(
                    df.loc[
                        (df["Free Variable 2"] == cycle)
                        & (df["Free Variable"] == int(mea["name"])),
                        param_name,
                    ].values
                )
        for cycle in mea["cycles"]:
            if "C" in params.keys():
                mea["C"].append(
                    (
                        mea["P"][cycle - 1]
                        * mea["R"][cycle - 1] ** (1 - mea["a"][cycle - 1])
                    )
                    ** (1 / mea["a"][cycle - 1])
                )
            else:
                mea["Cc"].append(
                    (
                        mea["Pc"][cycle - 1]
                        * mea["Rc"][cycle - 1] ** (1 - mea["ac"][cycle - 1])
                    )
                    ** (1 / mea["ac"][cycle - 1])
                )
                mea["Ca"].append(
                    (
                        mea["Pa"][cycle - 1]
                        * mea["Ra"][cycle - 1] ** (1 - mea["aa"][cycle - 1])
                    )
                    ** (1 / mea["aa"][cycle - 1])
                )


def freq_and_Z(df, val=1, freq_lims=[3, 2e4], control="Ewe"):
    freq = df.loc[(df["cycle number"] == val) & (df["freq/Hz"] != 0), "freq/Hz"].values
    if control == "Ewe":
        Z = (
            df.loc[
                (df["cycle number"] == val) & (df["freq/Hz"] != 0), "Re(Z)/Ohm"
            ].values
            - 1j
            * df.loc[
                (df["cycle number"] == val) & (df["freq/Hz"] != 0), "-Im(Z)/Ohm"
            ].values
        )
        E = np.mean(
            df.loc[(df["cycle number"] == val) & (df["freq/Hz"] != 0), "<Ewe>/V"]
        )
        I = np.mean(
            df.loc[(df["cycle number"] == val) & (df["freq/Hz"] != 0), "<I>/mA"]
        )
    else:
        Z = (
            df.loc[
                (df["cycle number"] == val) & (df["freq/Hz"] != 0), "Re(Zwe-ce)/Ohm"
            ].values
            - 1j
            * df.loc[
                (df["cycle number"] == val) & (df["freq/Hz"] != 0), "-Im(Zwe-ce)/Ohm"
            ].values
        )
        E = np.mean(
            df.loc[(df["cycle number"] == val) & (df["freq/Hz"] != 0), "<Ewe>/V"]
            - df.loc[(df["cycle number"] == val) & (df["freq/Hz"] != 0), "<Ece>/V"]
        )
        I = np.mean(
            df.loc[(df["cycle number"] == val) & (df["freq/Hz"] != 0), "<I>/mA"]
        )

    if np.diff(freq).any() < 0:
        print("problem")
    else:
        pass

    freq, Z = freq_and_Z_masked(freq, Z, freq_lims)
    return freq, Z, E, I


def freq_and_Z_masked(freq, Z, freq_lims):
    f_min = min(freq_lims)
    f_max = max(freq_lims)
    mask = (freq > f_min) & (freq <= f_max)
    mask &= np.isfinite(Z.real) & np.isfinite(Z.imag)
    freq = freq[mask]
    Z = Z[mask]
    return freq, Z


# class EISSpectrum:
#     def __init__(self, f, Z, I=None, metadata=None):
#         self.f = np.asarray(f)
#         self.Z = np.asarray(Z)
#         self.I = I
#         self.metadata = metadata or {}

#     # --- basic properties ---
#     @property
#     def omega(self):
#         return 2 * np.pi * self.f

#     @property
#     def Z_real(self):
#         return self.Z.real

#     @property
#     def Z_imag(self):
#         return self.Z.imag

#     @property
#     def magnitude(self):
#         return np.abs(self.Z)

#     @property
#     def phase(self):
#         return np.angle(self.Z, deg=True)

#     # --- preprocessing ---
#     def sort_by_frequency(self):
#         idx = np.argsort(self.f)
#         self.f = self.f[idx]
#         self.Z = self.Z[idx]

#     def remove_inductive_tail(self):
#         # example placeholder
#         pass

#     # --- DRT ---
#    # def compute_drt(self, method="tikhonov", **kwargs):
#         # plug your inversion routine here
#     #    tau, gamma = drt_solver(self.f, self.Z, method=method, **kwargs)
#     #    return tau, gamma

#     # --- plotting ---
#     def plot_nyquist(self, ax=None):
#         ax = ax or plt.gca()
#         ax.plot(self.Z_real, -self.Z_imag, 'o-')
#         ax.set_xlabel("Z' (Ω)")
#         ax.set_ylabel("-Z'' (Ω)")
#         return ax

#     def plot_bode(self, ax=None):
#         fig, ax = plt.subplots(2, 1, sharex=True)
#         ax[0].semilogx(self.f, self.magnitude)
#         ax[1].semilogx(self.f, self.phase)
#         return ax


def spectra_to_loops(df):
    unique_vals = np.unique(df["cycle number"].values)
    series = [[1]]
    i = 0
    t0, _ = we.data_average(df, "time/s", val=1)
    for val in unique_vals[1:]:
        t, _ = we.data_average(df, "time/s", val=val)
        if t - t0 > 2000:
            series.append([])
            i += 1
        series[i].append(int(val))
        t0 = t
    return np.array(series)


def calculate_Ro(file, cycles=[10]):
    skip_rows = we.get_header_lines(file)
    with open(file, "r", encoding="latin1") as f:
        df = pd.read_csv(f, skiprows=skip_rows - 1, delimiter="\t")

    unique_vals = np.unique(df["cycle number"].values)
    series = spectra_to_loops(df)
    if cycles == "all":
        cycles = unique_vals
    else:
        cycles = we.extend_cycles(cycles, series)

    drt = DRT()
    Ros = []
    Es = []
    for serie in series:
        Ros.append([])
        Es.append([])

    for val in cycles:
        f, Z = freq_and_Z(df, val=val)

        i, j = np.where(series == int(val))
        i = i[0]
        E, _ = we.data_average(df, "<Ewe>/V", val=val)
        Es[i].append(E)
        drt.fit_eis(f, Z)
        Ro = drt.predict_r_inf()
        Ros[i].append(Ro)
    return np.array(Ros), np.array(Es)


def calculate_Ro_files(files, cycles=[2, -2]):
    Ros = []
    ERs = []
    for file in files:
        file = glob(os.path.join(file + "*PEIS*" + ".mpt"))[0]
        Ro, ER = calculate_Ro(file, cycles=cycles)
        Ros.append(Ro)
        ERs.append(ER)
    return Ros, ERs


def get_drt(f, Z, method="hybrid"):
    """
    ## get_drt

    Compute the Distribution of Relaxation Times (DRT) from EIS data
    using the hybrid backend from hybdrt.

    **Input modes**
    - Single spectrum: f is one frequency vector and Z is one complex impedance vector.
    - Multiple spectra: Z is a list/tuple or 2D array-like of spectra; f can be shared
      across all spectra or provided as one frequency vector per spectrum.

    **Parameters**
    - f (array_like or sequence of array_like): Frequency data in Hz.
    - Z (array_like or sequence of array_like): Complex impedance spectrum/spectra.
    - method (str, default 'hybrid'): DRT solver backend. Supported values are
      'hybrid' and 'ridge'.

    **Returns**
    - Single spectrum: (tau, gamma) as arrays.
    - Multiple spectra: (tau_all, gamma_all) as lists of arrays.

    **Raises**
    - ValueError: If method is not recognized.
    """
    if method not in {"hybrid", "ridge"}:
        raise ValueError(f"Unsupported DRT method: {method}")

    drt = DRT() if method == "hybrid" else None

    def _compute_single_drt(f_single, z_single):
        if method == "hybrid":
            drt.fit_eis(f_single, z_single)
            tau_single = drt.get_tau_eval(20)
            gamma_single = drt.predict_drt(tau_single)
            Ro = drt.predict_r_inf()
            return tau_single, gamma_single, Ro

        # This follows EIS_fitting/eis_services.py::analyze_outliers. The
        # outlier call performs the Ridge inversion and populates the same
        # distribution/basis data used by the GUI's Ridge DRT display.
        inverter = Inverter()
        inverter.check_outliers(
            np.asarray(f_single),
            np.asarray(z_single),
            threshold=1,
            use_existing_fit=False,
        )
        distribution = next(iter(inverter.distributions))
        basis_tau = inverter.distributions[distribution]["tau"]
        minimum_log_tau = np.log10(np.min(basis_tau)) - 1
        maximum_log_tau = np.log10(np.max(basis_tau)) + 1
        tau_single = np.logspace(
            minimum_log_tau,
            maximum_log_tau,
            int(10 * (maximum_log_tau - minimum_log_tau) + 1),
        )
        gamma_single = inverter.predict_distribution(
            name=distribution,
            tau=tau_single,
        )
        return tau_single, np.asarray(gamma_single), float(inverter.R_inf)

    # Detect multiple spectra:
    # - Z as list/tuple of spectra
    # - or Z as 2D array-like (n_spectra, n_points)
    is_multi = False
    if isinstance(Z, (list, tuple)):
        is_multi = len(Z) > 0 and hasattr(Z[0], "__len__")
    elif hasattr(Z, "ndim"):
        is_multi = Z.ndim > 1

    if not is_multi:
        return _compute_single_drt(f, Z)

    Z_list = list(Z)

    # f can be shared for all spectra, or one frequency vector per spectrum
    if (
        isinstance(f, (list, tuple))
        and len(f) == len(Z_list)
        and hasattr(f[0], "__len__")
    ):
        f_list = list(f)
    else:
        f_list = [f] * len(Z_list)

    tau_all = []
    gamma_all = []
    Ro_all = []
    for f_i, Z_i in zip(f_list, Z_list):
        tau_i, gamma_i, Ro = _compute_single_drt(f_i, Z_i)
        tau_all.append(tau_i)
        gamma_all.append(gamma_i)
        Ro_all.append(Ro)

    return tau_all, gamma_all, Ro_all


def get_DRT(f, Z, method="hybrid"):
    """Compatibility entry point for :func:`get_drt`."""
    return get_drt(f, Z, method=method)


def capacitance(R, Q, alpha, Re=None, Qe=None, alphae=None):
    """
    Calculate the double-layer capacitance and its propagation error.

    Parameters
    ----------
    R : array_like or float
        Resistance value(s).
    Q : array_like or float
        Constant phase element (CPE) parameter(s).
    alpha : array_like or float
        Alpha parameter(s), typically between 0 and 1.
    Re : array_like or float, optional
        Error or uncertainty in resistance R.
    Qe : array_like or float, optional
        Error or uncertainty in parameter Q.
    alphae : array_like or float, optional
        Error or uncertainty in parameter alpha.

    Returns
    -------
    C : ndarray or float
        Calculated capacitance value(s).
    Ce : ndarray or float, optional
        Propagated error in capacitance, returned only if all of Re, Qe, and alphae are provided.
    """
    C = (Q ** (1 / alpha)) * (R ** ((1 - alpha) / alpha))

    if Re is not None and Qe is not None and alphae is not None:
        term_Q = (Qe / (alpha * Q)) ** 2
        term_R = ((1 - alpha) * Re / (alpha * R)) ** 2
        term_alpha = (np.log(Q * R) * alphae / (alpha**2)) ** 2

        Ce = C * np.sqrt(term_Q + term_R + term_alpha)
        return C, Ce
    else:
        return C


def find_outliers(f, Z, threshold=1):
    inv = Inverter()
    outliers = inv.check_outliers(f, Z, threshold=threshold, use_existing_fit=False)
    indices = np.ravel(outliers)
    return indices


def remove_outliers(f, Z, threshold=1):
    """
    Remove outliers from the impedance data based on the DRT inversion method.

    Parameters
    ----------
    f : array_like
        Frequency data (Hz).
    Z : array_like
        Complex impedance data.
    threshold : float, optional
        Threshold for outlier detection. Default is 1.

    Returns
    -------
    f_clean : ndarray
        Frequency data with outliers removed.
    Z_clean : ndarray
        Complex impedance data with outliers removed.
    """
    indices = find_outliers(f, Z, threshold=threshold)
    f = np.delete(f, indices)
    Z = np.delete(Z, indices)
    return f, Z


def tau(R, Q, alpha, Re=None, Qe=None, alphae=None):
    """
    Calculate the tau constant and its propagation error.

    Parameters
    ----------
    R : array_like or float
        Resistance value(s).
    Q : array_like or float
        Constant phase element (CPE) parameter(s).
    alpha : array_like or float
        Alpha parameter(s), typically between 0 and 1.
    Re : array_like or float, optional
        Error or uncertainty in resistance R.
    Qe : array_like or float, optional
        Error or uncertainty in parameter Q.
    alphae : array_like or float, optional
        Error or uncertainty in parameter alpha.

    Returns
    -------
    tau : ndarray or float
        Calculated tau value(s).
    taue : ndarray or float, optional
        Propagated error in tau, returned only if all of Re, Qe, and alphae are provided.
    """
    C = capacitance(R, Q, alpha)
    tau = R * C

    if Re is not None and Qe is not None and alphae is not None:
        C, Ce = capacitance(R, Q, alpha, Re, Qe, alphae)
        term_C = (Ce / C) ** 2
        term_R = (Re / R) ** 2

        taue = tau * np.sqrt(term_C + term_R)
        return tau, taue
    else:
        return tau


def fit_spectrum(
    f,
    Z,
    cir="R0-L0-p(R1,CPE1)",
    init=None,
    bounds=None,
    outliers=False,
    threshold=1,
    E=0,
    I=0,
    tau_sort=False,
):
    """
    Fit an impedance spectrum to a specified circuit model using CustomCircuit.

    Parameters
    ----------
    f : array_like
        Frequency data (Hz).
    Z : array_like
        Complex impedance data.
    circuit_str : str, optional
        Circuit string definition for fitting. Default is 'R0-L0-p(R1,CPE1)'.
    init_params : list or array_like, optional
        Initial guess for the circuit parameters.
        Default is [0.02, 1e-8, 0.1, 0.1, 0.9].
    bounds : tuple of array_like, optional
        Bounds for the circuit parameters as a tuple (lower_bounds, upper_bounds).
        Default is ([0.005, 1e-10, 1e-5, 1e-6, 0.6], [1, 1, 1000, 1, 1]).

    Returns
    -------
    params : dict
        Fitted parameter values.
    errors : dict
        Confidence intervals or errors for fitted parameters.
    """

    def _sort_parallel_rcpe_blocks(
        circuit_str, param_names, fitted_params, fitted_errors
    ):
        block_ids = re.findall(r"p\(\s*R(\d+)\s*,\s*CPE\1\s*\)", circuit_str)
        if len(block_ids) <= 1:
            return fitted_params, fitted_errors

        ordered_ids = list(dict.fromkeys(block_ids))

        r_idx = {}
        q_idx = {}
        a_idx = {}
        for idx, name in enumerate(param_names):
            m_r = re.fullmatch(r"R(\d+)", name)
            if m_r:
                r_idx[m_r.group(1)] = idx
                continue

            m_q = re.fullmatch(r"CPE(\d+)_0", name)
            if m_q:
                q_idx[m_q.group(1)] = idx
                continue

            m_a = re.fullmatch(r"CPE(\d+)_1", name)
            if m_a:
                a_idx[m_a.group(1)] = idx

        blocks = []
        for block_id in ordered_ids:
            if block_id not in r_idx or block_id not in q_idx or block_id not in a_idx:
                continue

            i_r = r_idx[block_id]
            i_q = q_idx[block_id]
            i_a = a_idx[block_id]
            R_val = fitted_params[i_r]
            Q_val = fitted_params[i_q]
            alpha_val = fitted_params[i_a]
            tau_val = R_val * capacitance(R_val, Q_val, alpha_val)
            blocks.append((tau_val, (i_r, i_q, i_a)))

        if len(blocks) <= 1:
            return fitted_params, fitted_errors

        # Order blocks by increasing tau (lowest first).
        sorted_blocks = sorted(blocks, key=lambda x: x[0])

        target_indices = []
        source_indices = []
        for old_block, new_block in zip(blocks, sorted_blocks):
            target_indices.extend(old_block[1])
            source_indices.extend(new_block[1])

        params_sorted = np.array(fitted_params, copy=True)
        errors_sorted = np.array(fitted_errors, copy=True)
        params_sorted[target_indices] = fitted_params[source_indices]
        errors_sorted[target_indices] = fitted_errors[source_indices]
        return params_sorted, errors_sorted

    if init is None:
        init = [0.02, 1e-8, 0.1, 0.1, 0.9]
    if bounds is None:
        bounds = ([0.005, 1e-10, 1e-5, 1e-6, 0.6], [1, 1, 1000, 1, 1])
    circuit = CustomCircuit(cir, initial_guess=init)

    if outliers:
        f, Z = remove_outliers(f, Z, threshold=threshold)
    circuit.fit(f, Z, bounds=bounds)

    params = np.asarray(circuit.parameters_)
    errors = np.asarray(circuit.conf_)
    param_names, _ = circuit.get_param_names()
    if tau_sort == True:
        params, errors = _sort_parallel_rcpe_blocks(cir, param_names, params, errors)

    params = np.array([E, I] + list(params))
    errors = np.array([E, I] + list(errors))
    return params, errors


def show_fit(f, cir, params, points=50, decades=(0, 0)):
    """
    Generate a model fit for an impedance spectrum using given circuit parameters.

    Parameters
    ----------
    f : array_like
        Frequency data (Hz).
    circuit_str : str
        Circuit string definition for prediction.
    params : dict or list or array_like
        Parameter values for the circuit model.
    points : int, optional
        Number of frequency points to generate for the fit curve.
        Default is 50.
    decades : tuple of two ints, optional
        Adjustment in decades for the frequency range:
        (how many decades to higher freqs, how many decades to lower freqs).
        Default is (0, 0).

    Returns
    -------
    f_fit : ndarray
        Frequencies used for predicted fit (Hz).
    Z_model : ndarray
        Complex impedance predicted by the circuit model at f_fit frequencies.
    """
    circuit = CustomCircuit(cir, initial_guess=params)
    f_fit = np.logspace(
        np.log10(f[0]) + decades[0], np.log10(f[-1]) - decades[1], points
    )
    Z_model = circuit.predict(f_fit)

    return f_fit, Z_model
