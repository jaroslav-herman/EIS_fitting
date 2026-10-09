import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from glob import glob
# from hybdrt.models import DRT
from scipy.interpolate import interp1d
import .basics as web
# import wepy as we
from impedance.models.circuits import CustomCircuit

def spectra_calculation(files,E, time_start = 0.5, time_end = 30, points = 60, plot = True, control = 'Ref'):
    Times = np.linspace(time_start,time_end,points)
    colors = web.get_colors(len(Times))
    spectra = []
    
    for j,t in enumerate(Times):
        Re_query = []
        Im_query= []
        I_query = []
        freq_query = []
        
        t_query = np.full(len(files), t)
        for i,file in enumerate(files):
    
            skip_rows = web.get_header_lines(file)
            with open(file,'r', encoding='latin1') as f:
                df = pd.read_csv(f,skiprows = skip_rows-1,delimiter = '\t')
            if control == 'Ref':
                Re = df['Re(Z)/Ohm']
                Im = df['-Im(Z)/Ohm']
            elif control == 'CE':
                Re = df['Re(Zwe-ce)/Ohm']
                Im = df['-Im(Zwe-ce)/Ohm']
            freq = df['freq/Hz']
            time = df['time/s']
            I = df['<I>/mA']
            Re_query.append(np.interp(t, time-time[0], Re))
            Im_query.append(np.interp(t, time-time[0], Im))
            I_query.append(np.interp(t, time-time[0], I))
            freq_query.append(np.interp(t, time-time[0], freq))

        spectra.append([freq_query,np.array(Re_query) - 1j*np.array(Im_query),np.array(I_query),E])
            
        if plot:
            plt.plot(Re_query,Im_query,'-',c=colors[j])
        
    if plot: 
        plt.gca().set_aspect('equal')
        plt.show()
    return spectra


def spectra_fitting(spectra, cir = 'R0-L0-p(R1,CPE1)-p(R2,CPE2)',
                    init = [0.015,10e-8, 0.025, 0.09, 0.8,0.05, 0.09, 0.8],
                    bounds = ([0.005,0, 0, 1e-6, 0.6,0, 1e-6, 0.6], [1,1, 1, 1, 1,1,1,1]),
                    freq_lims = [3,20000],
                   plot = True):
    colors = web.get_colors(len(spectra))
    params_all = []
    for spectrum,color in zip(spectra[1:],colors[1:]):
        if np.array(init).any() == 0:
            init = [0.015,10e-8, 0.025, 0.09, 0.8,0.05, 0.09, 0.8]
        else:
            pass
        
        circuit = CustomCircuit(cir, initial_guess=init)
        
        f, Z = we.eis.freq_and_Z_masked(np.array(spectrum[0]),np.array(spectrum[1]),freq_lims)
        
        circuit.fit(f, Z, bounds=bounds)
    
        params = circuit.parameters_
        errors = circuit.conf_
        # print(errors)
        # metadata = [np.mean(time_list[i]),np.mean(E_list[i]),np.mean(I_list[i])]
        params_all.append(np.concatenate((params,errors)))
        f_fit = np.logspace(np.log10(f[0]),np.log10(f[-1]),50)
        # print(f_fit)
        Z_model = circuit.predict(f_fit)
        init = params
        if plot:
            plt.plot(Z_model.real,-Z_model.imag,color='black',label='Fit')
            plt.scatter(Z.real,-Z.imag,s=20,color=color,label='Used for fit')
    
    if plot:
        plt.gca().set_aspect('equal')
        plt.show()
    params_all = np.array(params_all)
    return params_all