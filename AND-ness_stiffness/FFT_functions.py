"""

Simulator by Qikai Wu
https://journals.aps.org/pre/abstract/10.1103/PhysRevE.99.062901

Modified by Atoosa Parsa 

"""

import numpy as np
import matplotlib.pyplot as plt
from plot_functions import Line_multi, Line_single
#from numba import jit

def FFT_Fup(Nt, F, dt, Freq_Vibr):

    sampling_rate = 1/dt
    t = np.arange(Nt)*dt
    fft_size = Nt
    xs = F[:fft_size]
    xf = np.absolute(np.fft.rfft(xs)/fft_size)
    freqs = (2*np.pi)*np.linspace(0, sampling_rate/2, fft_size//2+1)
    
    ind = freqs<30
    freqs = freqs[ind]
    xf = xf[ind]

    if 1 == 0:
        Line_multi([freqs[1:], [Freq_Vibr, Freq_Vibr]], [xf[1:], [min(xf[1:]), max(xf[1:])]], ['o', 'r--'], 'Frequency', 'FFT', 'linear', 'log')        
            
    return freqs[1:], xf[1:]


def FFT_Fup_RealImag(Nt, F, dt, Freq_Vibr):

    sampling_rate = 1/dt
    t = np.arange(Nt)*dt
    fft_size = Nt
    xs = F[:fft_size]
    xf = np.fft.rfft(xs)/fft_size
    freqs = (2*np.pi)*np.linspace(0, sampling_rate/2, fft_size/2+1)
    
    ind = freqs<30
    freqs = freqs[ind]
    xf = xf[ind]
    xf_real = xf.real
    xf_imag = xf.imag

    if 1 == 0:
        Line_multi([freqs[1:], [Freq_Vibr, Freq_Vibr]], [xf[1:], [min(xf[1:]), max(xf[1:])]], ['o', 'r--'], 'Frequency', 'FFT')        
            
    return freqs[1:], xf_real[1:], xf_imag[1:]


#@jit
def vCorr_Cal(fft_size, Nt, y_raw):
    y_fft = np.zeros(fft_size)
    for jj in np.arange(fft_size):
        sum_vcf = 0
        sum_tt = 0
        count = 0
        for kk in np.arange(Nt-jj):
            count = count+1
            sum_vcf += y_raw[kk]*y_raw[kk+jj];
            sum_tt = sum_tt+y_raw[kk]*y_raw[kk];
        y_fft[jj] = sum_vcf/sum_tt;
    return y_fft
    

def FFT_vCorr(Nt, N, vx_rec, vy_rec, dt):
    sampling_rate = 1/dt
    fft_size = Nt-1
    freqs = (2*np.pi)*np.linspace(0, sampling_rate/2, fft_size/2+1)
    
    for ii in np.arange(2*N):
    #for ii in [0,4]:
        
        
        if np.mod(ii, 10) == 0:
            print('ii=%d\n' % (ii))                                
        if ii >= N:
            y_raw = vy_rec[:, ii-N]
        else:
            y_raw = vx_rec[:, ii]
                
        y_fft = vCorr_Cal(fft_size, Nt, y_raw)             
        if ii == 0:
            xf = np.absolute(np.fft.rfft(y_fft)/fft_size)
        else:
            xf += np.absolute(np.fft.rfft(y_fft)/fft_size)

    
    ind = freqs<30
    freqs = freqs[ind]
    xf = xf[ind]

    if 1 == 1:
        Line_single(freqs[1:], xf[1:], 'o', 'Frequency', 'FFT')        
            
    return freqs[1:], xf[1:]

def FFT_vCorr_3D(Nt, N, vx_rec, vy_rec, vz_rec, dt):
    sampling_rate = 1/dt
    fft_size = Nt-1
    freqs = (2*np.pi)*np.linspace(0, sampling_rate/2, fft_size/2+1)
    
    for ii in np.arange(3*N):
    #for ii in [0,4]:
        
        
        if np.mod(ii, 10) == 0:
            print('ii=%d\n' % (ii))                                
        if ii >= 2*N:
            y_raw = vz_rec[:, ii-2*N]
        elif ii < N:
            y_raw = vx_rec[:, ii]
        else:
            y_raw = vy_rec[:, ii-N]
                
        y_fft = vCorr_Cal(fft_size, Nt, y_raw)             
        if ii == 0:
            xf = np.absolute(np.fft.rfft(y_fft)/fft_size)
        else:
            xf += np.absolute(np.fft.rfft(y_fft)/fft_size)

    
    ind = freqs<30
    freqs = freqs[ind]
    xf = xf[ind]

    if 1 == 1:
        Line_single(freqs[1:], xf[1:], 'o', 'Frequency', 'FFT')        
            
    return freqs[1:], xf[1:]