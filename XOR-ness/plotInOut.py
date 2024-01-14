import constants as c
import numpy as np
from ConfigPlot import ConfigPlot_EigenMode_DiffMass, ConfigPlot_YFixed_rec, ConfigPlot_DiffMass_SP
from MD_functions import MD_YFixed_ConstV_SP_SD, MD_VibrSP_ConstV_Yfixed_FixSpr2, MD_VibrSP_ConstV_Yfixed_FixSpr4
from MD_functions import MD_YFixed_ConstV_SP_SD_2
from DynamicalMatrix import DM_mass_Yfixed
from plot_functions import Line_single, Line_multi
from ConfigPlot import ConfigPlot_DiffMass3
import random
import matplotlib.pyplot as plt
import pickle
from os.path import exists

from switch_binary import switch

def showPacking(indices):

    n_col = 6
    n_row = 5
    N = n_col*n_row

    m1=1
    m2=10
    
    dphi_index = -1
    dphi = 10**dphi_index
    d0 = 0.1
    Lx = d0*n_col
    Ly = (n_row-1)*np.sqrt(3)/2*d0+d0
    
    x0 = np.zeros(N)
    y0 = np.zeros(N)
    
    phi0 = N*np.pi*d0**2/4/(Lx*Ly)
    d_ini = d0*np.sqrt(1+dphi/phi0)
    D = np.zeros(N)+d_ini
    #D = np.zeros(N)+d0 
    
    x0 = np.zeros(N)
    y0 = np.zeros(N)
    for i_row in range(1, n_row+1):
        for i_col in range(1, n_col+1):
            ind = (i_row-1)*n_col+i_col-1
            if i_row%2 == 1:
                x0[ind] = (i_col-1)*d0
            else:
                x0[ind] = (i_col-1)*d0+0.5*d0
            y0[ind] = (i_row-1)*np.sqrt(3)/2*d0
    y0 = y0+0.5*d0
    
    mass = np.zeros(N)+m2*indices
    mass = mass + m1*(1-indices)

    # specify the input ports and the output port
    SP_scheme = 0
    digit_in = SP_scheme//2
    digit_out = SP_scheme-2*digit_in 

    ind_in1 = int((n_col+1)/2)+digit_in - 1
    ind_in2 = ind_in1 + 2
    ind_out = int(N-int((n_col+1)/2)+digit_out)
    ind_fix = int((n_row+1)/2)*n_col-int((n_col+1)/2)
    
    ConfigPlot_DiffMass3(N, x0, y0, D, [Lx,Ly], mass, 0, '/Users/atoosa/Desktop/results/packing.pdf', ind_in1, ind_in2, ind_out)

def plotInOut(indices):
    
    #%% Initial Configuration
    k = 1
    m1 = 1
    m2 = 10
    
    n_col = 6
    n_row = 5
    N = n_col*n_row
    
    eigen_num = 33
    mark_plot = 0  
    
    
    dt_ratio = 40
    Nt_SD = 1e5
    Nt_MD = 1e5
    
    
    dphi_index = -1
    dphi = 10**dphi_index
    d0 = 0.1
    d_ratio = 1.1
    Lx = d0*n_col
    Ly = (n_row-1)*np.sqrt(3)/2*d0+d0
    
    x0 = np.zeros(N)
    y0 = np.zeros(N)
    
    phi0 = N*np.pi*d0**2/4/(Lx*Ly)
    d_ini = d0*np.sqrt(1+dphi/phi0)
    D = np.zeros(N)+d_ini
    #D = np.zeros(N)+d0 
    
    x0 = np.zeros(N)
    y0 = np.zeros(N)
    for i_row in range(1, n_row+1):
        for i_col in range(1, n_col+1):
            ind = (i_row-1)*n_col+i_col-1
            if i_row%2 == 1:
                x0[ind] = (i_col-1)*d0
            else:
                x0[ind] = (i_col-1)*d0+0.5*d0
            y0[ind] = (i_row-1)*np.sqrt(3)/2*d0
    y0 = y0+0.5*d0
    
    mass = np.zeros(N)+m2*indices
    mass = mass + m1*(1-indices)
    
    # Steepest Descent to get energy minimum
    if (exists('initialPos.dat')):
        with open('initialPos.dat', "rb") as f:
            x_ini = pickle.load(f)
            y_ini = pickle.load(f)
        f.close()
    else:
        x_ini,y_ini, p_now = MD_YFixed_ConstV_SP_SD_2(Nt_SD, N, x0, y0, D, mass, Lx, Ly)
        f = open('initialPos.dat', 'ab')
        pickle.dump(x_ini, f)
        pickle.dump(y_ini, f)
        f.close()
    # Steepest Descent to get energy minimum      
    #x_ini,y_ini, p_now = MD_YFixed_ConstV_SP_SD_2(Nt_SD, N, x0, y0, D, mass, Lx, Ly)

    # skip the steepest descent for now to save time
    #x_ini = x0
    #y_ini = y0

    w,v = DM_mass_Yfixed(N, x_ini, y_ini, D, mass, Lx, 0, Ly, k)        
    w = np.real(w)
    v = np.real(v)
    freq = np.sqrt(np.absolute(w))
    ind_sort = np.argsort(freq)
    freq = freq[ind_sort]
    v = v[:, ind_sort]
    ind = freq > 1e-4
    eigen_freq = freq[ind]
    eigen_mode = v[:, ind]
    w_delta = eigen_freq[1:] - eigen_freq[0:-1]
    index = np.argmax(w_delta)
    F_low_exp = eigen_freq[index]
    F_high_exp = eigen_freq[index+1]

    plt.figure(figsize=(6.4,4.8))
    ax = plt.axes()
    plt.scatter(np.arange(0, len(eigen_freq)), eigen_freq, marker='x', color='blue')
    plt.xlabel(r"Index $(k)$", fontsize=18)
    plt.ylabel(r"Frequency $(\omega)$", fontsize=18)
    plt.title("Frequency Spectrum", fontsize=18, fontweight="bold")
    plt.grid(color='skyblue', linestyle=':', linewidth=0.75)
    props = dict(facecolor='green', alpha=0.1)
    myText = r'$\omega_{low}=$'+"{:.2f}".format(F_low_exp)+"\n"+r'$\omega_{high}=$'+"{:.2f}".format(F_high_exp)+"\n"+r'$\Delta \omega=$'+"{:.2f}".format(max(w_delta))
    plt.text(0.78, 0.15, myText, horizontalalignment='center', verticalalignment='center', transform=ax.transAxes, fontsize=18, bbox=props)
    plt.hlines(y=7, xmin=0, xmax=60, linewidth=1, linestyle='dotted', color='brown', alpha=0.9)
    plt.text(5, 7.5, '$\omega=7$', fontsize=14, color='brown', alpha=0.9)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.tight_layout()
    plt.show()

    print("specs:")

    print(F_low_exp)
    print(F_high_exp)
    print(max(w_delta))


    SP_scheme = 0
    digit_in = SP_scheme//2
    digit_out = SP_scheme-2*digit_in 

    ind_in1 = int((n_col+1)/2)+digit_in - 1
    ind_in2 = ind_in1 + 2
    ind_out = int(N-int((n_col+1)/2)+digit_out)
    ind_fix = int((n_row+1)/2)*n_col-int((n_col+1)/2)

    B = 1
    Nt = 1e4

    # we are designing an and gait at this frequency
    Freq_Vibr = 7

    # case 1, input [1, 1]
    Amp_Vibr1 = 1e-2
    Amp_Vibr2 = 1e-2

    # changed the resonator to one in MD_functions file and vibrations in x direction
    freq_fft, fft_in1, fft_in2, fft_x_out, fft_y_out, mean_cont, nt_rec, Ek_now, Ep_now, cont_now = MD_VibrSP_ConstV_Yfixed_FixSpr2(k, B, Nt, N, x_ini, y_ini, D, mass, [Lx, Ly], Freq_Vibr, Amp_Vibr1, ind_in1, Amp_Vibr2, ind_in2, ind_out)

    ind = np.where(freq_fft>Freq_Vibr)
    index_=ind[0][0]
    # fft of the output port at the driving frequency
    out1 = fft_x_out[index_-1] + (fft_x_out[index_]-fft_x_out[index_-1])*((Freq_Vibr-freq_fft[index_-1])/(freq_fft[index_]-freq_fft[index_-1]))

    # fft of input1 at driving frequency
    inp1 = fft_in1[index_-1] + (fft_in1[index_]-fft_in1[index_-1])*((Freq_Vibr-freq_fft[index_-1])/(freq_fft[index_]-freq_fft[index_-1]))

    # fft of input2 at driving frequency
    inp2 = fft_in2[index_-1] + (fft_in2[index_]-fft_in2[index_-1])*((Freq_Vibr-freq_fft[index_-1])/(freq_fft[index_]-freq_fft[index_-1]))

    gain1 = out1/(inp1+inp2)

    fig = plt.figure(figsize=(6.4,4.8))
    ax = plt.axes()
    plt.plot(freq_fft, fft_in1, color='lightgreen', label='Input1', linestyle='solid', linewidth=5)
    plt.plot(freq_fft, fft_in2, color='blue', label='Input2', linestyle='dashed', linewidth=1)
    plt.plot(freq_fft, fft_x_out, color='red', label='Output', linestyle='solid', linewidth=2)
    plt.xlabel("Frequency", fontsize=18)
    plt.ylabel("Amplitude of FFT", fontsize=18)
    plt.title("input = 11", fontsize=18, fontweight="bold")
    plt.grid(color='skyblue', linestyle=':', linewidth=0.75)
    #plt.legend(loc='upper right', fontsize=16)
    #plt.axvline(x=Freq_Vibr, color='purple', linestyle='solid', alpha=0.5)
    myText = 'Gain='+"{:.3f}".format(gain1)
    plt.text(0.5, 0.9, myText, horizontalalignment='center', verticalalignment='center', transform=ax.transAxes, fontsize=18)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.tight_layout()
    plt.show()

    x_in1, x_in2, x_out = MD_VibrSP_ConstV_Yfixed_FixSpr4(k, B, Nt, N, x_ini, y_ini, D, mass, [Lx, Ly], Freq_Vibr, Amp_Vibr1, ind_in1, Amp_Vibr2, ind_in2, ind_out)

    fig = plt.figure(figsize=(6.4,4.8))
    ax = plt.axes()
    plt.plot(x_in1, color='lightgreen', label='Input1', linestyle='solid', linewidth=5)
    plt.plot(x_in2, color='blue', label='Input2', linestyle='dashed', linewidth=1)
    plt.plot(x_out, color='red', label='Output', linestyle='solid', linewidth=2)
    plt.xlabel("Time Steps", fontsize=18)
    plt.ylabel("Displacement", fontsize=18)
    plt.title("input = 11", fontsize=18, fontweight="bold")
    plt.grid(color='skyblue', linestyle=':', linewidth=0.75)
    #plt.legend(loc='upper right', fontsize=16)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.tight_layout()
    plt.show()

    # case 2, input [1, 0]
    Amp_Vibr1 = 1e-2
    Amp_Vibr2 = 0

    # changed the resonator to one in MD_functions file and vibrations in x direction
    freq_fft, fft_in1, fft_in2, fft_x_out, fft_y_out, mean_cont, nt_rec, Ek_now, Ep_now, cont_now = MD_VibrSP_ConstV_Yfixed_FixSpr2(k, B, Nt, N, x_ini, y_ini, D, mass, [Lx, Ly], Freq_Vibr, Amp_Vibr1, ind_in1, Amp_Vibr2, ind_in2, ind_out)

    ind = np.where(freq_fft>Freq_Vibr)
    index_=ind[0][0]
    # fft of the output port at the driving frequency
    out2 = fft_x_out[index_-1] + (fft_x_out[index_]-fft_x_out[index_-1])*((Freq_Vibr-freq_fft[index_-1])/(freq_fft[index_]-freq_fft[index_-1]))

    # fft of input1 at driving frequency
    inp1 = fft_in1[index_-1] + (fft_in1[index_]-fft_in1[index_-1])*((Freq_Vibr-freq_fft[index_-1])/(freq_fft[index_]-freq_fft[index_-1]))

    # fft of input2 at driving frequency
    inp2 = fft_in2[index_-1] + (fft_in2[index_]-fft_in2[index_-1])*((Freq_Vibr-freq_fft[index_-1])/(freq_fft[index_]-freq_fft[index_-1]))

    gain2 = out2/(inp1+inp2)

    fig = plt.figure(figsize=(6.4,4.8))
    ax = plt.axes()
    plt.plot(freq_fft, fft_in1, color='lightgreen', label='Input1', linestyle='solid', linewidth=3)
    plt.plot(freq_fft, fft_in2, color='blue', label='Input2', linestyle='dashed', linewidth=1)
    plt.plot(freq_fft, fft_x_out, color='red', label='Output', linestyle='solid', linewidth=2)
    plt.xlabel("Frequency", fontsize=18)
    plt.ylabel("Amplitude of FFT", fontsize=18)
    plt.title("input = 10", fontsize=18, fontweight="bold")
    plt.grid(color='skyblue', linestyle=':', linewidth=0.75)
    #plt.legend(loc='upper right', fontsize=18)
    #plt.axvline(x=Freq_Vibr, color='purple', linestyle='solid', alpha=0.5)
    myText = 'Gain='+"{:.3f}".format(gain2)
    plt.text(0.5, 0.9, myText, horizontalalignment='center', verticalalignment='center', transform=ax.transAxes, fontsize=18)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.tight_layout()
    plt.show()

    x_in1, x_in2, x_out = MD_VibrSP_ConstV_Yfixed_FixSpr4(k, B, Nt, N, x_ini, y_ini, D, mass, [Lx, Ly], Freq_Vibr, Amp_Vibr1, ind_in1, Amp_Vibr2, ind_in2, ind_out)

    fig = plt.figure(figsize=(6.4,4.8))
    ax = plt.axes()
    plt.plot(x_in1, color='lightgreen', label='Input1', linestyle='solid', linewidth=3)
    plt.plot(x_in2, color='blue', label='Input2', linestyle='dashed', linewidth=1)
    plt.plot(x_out, color='red', label='Output', linestyle='solid', linewidth=2)
    plt.xlabel("Time Steps", fontsize=18)
    plt.ylabel("Displacement", fontsize=18)
    plt.title("input = 10", fontsize=18, fontweight='bold')
    plt.grid(color='skyblue', linestyle=':', linewidth=0.75)
    #plt.legend(loc='upper right', fontsize=16)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.tight_layout()
    plt.show()

    # case 3, input [0, 1]
    Amp_Vibr1 = 0
    Amp_Vibr2 = 1e-2

    # changed the resonator to one in MD_functions file and vibrations in x direction
    freq_fft, fft_in1, fft_in2, fft_x_out, fft_y_out, mean_cont, nt_rec, Ek_now, Ep_now, cont_now = MD_VibrSP_ConstV_Yfixed_FixSpr2(k, B, Nt, N, x_ini, y_ini, D, mass, [Lx, Ly], Freq_Vibr, Amp_Vibr1, ind_in1, Amp_Vibr2, ind_in2, ind_out)

    ind = np.where(freq_fft>Freq_Vibr)
    index_=ind[0][0]
    # fft of the output port at the driving frequency
    out3 = fft_x_out[index_-1] + (fft_x_out[index_]-fft_x_out[index_-1])*((Freq_Vibr-freq_fft[index_-1])/(freq_fft[index_]-freq_fft[index_-1]))

    # fft of input1 at driving frequency
    inp1 = fft_in1[index_-1] + (fft_in1[index_]-fft_in1[index_-1])*((Freq_Vibr-freq_fft[index_-1])/(freq_fft[index_]-freq_fft[index_-1]))

    # fft of input2 at driving frequency
    inp2 = fft_in2[index_-1] + (fft_in2[index_]-fft_in2[index_-1])*((Freq_Vibr-freq_fft[index_-1])/(freq_fft[index_]-freq_fft[index_-1]))

    gain3 = out3/(inp1+inp2)

    fig = plt.figure(figsize=(6.4,4.8))
    ax = plt.axes()
    plt.plot(freq_fft, fft_in1, color='lightgreen', label='Input1', linestyle='solid', linewidth=3)
    plt.plot(freq_fft, fft_in2, color='blue', label='Input2', linestyle='dashed', linewidth=1)
    plt.plot(freq_fft, fft_x_out, color='red', label='Output', linestyle='solid', linewidth=2)
    plt.xlabel("Frequency", fontsize=18)
    plt.ylabel("Amplitude of FFT", fontsize=18)
    plt.title("input = 01", fontsize=18, fontweight='bold')
    plt.grid(color='skyblue', linestyle=':', linewidth=0.75)
    plt.legend(loc='upper right', fontsize=18)
    #plt.axvline(x=Freq_Vibr, color='purple', linestyle='solid', alpha=0.5)
    myText = 'Gain='+"{:.3f}".format(gain3)
    plt.text(0.5, 0.9, myText, horizontalalignment='center', verticalalignment='center', transform=ax.transAxes, fontsize=18)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.tight_layout()
    plt.show()

    x_in1, x_in2, x_out = MD_VibrSP_ConstV_Yfixed_FixSpr4(k, B, Nt, N, x_ini, y_ini, D, mass, [Lx, Ly], Freq_Vibr, Amp_Vibr1, ind_in1, Amp_Vibr2, ind_in2, ind_out)

    fig = plt.figure(figsize=(6.4,4.8))
    ax = plt.axes()
    plt.plot(x_in1, color='lightgreen', label='Input1', linestyle='solid', linewidth=3)
    plt.plot(x_in2, color='blue', label='Input2', linestyle='dashed', linewidth=1)
    plt.plot(x_out, color='red', label='Output', linestyle='solid', linewidth=2)
    plt.xlabel("Time Steps", fontsize=18)
    plt.ylabel("Displacement", fontsize=18)
    plt.title("input = 01", fontsize=18, fontweight='bold')
    plt.grid(color='skyblue', linestyle=':', linewidth=0.75)
    plt.legend(loc='upper right', fontsize=18)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.tight_layout()
    plt.show()

    print("gain1:")
    print(gain1)
    print("gain2:")
    print(gain2)
    print("gain3:")
    print(gain3)

    XOR = (gain2+gain3)/(2*gain1)
    
    return XOR

runs = c.RUNS
gens = c.numGenerations


# running the best individuals
temp = []
rubish = []
with open('savedRobotsLastGenAfpoSeed.dat', "rb") as f:
    for r in range(1, runs+1):
        # population of the last generation
        temp = pickle.load(f)
        # best individual of last generation
        best = temp[0]
        showPacking(best.indv.genome)
        print(plotInOut(best.indv.genome))
        temp = []
f.close()
