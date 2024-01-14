"""

Simulator by Qikai Wu
https://journals.aps.org/pre/abstract/10.1103/PhysRevE.99.062901

Modified by Atoosa Parsa 

"""
import numpy as np
import time
#from numba import jit

from FFT_functions import FFT_Fup, FFT_vCorr
from plot_functions import Line_multi, Line_yy, Line_single
from ConfigPlot import ConfigPlot_YFixed_rec
import matplotlib.pyplot as plt

#import IPython.core.debugger
#dbg = IPython.core.debugger.Pdb()

#@jit
def force_YFixed(Fx, Fy, N, x, y, D, Lx, y_bot, y_up):
    
    Fup = 0
    Fbot = 0
    Ep = 0
    cont = 0
    cont_up = 0
    p_now = 0
    for nn in np.arange(N):
        d_up = y_up-y[nn]
        d_bot = y[nn]-y_bot
        r_now = 0.5*D[nn]
        
        if d_up<r_now:
            F = -(1-d_up/r_now)/(r_now)
            Fup -= F
            Fy[nn] += F
            Ep += (1/2)*(1-d_up/r_now)**2
            cont_up += 1
            cont += 1
            #dbg.set_trace()
            
        
        if d_bot<r_now:
            F = -(1-d_bot/r_now)/(r_now)
            Fbot += F
            Fy[nn] -= F
            Ep += (1/2)*(1-d_bot/r_now)**2
            cont += 1

        for mm in np.arange(nn+1, N):
            dy = y[mm]-y[nn]
            Dmn = 0.5*(D[mm]+D[nn])
            if abs(dy) < Dmn:
                dx = x[mm]-x[nn]
                dx = dx-round(dx/Lx)*Lx
                dmn = np.sqrt(dx**2+dy**2)
                if dmn < Dmn:
                    F = -(1-dmn/Dmn)/Dmn/dmn
                    Fx[nn] += F*dx
                    Fx[mm] -= F*dx
                    Fy[nn] += F*dy
                    Fy[mm] -= F*dy
                    Ep += (1/2)*(1-dmn/Dmn)**2  
                    cont += 1
                    p_now += (-F)*(dx**2+dy**2)
    return Fx, Fy, Fup, Fbot, Ep, cont, p_now, cont_up

def force_YFixed_DiffK_VL(Fx, Fy, N, x, y, D, Lx, y_bot, y_up, k_list, k_type, VL_list, VL_counter):
    
    Fup = 0
    Fbot = 0
    Ep = 0
    cont = 0
    cont_up = 0
    p_now = 0
    for nn in np.arange(N):
        d_up = y_up - y[nn]
        d_bot = y[nn] - y_bot
        r_now = 0.5 * D[nn]
        
        if d_up < r_now:
            F = -k_list[k_type[nn]] * (1 - d_up / r_now) / (r_now)
            Fup -= F
            Fy[nn] += F
            Ep += 0.5 * k_list[k_type[nn]] * (1 - d_up / r_now)**2
            cont_up += 1
            cont += 1
            #dbg.set_trace()
            
        
        if d_bot < r_now:
            F = -k_list[k_type[nn]] * (1 - d_bot / r_now) / (r_now)
            Fbot += F
            Fy[nn] -= F
            Ep += 0.5 * k_list[k_type[nn]] * (1 - d_bot / r_now)**2
            cont += 1

    for vl_idx in np.arange(VL_counter):
        nn = VL_list[vl_idx][0]
        mm = VL_list[vl_idx][1]
        dy = y[mm] - y[nn]
        Dmn = 0.5 * (D[mm] + D[nn])
        if abs(dy) < Dmn:
            dx = x[mm] - x[nn]
            dx = dx - round(dx / Lx) * Lx
            if abs(dx) < Dmn:
                dmn = np.sqrt(dx**2 + dy**2)
                if dmn < Dmn:
                    k = k_list[(k_type[nn] ^ k_type[mm]) + np.maximum(k_type[nn], k_type[mm])]
                    F = -k * (1 - dmn / Dmn) / Dmn / dmn
                    Fx[nn] += F * dx
                    Fx[mm] -= F * dx
                    Fy[nn] += F * dy
                    Fy[mm] -= F * dy
                    Ep += 0.5 * k * (1 - dmn / Dmn)**2  
                    cont += 1
                    p_now += (-F) * (dx**2 + dy**2)
    return Fx, Fy, Fup, Fbot, Ep, cont, p_now, cont_up


def force_YFixed_upDS(Fx, Fy, N, x, y, D, Lx, y_bot, y_up, ind_up):
    
    Fup = 0
    Fbot = 0
    Ep = 0
    cont = 0
    cont_up = 0
    p_now = 0
    for nn in np.arange(N):
        d_up = y_up-y[nn]
        d_bot = y[nn]-y_bot
        r_now = 0.5*D[nn]
        
        if ind_up[nn] == 1:
            F = -(1-d_up/r_now)/(r_now)
            Fup -= F
            Fy[nn] += F
            Ep += (1/2)*(1-d_up/r_now)**2
            #dbg.set_trace()
            
        if d_up<r_now:
            cont_up = cont_up+1
            cont += 1
        
        if d_bot<r_now:
            F = -(1-d_bot/r_now)/(r_now)
            Fbot += F
            Fy[nn] -= F
            Ep += (1/2)*(1-d_bot/r_now)**2
            cont += 1

        for mm in np.arange(nn+1, N):
            dy = y[mm]-y[nn]
            Dmn = 0.5*(D[mm]+D[nn])
            if abs(dy) < Dmn:
                dx = x[mm]-x[nn]
                dx = dx-round(dx/Lx)*Lx
                dmn = np.sqrt(dx**2+dy**2)
                if dmn < Dmn:
                    F = -(1-dmn/Dmn)/Dmn/dmn
                    Fx[nn] += F*dx
                    Fx[mm] -= F*dx
                    Fy[nn] += F*dy
                    Fy[mm] -= F*dy
                    Ep += (1/2)*(1-dmn/Dmn)**2  
                    cont += 1
                    p_now += (-F)*(dx**2+dy**2)
    return Fx, Fy, Fup, Fbot, Ep, cont, p_now, cont_up

#@jit
def force_Regular(Fx, Fy, N, x, y, D, Lx, Ly):        
    Ep = 0
    cont = 0
    p_now = 0
    for nn in np.arange(N):
        for mm in np.arange(nn+1, N):
            dy = y[mm]-y[nn]
            dy = dy-round(dy/Ly)*Ly
            Dmn = 0.5*(D[mm]+D[nn])
            if abs(dy) < Dmn:
                dx = x[mm]-x[nn]
                dx = dx-round(dx/Lx)*Lx
                dmn = np.sqrt(dx**2+dy**2)
                if dmn < Dmn:
                    F = -(1-dmn/Dmn)/Dmn/dmn
                    Fx[nn] += F*dx
                    Fx[mm] -= F*dx
                    Fy[nn] += F*dy
                    Fy[mm] -= F*dy
                    Ep += (1/2)*(1-dmn/Dmn)**2   
                    cont += 1
                    p_now += (-F)*(dx**2+dy**2)
    return Fx, Fy, Ep, cont, p_now

def MD_UpDownFixed_SD(N, x0, y0, D0, m0, L):    
    
    dt = min(D0)/40
    Nt = int(1e4)
    Ep = np.zeros(Nt)
    F_up = np.zeros(Nt)
    F_bot = np.zeros(Nt)
    F_tot = np.zeros(Nt)
    Fup_now = 0
    
    vx = np.zeros(N)
    vy = np.zeros(N)
    
    x = np.array(x0)
    y = np.array(y0)
    
    t_start = time.time()
    for nt in np.arange(Nt):
        Fx = np.zeros(N)
        Fy = np.zeros(N)
        
        Fx, Fy, Fup_now, Fbot_now, Ep_now, cont_now, p_now = force_YFixed(Fx, Fy, N, x, y, D0, L[0], 0, L[1])
        #Fx, Fy, Ep_now, cont_now, p_now = force_Regular(Fx, Fy, N, x, y, D0, L[0], L[1])
        F_up[nt] = Fup_now
        F_bot[nt] = Fbot_now
        Ep[nt] = Ep_now
        vx = np.divide(Fx, m0)
        vy = np.divide(Fy, m0)
        x += vx*dt
        y += vy*dt
        F_tot[nt] = sum(np.absolute(Fx)+np.absolute(Fy))
    t_end = time.time()
    print ("time=%.3e" %(t_end-t_start))
    
    if 1 == 0:
        # Plot the amplitide of F
        Line_single(range(Nt), F_tot[0:Nt], '-', 't', 'Ftot', 'log', yscale='log')        
    
    return x, y
    
 
    
def MD_VibrBot_ForceUp(N, x_ini, y_ini, D0, m0, L, Freq_Vibr, Amp_Vibr):
    
    dt = min(D0)/40
    Nt = int(5e4)
    Ep = np.zeros(Nt)
    Ek = np.zeros(Nt)
    F_up = np.zeros(Nt)
    cont = np.zeros(Nt)
    
    y_bot = Amp_Vibr*np.sin(Freq_Vibr*dt*np.arange(Nt)+1.5*np.pi)+Amp_Vibr
    #y_bot = Amp_Vibr*np.sin(Freq_Vibr*dt*np.arange(Nt))    
    
    vx = np.zeros(N)
    vy = np.zeros(N)
    
    if 1 == 0:
        y_bot = np.zeros(Nt)
        vx = np.random.rand(N)
        vy = np.random.rand(N)
        T_rd = sum(0.5*np.multiply(m0,np.multiply(vx, vx)+np.multiply(vy, vy)))
        T_set = 1e-6
        vx = vx*np.sqrt(N*T_set/T_rd)
        vy = vy*np.sqrt(N*T_set/T_rd)
    
    ax_old = np.zeros(N)
    ay_old = np.zeros(N)
    
    x = np.array(x_ini)
    y = np.array(y_ini)
    
    #t_start = time.time()
    for nt in np.arange(Nt):
        x = x+vx*dt+ax_old*dt**2/2;  # first step in Verlet integration
        y = y+vy*dt+ay_old*dt**2/2;

        Fx = np.zeros(N)
        Fy = np.zeros(N)                        
        Fx, Fy, Fup_now, Fbot_now, Ep_now, cont_now, p_now = force_YFixed(Fx, Fy, N, x, y, D0, L[0], y_bot[nt], L[1])
        #Fx, Fy, Ep_now, cont_now, p_now = force_Regular(Fx, Fy, N, x, y, D0, L[0], L[1])
        F_up[nt] = Fup_now
        Ep[nt] = Ep_now
        cont[nt] = cont_now
        
        ax = np.divide(Fx, m0);
        ay = np.divide(Fy, m0);
        vx = vx+(ax_old+ax)*dt/2;  # second step in Verlet integration
        vy = vy+(ay_old+ay)*dt/2;        
        ax_old = ax;
        ay_old = ay;
        Ek[nt] = sum(0.5*np.multiply(m0,np.multiply(vx, vx)+np.multiply(vy, vy)))
        
    #t_end = time.time()
    #print ("time=%.3e" %(t_end-t_start))
    CB_ratio = min(cont)/max(cont)
    print ("freq=%f, cont_min/cont_max=%f\n" %(Freq_Vibr, CB_ratio))
    
    freq_now, fft_now = FFT_Fup(Nt, F_up[:Nt], dt, Freq_Vibr)        
    
    # Plot the amplitide of F
    if 1 == 1:
        Line_yy([dt*range(Nt), dt*range(Nt)], [F_up[0:Nt],y_bot[0:Nt]], ['-', ':'], 't', ['$F_{up}$', '$y_{bottom}$'])      

        Etot = Ep[1:Nt]+Ek[1:Nt]
        xdata = [dt*range(Nt), dt*range(Nt), dt*range(Nt-1)]
        ydata = [Ep[0:Nt], Ek[0:Nt], Etot]
        line_spec = ['--', ':', 'r-']
        Line_multi(xdata, ydata, line_spec, 't', '$E$', 'log')                
        print("std(Etot)=%e\n" %(np.std(Etot)))

        
        #dt2 = 1e-3
        #xx = np.arange(0, 5, dt2)
        #yy = np.sin(50*xx)+np.sin(125*xx)
        #print("dt=%e, w=%f\n" % (dt, Freq_Vibr))
        FFT_Fup(Nt, F_up[:Nt], dt, Freq_Vibr)
        #FFT_Fup(yy.size, yy, dt2, 50)
        
    return freq_now, fft_now, np.mean(cont)
    
def MD_Periodic_equi(Nt, N, x0, y0, D0, m0, L, T_set, V_em, n_em):
    
    dt = min(D0)/40
    Nt = int(Nt)
    Ep = np.zeros(Nt)
    Ek = np.zeros(Nt)
    cont = np.zeros(Nt)
    vx_rec = np.zeros([Nt, N])
    vy_rec = np.zeros([Nt, N])
 

    vx = np.zeros(N)
    vy = np.zeros(N)
    for ii in np.arange(n_em):
    #for ii in [60]:
        ind1 = 2*np.arange(N)
        ind2 = ind1+1
        vx += V_em[ind1, ii]
        vy += V_em[ind2, ii]
        
    T_rd = sum(0.5*np.multiply(m0,np.multiply(vx, vx)+np.multiply(vy, vy)))
    vx = vx*np.sqrt(N*T_set/T_rd)
    vy = vy*np.sqrt(N*T_set/T_rd)
    
    ax_old = np.zeros(N)
    ay_old = np.zeros(N)
    
    x = np.array(x0)
    y = np.array(y0)
    
    t_start = time.time()
    for nt in np.arange(Nt):
        x = x+vx*dt+ax_old*dt**2/2;  # first step in Verlet integration
        y = y+vy*dt+ay_old*dt**2/2;

        Fx = np.zeros(N)
        Fy = np.zeros(N)                        
        Fx, Fy, Ep_now, cont_now, p_now = force_Regular(Fx, Fy, N, x, y, D0, L[0], L[1])
        Ep[nt] = Ep_now
        cont[nt] = cont_now
        
        ax = np.divide(Fx, m0);
        ay = np.divide(Fy, m0);
        vx = vx+(ax_old+ax)*dt/2;  # second step in Verlet integration
        vy = vy+(ay_old+ay)*dt/2;        
        ax_old = ax;
        ay_old = ay;
        Ek[nt] = sum(0.5*np.multiply(m0,np.multiply(vx, vx)+np.multiply(vy, vy)))
        vx_rec[nt] = vx
        vy_rec[nt] = vy
        
    t_end = time.time()
    print ("time=%.3e" %(t_end-t_start))
    CB_ratio = min(cont)/max(cont)
    
    print ("cont_min/cont_max=%f\n" %(CB_ratio))
    
    #Etot = Ep[1:Nt]+Ek[1:Nt]
    #xdata = [dt*range(Nt), dt*range(Nt), dt*range(Nt-1)]
    #ydata = [Ep[0:Nt], Ek[0:Nt], Etot]
    #line_spec = ['--', ':', 'r-']
    #Line_multi(xdata, ydata, line_spec, 't', '$E$', 'log', 'log')   
    
    
    freq_now, fft_now = FFT_vCorr(Nt, N, vx_rec, vy_rec, dt)        
    return freq_now, fft_now, np.mean(cont)
    
def MD_YFixed_ConstP_SD(Nt, N, x0, y0, D0, m0, L, F0_up):    
    
    dt = D0[0]/40
    Nt = int(Nt)
    #Nt = int(5e6)
    #Nt = int(5e2)
    Ep = np.zeros(Nt)
    F_up = np.zeros(Nt)
    F_bot = np.zeros(Nt)
    F_tot = np.zeros(Nt)
    Fup_now = 0
    y_up = y0[N]
    
    vx = np.zeros(N+1)
    vy = np.zeros(N+1)
    
    x = np.array(x0)
    y = np.array(y0)
    
    t_start = time.time()
    for nt in np.arange(Nt):
        
        Fx = np.zeros(N)
        Fy = np.zeros(N)
        
        Fx, Fy, Fup_now, Fbot_now, Ep_now, cont_now, p_now, cont_up_now = force_YFixed(Fx, Fy, N, x[0:N], y[0:N], D0[0:N], L[0], 0, y[N])
        F_up[nt] = Fup_now+F0_up
        F_bot[nt] = Fbot_now
        Ep[nt] = Ep_now+(y_up-y[N])*F0_up
        vx = 0.1*np.divide(np.append(Fx,0), m0)
        vy = 0.1*np.divide(np.append(Fy, F_up[nt]), m0)
        x += vx*dt
        y += vy*dt
        F_tot[nt] = sum(np.absolute(Fx)+np.absolute(Fy))
        #print("nt=%d, Fup=%e, Fup_tot=%e\n" % (nt, Fup_now, F_up[nt]))
        #dbg.set_trace()
    t_end = time.time()
    print ("F_tot=%.3e\n" %(F_tot[nt]))
    print ("time=%.3e" %(t_end-t_start))
    
    
    if 1 == 0:
        # Plot the amplitide of F
        Line_single(range(Nt), F_tot[0:Nt], '-', 't', 'Ftot', 'log', yscale='log')
        #Line_single(range(Nt), -F_up[0:Nt], '-', 't', 'Fup', 'log', yscale='log')
        #Line_single(range(Nt), Ep[0:Nt], '-', 't', 'Ep', 'log', yscale='linear')         
    
    return x, y, p_now

def MD_VibrBot_DispUp_ConstP(mark_upDS, N, x_ini, y_ini, D0, m0, L, Freq_Vibr, Amp_Vibr, F0_up):
    
    dt = D0[0]/40
    B = 0.1 # damping coefficient
    Nt = int(5e7) 
    #Nt = int(5e2) 
    Ep = np.zeros(Nt)
    Ek = np.zeros(Nt)
    y_up0 = y_ini[N]
    y_up = np.zeros(Nt)    
    F_up = np.zeros(Nt)
    cont = np.zeros(Nt)
    cont_up = np.zeros(Nt)
    p = np.zeros(Nt)
    
    y_bot = Amp_Vibr*np.sin(Freq_Vibr*dt*np.arange(Nt)+1.5*np.pi)+Amp_Vibr
    #y_bot = Amp_Vibr*np.sin(Freq_Vibr*dt*np.arange(Nt))    
    
    vx = np.zeros(N+1)
    vy = np.zeros(N+1)
    
    # for test
    if 1 == 0:
        y_bot = np.zeros(Nt)
        vx = np.random.rand(N+1)        
        vx[N] = 0
        vy = np.random.rand(N+1)
        vy[N] = 0
        T_rd = sum(0.5*np.multiply(m0,np.multiply(vx, vx)+np.multiply(vy, vy)))
        T_set = 1e-6
        vx = vx*np.sqrt(N*T_set/T_rd)
        vy = vy*np.sqrt(N*T_set/T_rd)
    
    ax_old = np.zeros(N+1)
    ay_old = np.zeros(N+1)
    
    x = np.array(x_ini)
    y = np.array(y_ini)
    
    #t_start = time.time()
    
    if mark_upDS == 1:
        ind_up = np.zeros(N)
        for ii in np.arange(N):
            d_up = y[N]-y[ii]
            r_now = 0.5*D0[ii]        
            if d_up<r_now:
                ind_up[ii] = 1
            
            
    for nt in np.arange(Nt):
        x = x+vx*dt+ax_old*dt**2/2;  # first step in Verlet integration
        y = y+vy*dt+ay_old*dt**2/2;
        y_up[nt] = y[N]
        
        Fx = np.zeros(N)
        Fy = np.zeros(N) 

        if mark_upDS == 0:                       
            Fx, Fy, Fup_now, Fbot_now, Ep_now, cont_now, p_now, cont_up_now = force_YFixed(Fx, Fy, N, x[0:N], y[0:N], D0[0:N], L[0], y_bot[nt], y[N])
        elif mark_upDS == 1:
            Fx, Fy, Fup_now, Fbot_now, Ep_now, cont_now, p_now, cont_up_now = force_YFixed_upDS(Fx, Fy, N, x[0:N], y[0:N], D0[0:N], L[0], y_bot[nt], y[N], ind_up)

        F_up[nt] = Fup_now+F0_up        
        Ep[nt] = Ep_now+(y_up0-y[N])*F0_up
        cont[nt] = cont_now
        cont_up[nt] = cont_up_now
        p[nt] = p_now
        
        Fx_all = np.append(Fx,0)-B*vx
        Fy_all = np.append(Fy, F_up[nt])-B*vy
        
        ax = np.divide(Fx_all, m0)
        ay = np.divide(Fy_all, m0)
        vx = vx+(ax_old+ax)*dt/2;  # second step in Verlet integration
        vy = vy+(ay_old+ay)*dt/2;        
        ax_old = ax;
        ay_old = ay;
        Ek[nt] = sum(0.5*np.multiply(m0,np.multiply(vx, vx)+np.multiply(vy, vy)))
        
    #t_end = time.time()
    #print ("time=%.3e" %(t_end-t_start))
    CB_ratio = min(cont)/max(cont)
    print ("freq=%f, cont_min/cont_max=%f\n" %(Freq_Vibr, CB_ratio))
    
    y_up = y_up-y_up0
    freq_y, fft_y = FFT_Fup(int(Nt/2), y_up[int(Nt/2):Nt], dt, Freq_Vibr)
    #freq_y, fft_y_real, fft_y_imag = FFT_Fup_RealImag(int(Nt/2), y_up[int(Nt/2):Nt], dt, Freq_Vibr) 
    freq_bot, fft_bot = FFT_Fup(int(Nt/2), y_bot[int(Nt/2):Nt], dt, Freq_Vibr)  
            
    # plot the energy to see when the system reaches steady state
    if 1 == 0:
        Etot = Ep+Ek
        nt_start = int(1e3)
        xdata = [range(nt_start, Nt), range(nt_start, Nt), range(Nt)]
        ydata = [Ep[nt_start:Nt], Ek[nt_start:Nt], Etot]
        line_spec = [':', ':', 'r-']
        Line_multi(xdata, ydata, line_spec, 't', '$E$', 'linear', 'log')
    
    # Plot the amplitide of F
    if 1 == 0:
        Line_yy([dt*range(Nt), dt*range(Nt)], [F_up[0:Nt],y_bot[0:Nt]], ['-', ':'], 't', ['$F_{up}$', '$y_{bottom}$'])      
        Line_yy([dt*range(Nt), dt*range(Nt)], [y_up[0:Nt],y_bot[0:Nt]], ['-', ':'], 't', ['$y_{up}$', '$y_{bottom}$'])
        Line_single(range(Nt), p[0:Nt], '-', 't', 'p', 'log', 'linear')
        Etot = Ep[1:Nt]+Ek[1:Nt]
        xdata = [dt*range(Nt), dt*range(Nt), dt*range(Nt-1)]
        ydata = [Ep[0:Nt], Ek[0:Nt], Etot]
        line_spec = ['--', ':', 'r-']
        #Line_multi(xdata, ydata, line_spec, 't', '$E$', 'log')                
        print("std(Etot)=%e\n" %(np.std(Etot)))

    return freq_y, fft_y, freq_bot, fft_bot, np.mean(cont), np.mean(cont_up)
    #return freq_y, fft_y_real, fft_y_imag, freq_bot, fft_bot, np.mean(cont)
    
    
def MD_VibrBot_DispUp_ConstP_ConfigRec(N, x_ini, y_ini, D0, m0, L, Freq_Vibr, Amp_Vibr, F0_up, fn):
    
    dt = D0[0]/40
    B = 0.1 # damping coefficient
    Nt = int(5e6)
    nt_rec = np.linspace(Nt-5e4, Nt, 500)
        
    #Nt = int(1e4)
    #nt_rec = np.linspace(0.5*Nt, Nt, 50)
    
    nt_rec = nt_rec.astype(int)
    ind_nt = 0
    Ep = np.zeros(Nt)
    Ek = np.zeros(Nt)
    y_up0 = y_ini[N]
    y_up = np.zeros(Nt)    
    F_up = np.zeros(Nt)
    cont = np.zeros(Nt)
    cont_up = np.zeros(Nt)
    p = np.zeros(Nt)
    
    y_bot = Amp_Vibr*np.sin(Freq_Vibr*dt*np.arange(Nt)+1.5*np.pi)+Amp_Vibr
    #y_bot = Amp_Vibr*np.sin(Freq_Vibr*dt*np.arange(Nt))    
    
    vx = np.zeros(N+1)
    vy = np.zeros(N+1)
    
    ax_old = np.zeros(N+1)
    ay_old = np.zeros(N+1)
    
    x = np.array(x_ini)
    y = np.array(y_ini)
    
    #t_start = time.time()
    for nt in np.arange(Nt):
        
        if nt == nt_rec[ind_nt]:                        
            ConfigPlot_YFixed_rec(N, x[0:N], y[0:N], D0[0:N], L[0], y[N], y_bot[nt], m0[0:N], ind_nt, fn)
            ind_nt += 1

        
        x = x+vx*dt+ax_old*dt**2/2;  # first step in Verlet integration
        y = y+vy*dt+ay_old*dt**2/2;
        y_up[nt] = y[N]
        
        Fx = np.zeros(N)
        Fy = np.zeros(N) 
                     
        Fx, Fy, Fup_now, Fbot_now, Ep_now, cont_now, p_now, cont_up_now = force_YFixed(Fx, Fy, N, x[0:N], y[0:N], D0[0:N], L[0], y_bot[nt], y[N])

        F_up[nt] = Fup_now+F0_up        
        Ep[nt] = Ep_now+(y_up0-y[N])*F0_up
        cont[nt] = cont_now
        cont_up[nt] = cont_up_now
        p[nt] = p_now
        
        Fx_all = np.append(Fx,0)-B*vx
        Fy_all = np.append(Fy, F_up[nt])-B*vy
        
        ax = np.divide(Fx_all, m0)
        ay = np.divide(Fy_all, m0)
        vx = vx+(ax_old+ax)*dt/2;  # second step in Verlet integration
        vy = vy+(ay_old+ay)*dt/2;        
        ax_old = ax;
        ay_old = ay;
        Ek[nt] = sum(0.5*np.multiply(m0,np.multiply(vx, vx)+np.multiply(vy, vy)))
        
    #t_end = time.time()
    #print ("time=%.3e" %(t_end-t_start))
    CB_ratio = min(cont)/max(cont)
    print ("freq=%f, cont_min/cont_max=%f\n" %(Freq_Vibr, CB_ratio))
    
    y_up = y_up-y_up0
    freq_y, fft_y = FFT_Fup(int(Nt/2), y_up[int(Nt/2):Nt], dt, Freq_Vibr) 
    freq_bot, fft_bot = FFT_Fup(int(Nt/2), y_bot[int(Nt/2):Nt], dt, Freq_Vibr)
    
    return freq_y, fft_y, freq_bot, fft_bot, np.mean(cont), np.mean(cont_up)

def MD_VibrBot_DispUp_ConstP_EkCheck(Nt, mark_upDS, N, x_ini, y_ini, D0, m0, L, Freq_Vibr, Amp_Vibr, F0_up):
    
    dt = D0[0]/40
    B = 0.1 # damping coefficient
    Nt = int(Nt)
    #Nt = int(5e7) 
    #Nt = int(1e4) 
    
    nt_rec = np.linspace(0, Nt, int(Nt/5e4)+1)
    Ek_now = np.array(0)
    Ek_up_now = np.array(0)
    Ep_now = np.array(0)
    Ep_up_now = np.array(0)
        
    #nt_rec = np.linspace(0.5*Nt, Nt, 50)
    
    nt_rec = nt_rec.astype(int)
    
    Ep = np.zeros(Nt)
    Ep_up = np.zeros(Nt)
    Ek = np.zeros(Nt)
    Ek_up = np.zeros(Nt)
    y_up0 = y_ini[N]
    y_up = np.zeros(Nt)    
    F_up = np.zeros(Nt)
    cont = np.zeros(Nt)
    cont_up = np.zeros(Nt)
    p = np.zeros(Nt)
    
    y_bot = Amp_Vibr*np.sin(Freq_Vibr*dt*np.arange(Nt)+1.5*np.pi)+Amp_Vibr
  
    vx = np.zeros(N+1)
    vy = np.zeros(N+1)    
    
    ax_old = np.zeros(N+1)
    ay_old = np.zeros(N+1)
    
    x = np.array(x_ini)
    y = np.array(y_ini)
    
    #t_start = time.time()
    
    if mark_upDS == 1:
        ind_up = np.zeros(N)
        for ii in np.arange(N):
            d_up = y[N]-y[ii]
            r_now = 0.5*D0[ii]        
            if d_up<r_now:
                ind_up[ii] = 1
            
            
    for nt in np.arange(Nt):
        x = x+vx*dt+ax_old*dt**2/2;  # first step in Verlet integration
        y = y+vy*dt+ay_old*dt**2/2;
        y_up[nt] = y[N]
        
        Fx = np.zeros(N)
        Fy = np.zeros(N) 

        if mark_upDS == 0:                       
            Fx, Fy, Fup_now, Fbot_now, Ep_now, cont_now, p_now, cont_up_now = force_YFixed(Fx, Fy, N, x[0:N], y[0:N], D0[0:N], L[0], y_bot[nt], y[N])
        elif mark_upDS == 1:
            Fx, Fy, Fup_now, Fbot_now, Ep_now, cont_now, p_now, cont_up_now = force_YFixed_upDS(Fx, Fy, N, x[0:N], y[0:N], D0[0:N], L[0], y_bot[nt], y[N], ind_up)

        F_up[nt] = Fup_now+F0_up        
        Ep[nt] = Ep_now+(y_up0-y[N])*F0_up
        Ep_up[nt] = (y_up0-y[N])*F0_up
        cont[nt] = cont_now
        cont_up[nt] = cont_up_now
        p[nt] = p_now
        
        Fx_all = np.append(Fx,0)-B*vx
        Fy_all = np.append(Fy, F_up[nt])-B*vy
        
        ax = np.divide(Fx_all, m0)
        ay = np.divide(Fy_all, m0)
        vx = vx+(ax_old+ax)*dt/2;  # second step in Verlet integration
        vy = vy+(ay_old+ay)*dt/2;        
        ax_old = ax;
        ay_old = ay;
        Ek[nt] = sum(0.5*np.multiply(m0,np.multiply(vx, vx)+np.multiply(vy, vy)))
        Ek_up[nt] = 0.5*m0[N]*(vx[N]**2+vy[N]**2)
    
    for ii in np.arange(len(nt_rec)-1):
        Ek_now = np.append(Ek_now, np.mean(Ek[nt_rec[ii]:nt_rec[ii+1]]))
        Ek_up_now = np.append(Ek_up_now, np.mean(Ek_up[nt_rec[ii]:nt_rec[ii+1]]))
        Ep_now = np.append(Ep_now, np.mean(Ep[nt_rec[ii]:nt_rec[ii+1]]))
        Ep_up_now = np.append(Ep_up_now, np.mean(Ep_up[nt_rec[ii]:nt_rec[ii+1]]))

    #t_end = time.time()
    #print ("time=%.3e" %(t_end-t_start))
    CB_ratio = min(cont)/max(cont)
    print ("freq=%f, cont_min/cont_max=%f\n" %(Freq_Vibr, CB_ratio))
    
    y_up = y_up-np.mean(y_up)
    y_up = y_up/np.mean(np.absolute(y_up))
    freq_y, fft_y = FFT_Fup(int(Nt/2), y_up[int(Nt/2):Nt], dt, Freq_Vibr)
    freq_bot, fft_bot = FFT_Fup(int(Nt/2), y_bot[int(Nt/2):Nt], dt, Freq_Vibr)  
            
    return freq_y, fft_y, freq_bot, fft_bot, np.mean(cont), np.mean(cont_up)

#@jit
def force_YFixed_collision_ConstP(beta, Fx, Fy, N, x, y, vx, vy, D, Lx, y_bot, v_bot, y_up):
    
    Fup = 0
    Fbot = 0
    Ep = 0
    cont = 0
    cont_up = 0
    p_now = 0
    #betta = 1
    for nn in np.arange(N):
        d_up = y_up-y[nn]
        d_bot = y[nn]-y_bot
        r_now = 0.5*D[nn]
        
        if d_up<r_now:
            F = -(1-d_up/r_now)/(r_now)
            Fup -= F
            Fy[nn] += F            
            
            dvy = vy[N]-vy[nn]
            FD = beta*dvy
            #FD = np.absolute(FD)
            Fy[nn] += FD
            Fup -= FD
            
            Ep += (1/2)*(1-d_up/r_now)**2
            cont_up += 1
            cont += 1
            #dbg.set_trace()
            
        
        if d_bot<r_now:
            F = -(1-d_bot/r_now)/(r_now)
            Fbot += F
            Fy[nn] -= F
            
            dvy = v_bot-vy[nn]
            FD = beta*dvy
            Fy[nn] += FD
            
            Ep += (1/2)*(1-d_bot/r_now)**2
            cont += 1

        for mm in np.arange(nn+1, N):
            dy = y[mm]-y[nn]
            Dmn = 0.5*(D[mm]+D[nn])
            if abs(dy) < Dmn:
                dx = x[mm]-x[nn]
                dx = dx-round(dx/Lx)*Lx
                dmn = np.sqrt(dx**2+dy**2)
                if dmn < Dmn:
                    F = -(1-dmn/Dmn)/Dmn/dmn
                    Fx[nn] += F*dx
                    Fx[mm] -= F*dx
                    Fy[nn] += F*dy
                    Fy[mm] -= F*dy
                    
                    dvx = vx[mm]-vx[nn]
                    dvy = vy[mm]-vy[nn]
                    FD = beta*(dvx*dx+dvy*dy)/dmn
                    #FD = np.absolute(FD) 
                    Fx[nn] += FD*dx/dmn
                    Fx[mm] -= FD*dx/dmn
                    Fy[nn] += FD*dy/dmn
                    Fy[mm] -= FD*dy/dmn 
                                                            
                    Ep += (1/2)*(1-dmn/Dmn)**2  
                    cont += 1
                    p_now += (-F)*(dx**2+dy**2)

    return Fx, Fy, Fup, Fbot, Ep, cont, p_now, cont_up

def MD_VibrBot_DispUp_ConstP_EkCheck_Collision(beta, Nt, mark_upDS, N, x_ini, y_ini, D0, m0, L, Freq_Vibr, Amp_Vibr, F0_up, mark_norm):
    
    dt = D0[0]/40
    Nt = int(Nt)
    #Nt = int(5e7) 
    #Nt = int(1e4) 
    
    nt_rec = np.linspace(0, Nt, int(Nt/5e4)+1)
    Ek_now = np.array(0)
    Ek_up_now = np.array(0)
    Ep_now = np.array(0)
    Ep_up_now = np.array(0)
        
    #nt_rec = np.linspace(0.5*Nt, Nt, 50)
    
    nt_rec = nt_rec.astype(int)
    
    Ep = np.zeros(Nt)
    Ep_up = np.zeros(Nt)
    Ek = np.zeros(Nt)
    Ek_up = np.zeros(Nt)
    y_up0 = y_ini[N]
    y_up = np.zeros(Nt)    
    F_up = np.zeros(Nt)
    cont = np.zeros(Nt)
    cont_up = np.zeros(Nt)
    p = np.zeros(Nt)
    
    y_bot = Amp_Vibr*np.sin(Freq_Vibr*dt*np.arange(Nt)+1.5*np.pi)+Amp_Vibr
    vy_bot = Amp_Vibr*Freq_Vibr*np.cos(Freq_Vibr*dt*np.arange(Nt)+1.5*np.pi)
    
    vx = np.zeros(N+1)
    vy = np.zeros(N+1)    
    
    ax_old = np.zeros(N+1)
    ay_old = np.zeros(N+1)
    
    x = np.array(x_ini)
    y = np.array(y_ini)
    
    #t_start = time.time()
    
    if mark_upDS == 1:
        ind_up = np.zeros(N)
        for ii in np.arange(N):
            d_up = y[N]-y[ii]
            r_now = 0.5*D0[ii]        
            if d_up<r_now:
                ind_up[ii] = 1
            
            
    for nt in np.arange(Nt):
        x = x+vx*dt+ax_old*dt**2/2;  # first step in Verlet integration
        y = y+vy*dt+ay_old*dt**2/2;
        y_up[nt] = y[N]
        
        Fx = np.zeros(N)
        Fy = np.zeros(N) 

        Fx, Fy, Fup_now, Fbot_now, Ep_now, cont_now, p_now, cont_up_now = force_YFixed_collision_ConstP(beta, Fx, Fy, N, x, y, vx, vy, D0[0:N], L[0], y_bot[nt], vy_bot[nt], y[N])

        F_up[nt] = Fup_now+F0_up        
        Ep[nt] = Ep_now+(y_up0-y[N])*F0_up
        Ep_up[nt] = (y_up0-y[N])*F0_up
        cont[nt] = cont_now
        cont_up[nt] = cont_up_now
        p[nt] = p_now
        
        Fx_all = np.append(Fx,0)
        Fy_all = np.append(Fy, F_up[nt])
        
        ax = np.divide(Fx_all, m0)
        ay = np.divide(Fy_all, m0)
        vx = vx+(ax_old+ax)*dt/2;  # second step in Verlet integration
        vy = vy+(ay_old+ay)*dt/2;        
        ax_old = ax;
        ay_old = ay;
        Ek[nt] = sum(0.5*np.multiply(m0,np.multiply(vx, vx)+np.multiply(vy, vy)))
        Ek_up[nt] = 0.5*m0[N]*(vx[N]**2+vy[N]**2)
    
    for ii in np.arange(len(nt_rec)-1):
        Ek_now = np.append(Ek_now, np.mean(Ek[nt_rec[ii]:nt_rec[ii+1]]))
        Ek_up_now = np.append(Ek_up_now, np.mean(Ek_up[nt_rec[ii]:nt_rec[ii+1]]))
        Ep_now = np.append(Ep_now, np.mean(Ep[nt_rec[ii]:nt_rec[ii+1]]))
        Ep_up_now = np.append(Ep_up_now, np.mean(Ep_up[nt_rec[ii]:nt_rec[ii+1]]))

    #t_end = time.time()
    #print ("time=%.3e" %(t_end-t_start))
    CB_ratio = min(cont)/max(cont)
    print ("freq=%f, cont_min/cont_max=%f\n" %(Freq_Vibr, CB_ratio))
    
    y_up = y_up-np.mean(y_up)
    if mark_norm == 1:
        y_up = y_up/np.mean(np.absolute(y_up))
    freq_y, fft_y = FFT_Fup(int(Nt/2), y_up[int(Nt/2):Nt], dt, Freq_Vibr)
    freq_bot, fft_bot = FFT_Fup(int(Nt/2), y_bot[int(Nt/2):Nt], dt, Freq_Vibr)  
            

    return freq_y, fft_y, fft_bot, np.mean(cont), np.mean(cont_up), nt_rec[1:], Ek_now[1:],Ek_up_now[1:],Ep_now[1:],Ep_up_now[1:]

def MD_YFixed_ConstP_Gravity_SD(N, x0, y0, D0, m0, L, F0_up):    
    
    g = 1e-5
    dt = D0[0]/40
    Nt = int(5e6)
    #Nt = int(1e4)
    Ep = np.zeros(Nt)
    F_up = np.zeros(Nt)
    F_bot = np.zeros(Nt)
    F_tot = np.zeros(Nt)
    Fup_now = 0
    y_up = y0[N]
    
    vx = np.zeros(N+1)
    vy = np.zeros(N+1)
    
    x = np.array(x0)
    y = np.array(y0)
    
    t_start = time.time()
    for nt in np.arange(Nt):
        
        Fx = np.zeros(N)
        Fy = np.zeros(N)
        
        Fx, Fy, Fup_now, Fbot_now, Ep_now, cont_now, p_now, cont_up = force_YFixed(Fx, Fy, N, x[0:N], y[0:N], D0[0:N], L[0], 0, y[N])
        Fy -= g*m0[0:N] 
        F_up[nt] = Fup_now+F0_up-g*m0[N]
        F_bot[nt] = Fbot_now
        Ep[nt] = Ep_now+(y_up-y[N])*F0_up+sum(g*np.multiply(m0, y-y0))
        vx = 0.1*np.divide(np.append(Fx,0), m0)
        vy = 0.1*np.divide(np.append(Fy, F_up[nt]), m0)
        x += vx*dt
        y += vy*dt
        F_tot[nt] = sum(np.absolute(Fx)+np.absolute(Fy))
        #print("nt=%d, Fup=%e, Fup_tot=%e\n" % (nt, Fup_now, F_up[nt]))
        #dbg.set_trace()
    t_end = time.time()
    print ("F_tot=%.3e\n" %(F_tot[nt]))
    print ("time=%.3e" %(t_end-t_start))
    
    return x, y, p_now

def MD_VibrBot_DispUp_ConstP_EkCheck_Gravity(Nt, mark_upDS, N, x_ini, y_ini, D0, m0, L, Freq_Vibr, Amp_Vibr, F0_up):
    
    dt = D0[0]/40
    #B = 0.1 # damping coefficient
    g = 1e-5
    Nt = int(Nt)
    #Nt = int(5e7) 
    #Nt = int(1e4) 
    
    nt_rec = np.linspace(0, Nt, int(Nt/5e4)+1)
    Ek_now = np.array(0)
    Ek_up_now = np.array(0)
    Ep_now = np.array(0)
    Ep_up_now = np.array(0)
        
    #nt_rec = np.linspace(0.5*Nt, Nt, 50)
    
    nt_rec = nt_rec.astype(int)
    
    Ep = np.zeros(Nt)
    Ep_up = np.zeros(Nt)
    Ek = np.zeros(Nt)
    Ek_up = np.zeros(Nt)
    y_up0 = y_ini[N]
    y_up = np.zeros(Nt)    
    F_up = np.zeros(Nt)
    cont = np.zeros(Nt)
    cont_up = np.zeros(Nt)
    p = np.zeros(Nt)
    
    y_bot = Amp_Vibr*np.sin(Freq_Vibr*dt*np.arange(Nt)+1.5*np.pi)+Amp_Vibr
    vy_bot = Amp_Vibr*Freq_Vibr*np.cos(Freq_Vibr*dt*np.arange(Nt)+1.5*np.pi)
    
    vx = np.zeros(N+1)
    vy = np.zeros(N+1)    
    
    ax_old = np.zeros(N+1)
    ay_old = np.zeros(N+1)
    
    x = np.array(x_ini)
    y = np.array(y_ini)
    
    #t_start = time.time()
    
    if mark_upDS == 1:
        ind_up = np.zeros(N)
        for ii in np.arange(N):
            d_up = y[N]-y[ii]
            r_now = 0.5*D0[ii]        
            if d_up<r_now:
                ind_up[ii] = 1
            
            
    for nt in np.arange(Nt):
        x = x+vx*dt+ax_old*dt**2/2;  # first step in Verlet integration
        y = y+vy*dt+ay_old*dt**2/2;
        y_up[nt] = y[N]
        
        Fx = np.zeros(N)
        Fy = np.zeros(N) 

        #Fx, Fy, Fup_now, Fbot_now, Ep_now, cont_now, p_now, cont_up_now = force_YFixed(Fx, Fy, N, x[0:N], y[0:N], D0[0:N], L[0], y_bot[nt], y[N])
        beta = 1
        Fx, Fy, Fup_now, Fbot_now, Ep_now, cont_now, p_now, cont_up_now = force_YFixed_collision_ConstP(beta, Fx, Fy, N, x, y, vx, vy, D0[0:N], L[0], y_bot[nt], vy_bot[nt], y[N])
        
        F_up[nt] = Fup_now+F0_up       
        Ep[nt] = Ep_now+(y_up0-y[N])*F0_up+sum(g*np.multiply(m0, y-y_ini))        
        Ep_up[nt] = (y_up0-y[N])*F0_up
        cont[nt] = cont_now
        cont_up[nt] = cont_up_now
        p[nt] = p_now
        
        #Fx_all = np.append(Fx,0)-B*vx
        #Fy_all = np.append(Fy, F_up[nt])-B*vy-g*m0
        Fx_all = np.append(Fx,0)
        Fy_all = np.append(Fy, F_up[nt])-g*m0
        
        ax = np.divide(Fx_all, m0)
        ay = np.divide(Fy_all, m0)
        vx = vx+(ax_old+ax)*dt/2;  # second step in Verlet integration
        vy = vy+(ay_old+ay)*dt/2;        
        ax_old = ax;
        ay_old = ay;
        Ek[nt] = sum(0.5*np.multiply(m0,np.multiply(vx, vx)+np.multiply(vy, vy)))
        Ek_up[nt] = 0.5*m0[N]*(vx[N]**2+vy[N]**2)
    
    for ii in np.arange(len(nt_rec)-1):
        Ek_now = np.append(Ek_now, np.mean(Ek[nt_rec[ii]:nt_rec[ii+1]]))
        Ek_up_now = np.append(Ek_up_now, np.mean(Ek_up[nt_rec[ii]:nt_rec[ii+1]]))
        Ep_now = np.append(Ep_now, np.mean(Ep[nt_rec[ii]:nt_rec[ii+1]]))
        Ep_up_now = np.append(Ep_up_now, np.mean(Ep_up[nt_rec[ii]:nt_rec[ii+1]]))

    #t_end = time.time()
    #print ("time=%.3e" %(t_end-t_start))
    CB_ratio = min(cont)/max(cont)
    print ("freq=%f, cont_min/cont_max=%f\n" %(Freq_Vibr, CB_ratio))
    
    y_up = y_up-np.mean(y_up)
    #y_up = y_up/np.mean(np.absolute(y_up))
    freq_y, fft_y = FFT_Fup(int(Nt/2), y_up[int(Nt/2):Nt], dt, Freq_Vibr)
    freq_bot, fft_bot = FFT_Fup(int(Nt/2), y_bot[int(Nt/2):Nt], dt, Freq_Vibr)  
            

    return freq_y, fft_y, fft_bot, np.mean(cont), np.mean(cont_up), nt_rec[1:], Ek_now[1:],Ek_up_now[1:],Ep_now[1:],Ep_up_now[1:]

def MD_YFixed_ConstV_SP_SD(Nt, N, x0, y0, D0, m0, Lx, Ly):    
    
    dt = D0[0]/40
    Nt = int(Nt)
    Ep = np.zeros(Nt)
    F_tot = np.zeros(Nt)
    
    vx = np.zeros(N)
    vy = np.zeros(N)    
    x = np.array(x0)
    y = np.array(y0)
    
    t_start = time.time()
    for nt in np.arange(Nt):        
        Fx = np.zeros(N)
        Fy = np.zeros(N)        
        Fx, Fy, Fup_now, Fbot_now, Ep_now, cont_now, p_now, cont_up = force_YFixed(Fx, Fy, N, x, y, D0, Lx, 0, Ly)
        Ep[nt] = Ep_now
        vx = 0.1*np.divide(Fx, m0)
        vy = 0.1*np.divide(Fy, m0)
        x += vx*dt
        y += vy*dt
        F_tot[nt] = sum(np.absolute(Fx)+np.absolute(Fy))
    t_end = time.time()
    print ("F_tot=%.3e" %(F_tot[nt]))
    print ("time=%.3e" %(t_end-t_start))

    plt.figure(figsize=(6.4,4.8))
    plt.plot(range(Nt), F_tot[0:Nt], color='blue')
    ax = plt.gca()
    ax.set_yscale('log')
    plt.xlabel("t")
    plt.ylabel("F_total")
    plt.title("Finding the Equilibrium", fontsize='small')
    plt.grid(color='skyblue', linestyle=':', linewidth=0.5)
    plt.tight_layout()
    plt.show()

    return x, y, p_now

def MD_YFixed_ConstV_SP_SD_2(Nt, N, x0, y0, D0, m0, Lx, Ly):    
    
    dt = D0[0]/40
    Nt = int(Nt)
    Ep = np.zeros(Nt)
    F_tot = np.zeros(Nt)
    
    vx = np.zeros(N)
    vy = np.zeros(N)    
    x = np.array(x0)
    y = np.array(y0)
    
    t_start = time.time()
    for nt in np.arange(Nt):        
        Fx = np.zeros(N)
        Fy = np.zeros(N)        
        Fx, Fy, Fup_now, Fbot_now, Ep_now, cont_now, p_now, cont_up = force_YFixed(Fx, Fy, N, x, y, D0, Lx, 0, Ly)
        Ep[nt] = Ep_now
        vx = 0.1*np.divide(Fx, m0)
        vy = 0.1*np.divide(Fy, m0)
        x += vx*dt
        y += vy*dt
        F_tot[nt] = sum(np.absolute(Fx)+np.absolute(Fy))
        # putting a threshold on total force
        if (F_tot[nt]<1e-11):
            break
    t_end = time.time()
    #print ("F_tot=%.3e" %(F_tot[nt]))
    #print ("time=%.3e" %(t_end-t_start))

    return x, y, p_now

#@jit
def force_YFixed_collision_ConstV(beta, Fx, Fy, N, x, y, vx, vy, D, Lx, y_bot, y_up):
    
    Ep = 0
    cont = 0
    p_now = 0
    for nn in np.arange(N):
        d_up = y_up-y[nn]
        d_bot = y[nn]-y_bot
        r_now = 0.5*D[nn]
        
        if d_up<r_now:
            F = -(1-d_up/r_now)/(r_now)
            Fy[nn] += F            
            
            dvy = -vy[nn]
            FD = beta*dvy
            Fy[nn] += FD
            
            Ep += (1/2)*(1-d_up/r_now)**2
            cont += 1
            #dbg.set_trace()
            
        
        if d_bot<r_now:
            F = -(1-d_bot/r_now)/(r_now)
            Fy[nn] -= F
            
            dvy = -vy[nn]
            FD = beta*dvy
            Fy[nn] += FD
            
            Ep += (1/2)*(1-d_bot/r_now)**2
            cont += 1

        for mm in np.arange(nn+1, N):
            dy = y[mm]-y[nn]
            Dmn = 0.5*(D[mm]+D[nn])
            if abs(dy) < Dmn:
                dx = x[mm]-x[nn]
                dx = dx-round(dx/Lx)*Lx
                dmn = np.sqrt(dx**2+dy**2)
                if dmn < Dmn:
                    F = -(1-dmn/Dmn)/Dmn/dmn
                    Fx[nn] += F*dx
                    Fx[mm] -= F*dx
                    Fy[nn] += F*dy
                    Fy[mm] -= F*dy
                    
                    dvx = vx[mm]-vx[nn]
                    dvy = vy[mm]-vy[nn]
                    FD = beta*(dvx*dx+dvy*dy)/dmn
                    #FD = np.absolute(FD) 
                    Fx[nn] += FD*dx/dmn
                    Fx[mm] -= FD*dx/dmn
                    Fy[nn] += FD*dy/dmn
                    Fy[mm] -= FD*dy/dmn 
                                                            
                    Ep += (1/2)*(1-dmn/Dmn)**2  
                    cont += 1
                    p_now += (-F)*(dx**2+dy**2)

    return Fx, Fy, Ep, cont, p_now

def MD_VibrSP_ConstV_Collision(beta, Nt, N, x_ini, y_ini, D0, m0, L, Freq_Vibr, Amp_Vibr, ind_in, ind_out, mark_vibrY):
    
    dt = D0[0]/40
    Nt = int(Nt)
    
    nt_rec = np.linspace(0, Nt, int(Nt/5e4)+1)    
    
    nt_rec = nt_rec.astype(int)
    
    Ep = np.zeros(Nt)
    Ek = np.zeros(Nt)   
    cont = np.zeros(Nt)
    p = np.zeros(Nt)
    
    x_out = np.zeros(Nt)
    y_out = np.zeros(Nt)
    
    if mark_vibrY == 0:
        x_in = Amp_Vibr*np.sin(Freq_Vibr*dt*np.arange(Nt))+x_ini[ind_in]
        vx_in = Amp_Vibr*Freq_Vibr*np.cos(Freq_Vibr*dt*np.arange(Nt))
    elif mark_vibrY == 1:
        y_in = Amp_Vibr*np.sin(Freq_Vibr*dt*np.arange(Nt))+y_ini[ind_in]
        vy_in = Amp_Vibr*Freq_Vibr*np.cos(Freq_Vibr*dt*np.arange(Nt))
    #y_bot = np.zeros(Nt)
    
    vx = np.zeros(N)
    vy = np.zeros(N)    
    
    ax_old = np.zeros(N)
    ay_old = np.zeros(N)
    
    x = np.array(x_ini)
    y = np.array(y_ini)
    
    #t_start = time.time()

    for nt in np.arange(Nt):
        x = x+vx*dt+ax_old*dt**2/2;  # first step in Verlet integration
        y = y+vy*dt+ay_old*dt**2/2;
        if mark_vibrY == 0:
            x[ind_in] = x_in[nt]
            y[ind_in] = y_ini[ind_in]
            vx[ind_in] = vx_in[nt]
            vy[ind_in] = 0
        elif mark_vibrY == 1:
            x[ind_in] = x_ini[ind_in]
            y[ind_in] = y_in[nt]
            vx[ind_in] = 0
            vy[ind_in] = vy_in[nt]
        
        Fx = np.zeros(N)
        Fy = np.zeros(N) 

        Fx, Fy, Ep_now, cont_now, p_now = force_YFixed_collision_ConstV(beta, Fx, Fy, N, x, y, vx, vy, D0, L[0], 0, L[1])
    
        Ep[nt] = Ep_now
        cont[nt] = cont_now
        p[nt] = p_now
        
        Fx_all = Fx
        Fy_all = Fy

        x_out[nt] = x[ind_out]
        y_out[nt] = y[ind_out]
        
        ax = np.divide(Fx_all, m0)
        ay = np.divide(Fy_all, m0)
        vx = vx+(ax_old+ax)*dt/2;  # second step in Verlet integration
        vy = vy+(ay_old+ay)*dt/2;        
        ax_old = ax;
        ay_old = ay;
        Ek[nt] = sum(0.5*np.multiply(m0,np.multiply(vx, vx)+np.multiply(vy, vy)))
    
    Ek_now = np.array(0)
    Ep_now = np.array(0)
    cont_now = np.array(0)
    for ii in np.arange(len(nt_rec)-1):
        Ek_now = np.append(Ek_now, np.mean(Ek[nt_rec[ii]:nt_rec[ii+1]]))
        Ep_now = np.append(Ep_now, np.mean(Ep[nt_rec[ii]:nt_rec[ii+1]]))
        cont_now = np.append(cont_now, np.mean(cont[nt_rec[ii]:nt_rec[ii+1]]))
    cont_now[0] = cont_now[1]

    #t_end = time.time()
    #print ("time=%.3e" %(t_end-t_start))
    CB_ratio = min(cont)/max(cont)
    print ("freq=%f, cont_min/cont_max=%f\n" %(Freq_Vibr, CB_ratio))
    
    if mark_vibrY == 0:
        freq_fft, fft_in = FFT_Fup(int(Nt/2), x_in[int(Nt/2):Nt], dt, Freq_Vibr)        
    elif mark_vibrY == 1:
        freq_fft, fft_in = FFT_Fup(int(Nt/2), y_in[int(Nt/2):Nt], dt, Freq_Vibr)
    freq_fft, fft_x_out = FFT_Fup(int(Nt/2), x_out[int(Nt/2):Nt], dt, Freq_Vibr)
    freq_fft, fft_y_out = FFT_Fup(int(Nt/2), y_out[int(Nt/2):Nt], dt, Freq_Vibr)
       
    return freq_fft, fft_in, fft_x_out, fft_y_out, np.mean(cont), nt_rec, Ek_now, Ep_now, cont_now

def MD_VibrSP_ConstV(B, Nt, N, x_ini, y_ini, D0, m0, L, Freq_Vibr, Amp_Vibr, ind_in, ind_out, mark_vibrY, mark_resonator):
    
    dt = D0[0]/40
    Nt = int(Nt)
    
    nt_rec = np.linspace(0, Nt, int(Nt/5e4)+1)    
    
    nt_rec = nt_rec.astype(int)
    
    Ep = np.zeros(Nt)
    Ek = np.zeros(Nt)   
    cont = np.zeros(Nt)
    p = np.zeros(Nt)
    
    x_out = np.zeros(Nt)
    y_out = np.zeros(Nt)
    
    if mark_vibrY == 0:
        x_in = Amp_Vibr*np.sin(Freq_Vibr*dt*np.arange(Nt))+x_ini[ind_in]
    elif mark_vibrY == 1:
        y_in = Amp_Vibr*np.sin(Freq_Vibr*dt*np.arange(Nt))+y_ini[ind_in]

    #y_bot = np.zeros(Nt)
    
    vx = np.zeros(N)
    vy = np.zeros(N)    
    
    ax_old = np.zeros(N)
    ay_old = np.zeros(N)
    
    x = np.array(x_ini)
    y = np.array(y_ini)
    
    #t_start = time.time()

    for nt in np.arange(Nt):
        x = x+vx*dt+ax_old*dt**2/2;  # first step in Verlet integration
        y = y+vy*dt+ay_old*dt**2/2;
        
        if mark_vibrY == 0:
            x[ind_in] = x_in[nt]
            y[ind_in] = y_ini[ind_in]
        elif mark_vibrY == 1:
            x[ind_in] = x_ini[ind_in]
            y[ind_in] = y_in[nt]        
        
        Fx = np.zeros(N)
        Fy = np.zeros(N) 

        Fx, Fy, Fup_now, Fbot_now, Ep_now, cont_now, p_now, cont_up = force_YFixed(Fx, Fy, N, x, y, D0, L[0], 0, L[1])
        
        Ep[nt] = Ep_now
        cont[nt] = cont_now
        p[nt] = p_now
        
        Fx_all = Fx-B*vx
        Fy_all = Fy-B*vy

        x_out[nt] = x[ind_out]
        y_out[nt] = y[ind_out]
        
        ax = np.divide(Fx_all, m0)
        ay = np.divide(Fy_all, m0)
        vx = vx+(ax_old+ax)*dt/2;  # second step in Verlet integration
        vy = vy+(ay_old+ay)*dt/2;        
        ax_old = ax;
        ay_old = ay;
        Ek[nt] = sum(0.5*np.multiply(m0,np.multiply(vx, vx)+np.multiply(vy, vy)))
    
    
    
    Ek_now = np.array(0)
    Ep_now = np.array(0)
    cont_now = np.array(0)
    for ii in np.arange(len(nt_rec)-1):
        Ek_now = np.append(Ek_now, np.mean(Ek[nt_rec[ii]:nt_rec[ii+1]]))
        Ep_now = np.append(Ep_now, np.mean(Ep[nt_rec[ii]:nt_rec[ii+1]]))
        cont_now = np.append(cont_now, np.mean(cont[nt_rec[ii]:nt_rec[ii+1]]))
    cont_now[0] = cont_now[1]

    #t_end = time.time()
    #print ("time=%.3e" %(t_end-t_start))
    CB_ratio = min(cont)/max(cont)
    print ("freq=%f, cont_min/cont_max=%f\n" %(Freq_Vibr, CB_ratio))
    
    if mark_vibrY == 0:
        freq_fft, fft_in = FFT_Fup(int(Nt/2), x_in[int(Nt/2):Nt], dt, Freq_Vibr)        
    elif mark_vibrY == 1:
        freq_fft, fft_in = FFT_Fup(int(Nt/2), y_in[int(Nt/2):Nt], dt, Freq_Vibr)
    
    if mark_resonator == 0:
        freq_fft, fft_x_out = FFT_Fup(int(Nt/2), x_out[int(Nt/2):Nt], dt, Freq_Vibr)
        freq_fft, fft_y_out = FFT_Fup(int(Nt/2), y_out[int(Nt/2):Nt], dt, Freq_Vibr)
    elif mark_resonator == 1:
        freq_fft, fft_x_out = Output_resonator_1D(Nt, x_out[0:Nt], x_ini[ind_out], m0[ind_out], Freq_Vibr, dt)
        freq_fft, fft_y_out = Output_resonator_1D(Nt, y_out[0:Nt], y_ini[ind_out], m0[ind_out], Freq_Vibr, dt)
       
    if Nt == 5e5:
        print(x[ind_out], y[ind_out])
        print(fft_x_out[100], fft_y_out[100])
        print(fft_in[100])
        
    return freq_fft, fft_in, fft_x_out, fft_y_out, np.mean(cont), nt_rec, Ek_now, Ep_now, cont_now


def MD_Periodic_ConstV_SP_SD(Nt, N, x0, y0, D0, m0, Lx, Ly):    
    
    dt = D0[0]/40
    Nt = int(Nt)
    Ep = np.zeros(Nt)
    F_tot = np.zeros(Nt)
    
    vx = np.zeros(N)
    vy = np.zeros(N)    
    x = np.array(x0)
    y = np.array(y0)
    
    t_start = time.time()
    for nt in np.arange(Nt):        
        Fx = np.zeros(N)
        Fy = np.zeros(N)        
        Fx, Fy, Ep_now, cont_now, p_now = force_Regular(Fx, Fy, N, x, y, D0, Lx, Ly)
        Ep[nt] = Ep_now
        vx = 0.1*np.divide(Fx, m0)
        vy = 0.1*np.divide(Fy, m0)
        x += vx*dt
        y += vy*dt
        F_tot[nt] = sum(np.absolute(Fx)+np.absolute(Fy))

    t_end = time.time()
    print ("F_tot=%.3e\n" %(F_tot[nt]))
    print ("nt=%e, time=%.3e" %(nt, t_end-t_start))
        
    return x, y, p_now

def MD_Periodic_equi_Ekcheck(Nt, N, x0, y0, D0, m0, L, T_set, V_em, n_em):
    
    dt = min(D0)/40
    Nt = int(Nt)
    Ep = np.zeros(Nt)
    Ek = np.zeros(Nt)
    cont = np.zeros(Nt)
    vx_rec = np.zeros([Nt, N])
    vy_rec = np.zeros([Nt, N])
    
    nt_rec = np.linspace(0, Nt, int(Nt/1e3)+1)
    #nt_rec = np.linspace(0, Nt, int(Nt/1e2)+1)
    nt_rec = nt_rec.astype(int)

    Ek_now = np.array(0)
    Ep_now = np.array(0)
 

    vx = np.zeros(N)
    vy = np.zeros(N)
    for ii in np.arange(n_em):
        ind1 = 2*np.arange(N)
        ind2 = ind1+1
        vx += V_em[ind1, ii]
        vy += V_em[ind2, ii]
        
    T_rd = sum(0.5*np.multiply(m0,np.multiply(vx, vx)+np.multiply(vy, vy)))
    vx = vx*np.sqrt(N*T_set/T_rd)
    vy = vy*np.sqrt(N*T_set/T_rd)
    
    ax_old = np.zeros(N)
    ay_old = np.zeros(N)
    
    x = np.array(x0)
    y = np.array(y0)
    
    #t_start = time.time()
    for nt in np.arange(Nt):
        x = x+vx*dt+ax_old*dt**2/2;  # first step in Verlet integration
        y = y+vy*dt+ay_old*dt**2/2;

        Fx = np.zeros(N)
        Fy = np.zeros(N)                        
        Fx, Fy, Ep_now, cont_now, p_now = force_Regular(Fx, Fy, N, x, y, D0, L[0], L[1])
        Ep[nt] = Ep_now
        cont[nt] = cont_now
        
        ax = np.divide(Fx, m0);
        ay = np.divide(Fy, m0);
        vx = vx+(ax_old+ax)*dt/2;  # second step in Verlet integration
        vy = vy+(ay_old+ay)*dt/2;        
        ax_old = ax;
        ay_old = ay;
        Ek[nt] = sum(0.5*np.multiply(m0,np.multiply(vx, vx)+np.multiply(vy, vy)))
        vx_rec[nt] = vx
        vy_rec[nt] = vy
    
    for ii in np.arange(len(nt_rec)-1):
        Ek_now = np.append(Ek_now, np.mean(Ek[nt_rec[ii]:nt_rec[ii+1]]))
        Ep_now = np.append(Ep_now, np.mean(Ep[nt_rec[ii]:nt_rec[ii+1]]))
    
    #t_end = time.time()
    #print ("time=%.3e" %(t_end-t_start))
    CB_ratio = min(cont)/max(cont)
    
    print ("cont_min/cont_max=%f\n" %(CB_ratio))
    
    freq_now, fft_now = FFT_vCorr(int(Nt/2), N, vx_rec[int(Nt/2):Nt], vy_rec[int(Nt/2):Nt], dt)        
    return freq_now, fft_now, np.mean(cont), nt_rec, Ek_now, Ep_now

#@jit
def force_Xfixed(Fx, Fy, N, x, y, D, x_l, x_r, Ly, ind_wall):
    
    F_l = 0
    F_r = 0
    Ep = 0
    cont = 0
    p_now = 0
    for nn in np.arange(N):
        d_l = x[nn]-x_l
        d_r = x_r-x[nn]
        r_now = 0.5*D[nn]
        
        if (ind_wall[nn]==0) and (d_r<r_now):
            F = -(1-d_r/r_now)/(r_now)
            F_r -= F
            Fx[nn] += F
            Ep += (1/2)*(1-d_r/r_now)**2
            cont += 1
            #dbg.set_trace()
            
        
        if (ind_wall[nn]==0) and (d_l<r_now):
            F = -(1-d_l/r_now)/(r_now)
            F_l += F
            Fx[nn] -= F
            Ep += (1/2)*(1-d_l/r_now)**2
            cont += 1

        for mm in np.arange(nn+1, N):
            dx = x[mm]-x[nn]
            Dmn = 0.5*(D[mm]+D[nn])
            if abs(dx) < Dmn:
                dy = y[mm]-y[nn]
                dy = dy-round(dy/Ly)*Ly
                dmn = np.sqrt(dx**2+dy**2)
                if dmn < Dmn:
                    F = -(1-dmn/Dmn)/Dmn/dmn
                    Fx[nn] += F*dx
                    Fx[mm] -= F*dx
                    Fy[nn] += F*dy
                    Fy[mm] -= F*dy
                    Ep += (1/2)*(1-dmn/Dmn)**2  
                    cont += 1
                    p_now += (-F)*(dx**2+dy**2)
    return Fx, Fy, F_l, F_r, Ep, cont, p_now

def MD_Xfixed_SD(Nt, N, x0, y0, D0, m0, Lx, Ly, ind_wall):    
    
    wall = np.where(ind_wall>0)
    dt = D0[0]/40
    Nt = int(Nt)
    #Nt = int(5e6)
    #Nt = int(5e2)
    Ep = np.zeros(Nt)
    F_l = np.zeros(Nt)
    F_r = np.zeros(Nt)
    F_tot = np.zeros(Nt)
    
    vx = np.zeros(N)
    vy = np.zeros(N)
    
    x = np.array(x0)
    y = np.array(y0)
    
    t_start = time.time()
    for nt in np.arange(Nt):
        
        Fx = np.zeros(N)
        Fy = np.zeros(N)
        
        Fx, Fy, Fl_now, Fr_now, Ep_now, cont_now, p_now = force_Xfixed(Fx, Fy, N, x, y, D0, 0, Lx, Ly, ind_wall)
        F_l[nt] = Fl_now
        F_r[nt] = Fr_now
        Ep[nt] = Ep_now
        
        Fx[wall] = 0
        Fy[wall] = 0
        vx = 0.1*np.divide(Fx, m0)
        vy = 0.1*np.divide(Fy, m0)    
        
        x += vx*dt
        y += vy*dt
        F_tot[nt] = sum(np.absolute(Fx)+np.absolute(Fy))
        #print("nt=%d, Fup=%e, Fup_tot=%e\n" % (nt, Fup_now, F_up[nt]))
        #dbg.set_trace()
    t_end = time.time()
    print ("F_tot=%.3e" %(F_tot[nt]))
    #print ("Ep_tot=%.3e\n" %(Ep[nt]))
    print ("time=%.3e" %(t_end-t_start))
    

    return x, y, p_now


def MD_VibrWall_DiffP_Xfixed(Nt, N, x_ini, y_ini,D0, m0, Lx, Ly, Freq_Vibr, Amp_Vibr, ind_wall, B): 
    
    dt = D0[0]/40
    # B damping coefficient
    Nt = int(Nt)
    #Nt = int(5e7) 
    #Nt = int(1e4) 
    
    nt_rec = np.linspace(0, Nt, int(Nt/5e4)+1)
    Ek_now = np.array(0)
    Ep_now = np.array(0)
        
    #nt_rec = np.linspace(0.5*Nt, Nt, 50)
    
    nt_rec = nt_rec.astype(int)
    
    Ep = np.zeros(Nt)
    Ek = np.zeros(Nt)
    x_l = np.zeros(Nt)
    F_r = np.zeros(Nt)    
    cont = np.zeros(Nt)
    p = np.zeros(Nt)
    
    x_l = Amp_Vibr*np.sin(Freq_Vibr*dt*np.arange(Nt)+1.5*np.pi)+Amp_Vibr
    
    wall_l = np.where(ind_wall==1)
    wall_r = np.where(ind_wall==2)
    wall = np.where(ind_wall>0)
    
    vx = np.zeros(N)
    vy = np.zeros(N)    
    
    ax_old = np.zeros(N)
    ay_old = np.zeros(N)
    
    x = np.array(x_ini)
    y = np.array(y_ini)
    
    for nt in np.arange(Nt):
        x = x+vx*dt+ax_old*dt**2/2;  # first step in Verlet integration
        y = y+vy*dt+ay_old*dt**2/2;
        
        Fx = np.zeros(N)
        Fy = np.zeros(N) 
        
        x[wall_l] = x_l[nt]
        Fx, Fy, Fl_now, Fr_now, Ep_now, cont_now, p_now = force_Xfixed(Fx, Fy, N, x, y, D0, x_l[nt], Lx, Ly, ind_wall)

        F_r[nt] = Fr_now+sum(Fx[wall_r])        
        Ep[nt] = Ep_now
        cont[nt] = cont_now
        p[nt] = p_now
        
        Fx_all = Fx-B*vx
        Fy_all = Fy-B*vy
        
                
        ax = np.divide(Fx_all, m0)
        ay = np.divide(Fy_all, m0)
        vx = vx+(ax_old+ax)*dt/2;  # second step in Verlet integration
        vy = vy+(ay_old+ay)*dt/2;        
        
        ax[wall] = 0
        ay[wall] = 0
        vx[wall] = 0
        vy[wall] = 0
        
        ax_old = ax;
        ay_old = ay;
        Ek[nt] = sum(0.5*np.multiply(m0,np.multiply(vx, vx)+np.multiply(vy, vy)))

    
    #for ii in np.arange(len(nt_rec)-1):
    #    Ek_now = np.append(Ek_now, np.mean(Ek[nt_rec[ii]:nt_rec[ii+1]]))
    #    Ep_now = np.append(Ep_now, np.mean(Ep[nt_rec[ii]:nt_rec[ii+1]]))

    #CB_ratio = min(cont)/max(cont)
    #print ("freq=%f, cont_min/cont_max=%f\n" %(Freq_Vibr, CB_ratio))
    

    freq_fft, fft_receive = FFT_Fup(int(Nt/2), F_r[int(Nt/2):Nt], dt, Freq_Vibr)
    freq_fft, fft_drive = FFT_Fup(int(Nt/2), x_l[int(Nt/2):Nt], dt, Freq_Vibr)  
            
    return freq_fft, fft_receive, fft_drive, cont_now, nt_rec, Ek_now, Ep_now


#@jit
def force_XFixed_collision_VibrLx(beta, Fx, Fy, N, x, y, vx, vy, D, x_l, Lx, Ly, vx_l, ind_wall):
    
    Fr = 0
    Fl = 0
    Ep = 0
    cont = 0
    p_now = 0
    #betta = 1
    for nn in np.arange(N):
        
        if ind_wall[nn] == 0:
            d_r = Lx-x[nn]
            d_l = x[nn]-x_l
            r_now = 0.5*D[nn]
            
            if d_r<r_now:
                F = -(1-d_r/r_now)/(r_now)
                Fr -= F
                Fx[nn] += F                            
                dvx = -vx[nn]
                FD = beta*dvx
                Fx[nn] += FD
                Fr -= FD                
                Ep += (1/2)*(1-d_r/r_now)**2
                cont += 1
                #dbg.set_trace()
                
            
            if d_l<r_now:
                F = -(1-d_l/r_now)/(r_now)
                Fl += F
                Fx[nn] -= F                
                dvx = vx_l-vx[nn]
                FD = beta*dvx
                Fx[nn] += FD                
                Ep += (1/2)*(1-d_l/r_now)**2
                cont += 1

        for mm in np.arange(nn+1, N):
            dx = x[mm]-x[nn]
            Dmn = 0.5*(D[mm]+D[nn])
            if abs(dx) < Dmn:
                dy = y[mm]-y[nn]
                dy = dy-round(dy/Ly)*Ly
                dmn = np.sqrt(dx**2+dy**2)
                if dmn < Dmn:
                    F = -(1-dmn/Dmn)/Dmn/dmn
                    Fx[nn] += F*dx
                    Fx[mm] -= F*dx
                    Fy[nn] += F*dy
                    Fy[mm] -= F*dy
                    
                    dvx = vx[mm]-vx[nn]
                    dvy = vy[mm]-vy[nn]
                    FD = beta*(dvx*dx+dvy*dy)/dmn
                    #FD = np.absolute(FD) 
                    Fx[nn] += FD*dx/dmn
                    Fx[mm] -= FD*dx/dmn
                    Fy[nn] += FD*dy/dmn
                    Fy[mm] -= FD*dy/dmn 
                                                            
                    Ep += (1/2)*(1-dmn/Dmn)**2  
                    cont += 1
                    p_now += (-F)*(dx**2+dy**2)

    return Fx, Fy, Fl, Fr, Ep, cont, p_now


def MD_VibrWall_DiffP_Xfixed_Collision(Nt, N, x_ini, y_ini,D0, m0, Lx, Ly, Freq_Vibr, Amp_Vibr, ind_wall, beta): 
    
    dt = D0[0]/40
    # B damping coefficient
    Nt = int(Nt)
    #Nt = int(5e7) 
    #Nt = int(1e4) 
    
    nt_rec = np.linspace(0, Nt, int(Nt/5e4)+1)
    Ek_now = np.array(0)
    Ep_now = np.array(0)
        
    #nt_rec = np.linspace(0.5*Nt, Nt, 50)
    
    nt_rec = nt_rec.astype(int)
    
    Ep = np.zeros(Nt)
    Ek = np.zeros(Nt)
    x_l = np.zeros(Nt)
    F_r = np.zeros(Nt)    
    cont = np.zeros(Nt)
    p = np.zeros(Nt)
    
    x_l = Amp_Vibr*np.sin(Freq_Vibr*dt*np.arange(Nt)+1.5*np.pi)+Amp_Vibr
    vx_l = Amp_Vibr*Freq_Vibr*np.cos(Freq_Vibr*dt*np.arange(Nt)+1.5*np.pi)
    
    
    wall_l = np.where(ind_wall==1)
    wall_r = np.where(ind_wall==2)
    wall = np.where(ind_wall>0)
    
    vx = np.zeros(N)
    vy = np.zeros(N)    
    
    ax_old = np.zeros(N)
    ay_old = np.zeros(N)
    
    x = np.array(x_ini)
    y = np.array(y_ini)
    
    for nt in np.arange(Nt):
        x = x+vx*dt+ax_old*dt**2/2;  # first step in Verlet integration
        y = y+vy*dt+ay_old*dt**2/2;
        
        Fx = np.zeros(N)
        Fy = np.zeros(N) 
        
        x[wall_l] = x_l[nt]
        vx[wall_l] = vx_l[nt]
        Fx, Fy, Fl_now, Fr_now, Ep_now, cont_now, p_now = force_XFixed_collision_VibrLx(beta, Fx, Fy, N, x, y, vx, vy, D0, x_l[nt], Lx, Ly, vx_l[nt], ind_wall)

        F_r[nt] = Fr_now+sum(Fx[wall_r])        
        Ep[nt] = Ep_now
        cont[nt] = cont_now
        p[nt] = p_now
        
        Fx_all = Fx
        Fy_all = Fy
                        
        ax = np.divide(Fx_all, m0)
        ay = np.divide(Fy_all, m0)
        vx = vx+(ax_old+ax)*dt/2;  # second step in Verlet integration
        vy = vy+(ay_old+ay)*dt/2;        
        
        ax[wall] = 0
        ay[wall] = 0
        vx[wall] = 0
        vy[wall] = 0
        
        ax_old = ax;
        ay_old = ay;
        Ek[nt] = sum(0.5*np.multiply(m0,np.multiply(vx, vx)+np.multiply(vy, vy)))
    
    for ii in np.arange(len(nt_rec)-1):
        Ek_now = np.append(Ek_now, np.mean(Ek[nt_rec[ii]:nt_rec[ii+1]]))
        Ep_now = np.append(Ep_now, np.mean(Ep[nt_rec[ii]:nt_rec[ii+1]]))

    CB_ratio = min(cont)/max(cont)
    print ("freq=%f, cont_min/cont_max=%f\n" %(Freq_Vibr, CB_ratio))
    

    freq_fft, fft_receive = FFT_Fup(int(Nt/2), F_r[int(Nt/2):Nt], dt, Freq_Vibr)
    freq_fft, fft_drive = FFT_Fup(int(Nt/2), x_l[int(Nt/2):Nt], dt, Freq_Vibr)  
            
    return freq_fft, fft_receive, fft_drive, cont_now, nt_rec, Ek_now, Ep_now

def MD_VibrWall_LySignal_Collision(Nt, N, x_ini, y_ini,D0, m0, Lx0, Ly0, Freq_Vibr, Amp_Vibr, ind_wall, beta, dLy_scheme, num_gap): 
    
    dt = D0[0]/40
    # B damping coefficient
    Nt = int(Nt)
    #Nt = int(5e7) 
    #Nt = int(1e4)
    dLy_max = 0.1
    nt_transition = int(Nt/num_gap/20)
    dLy_inc = np.linspace(0, dLy_max, nt_transition)
    dLy_dec = np.linspace(dLy_max, 0, nt_transition)
    
    if dLy_scheme == 0:
        dLy_all = np.zeros(Nt)
    elif dLy_scheme == 1:
        dLy_all = np.ones(Nt)*dLy_max
        dLy_all[0:nt_transition] = dLy_inc        
    elif dLy_scheme == 2:
        dLy_all = np.zeros(Nt)
        nt_Ly = np.linspace(0, Nt, num_gap+1)
        nt_Ly = nt_Ly.astype(int)
        for ii in np.arange(1, num_gap):
            nt1 = nt_Ly[ii]-int(nt_transition/2)
            nt2 = nt_Ly[ii]+int(nt_transition/2)
            if ii%2 == 1:
                dLy_all[nt_Ly[ii]:nt_Ly[ii+1]] = dLy_max
                dLy_all[nt1:nt2] = dLy_inc
            else:
                dLy_all[nt1:nt2] = dLy_dec
    
    nt_rec = np.linspace(0, Nt, int(Nt/5e4*num_gap/5)+1)
    Ek_now = np.array(0)
    Ep_now = np.array(0)
        
    #nt_rec = np.linspace(0.5*Nt, Nt, 50)
    
    nt_rec = nt_rec.astype(int)
    
    Ep = np.zeros(Nt)
    Ek = np.zeros(Nt)
    x_l = np.zeros(Nt)
    F_r = np.zeros(Nt)    
    cont = np.zeros(Nt)
    p = np.zeros(Nt)
    
    x_l = Amp_Vibr*np.sin(Freq_Vibr*dt*np.arange(Nt)+1.5*np.pi)+Amp_Vibr
    vx_l = Amp_Vibr*Freq_Vibr*np.cos(Freq_Vibr*dt*np.arange(Nt)+1.5*np.pi)
    
    
    wall_l = np.where(ind_wall==1)
    wall_r = np.where(ind_wall==2)
    wall = np.where(ind_wall>0)
    
    vx = np.zeros(N)
    vy = np.zeros(N)    
    
    ax_old = np.zeros(N)
    ay_old = np.zeros(N)
    
    x = np.array(x_ini)
    y = np.array(y_ini)
    
    for nt in np.arange(Nt):
        Ly = Ly0+dLy_all[nt]
        x = x+vx*dt+ax_old*dt**2/2;  # first step in Verlet integration
        y = y+vy*dt+ay_old*dt**2/2;
        
        Fx = np.zeros(N)
        Fy = np.zeros(N) 
        
        x[wall_l] = x_l[nt]
        vx[wall_l] = vx_l[nt]
        Fx, Fy, Fl_now, Fr_now, Ep_now, cont_now, p_now = force_XFixed_collision_VibrLx(beta, Fx, Fy, N, x, y, vx, vy, D0, x_l[nt], Lx0, Ly, vx_l[nt], ind_wall)

        F_r[nt] = Fr_now+sum(Fx[wall_r])        
        Ep[nt] = Ep_now
        cont[nt] = cont_now
        p[nt] = p_now
        
        Fx_all = Fx
        Fy_all = Fy
                        
        ax = np.divide(Fx_all, m0)
        ay = np.divide(Fy_all, m0)
        vx = vx+(ax_old+ax)*dt/2;  # second step in Verlet integration
        vy = vy+(ay_old+ay)*dt/2;        
        
        ax[wall] = 0
        ay[wall] = 0
        vx[wall] = 0
        vy[wall] = 0
        
        ax_old = ax;
        ay_old = ay;
        Ek[nt] = sum(0.5*np.multiply(m0,np.multiply(vx, vx)+np.multiply(vy, vy)))
    
    for ii in np.arange(len(nt_rec)-1):
        Ek_now = np.append(Ek_now, np.mean(Ek[nt_rec[ii]:nt_rec[ii+1]]))
        Ep_now = np.append(Ep_now, np.mean(Ep[nt_rec[ii]:nt_rec[ii+1]]))

    CB_ratio = min(cont)/max(cont)
    print ("freq=%f, cont_min/cont_max=%f\n" %(Freq_Vibr, CB_ratio))
    
    nt_dLy = np.arange(0, Nt, 100)
    return nt_dLy, dLy_all[nt_dLy], F_r, nt_rec, Ek_now, Ep_now

def MD_VibrWall_LySignal(Nt, N, x_ini, y_ini,D0, m0, Lx0, Ly0, Freq_Vibr, Amp_Vibr, ind_wall, B, dLy_scheme, num_gap): 
    
    dt = D0[0]/40
    # B damping coefficient
    Nt = int(Nt)
    #Nt = int(5e7) 
    #Nt = int(1e4)
    dLy_max = 0.1
    nt_transition = int(Nt/num_gap/20)
    dLy_inc = np.linspace(0, dLy_max, nt_transition)
    dLy_dec = np.linspace(dLy_max, 0, nt_transition)
    
    if dLy_scheme == 0:
        dLy_all = np.zeros(Nt)
    elif dLy_scheme == 1:
        dLy_all = np.ones(Nt)*dLy_max
        dLy_all[0:nt_transition] = dLy_inc        
    elif dLy_scheme == 2:
        dLy_all = np.zeros(Nt)
        nt_Ly = np.linspace(0, Nt, num_gap+1)
        nt_Ly = nt_Ly.astype(int)
        for ii in np.arange(1, num_gap):
            nt1 = nt_Ly[ii]-int(nt_transition/2)
            nt2 = nt_Ly[ii]+int(nt_transition/2)
            if ii%2 == 1:
                dLy_all[nt_Ly[ii]:nt_Ly[ii+1]] = dLy_max
                dLy_all[nt1:nt2] = dLy_inc
            else:
                dLy_all[nt1:nt2] = dLy_dec
    
    nt_rec = np.linspace(0, Nt, int(Nt/5e4*num_gap/5)+1)
    Ek_now = np.array(0)
    Ep_now = np.array(0)
        
    #nt_rec = np.linspace(0.5*Nt, Nt, 50)
    
    nt_rec = nt_rec.astype(int)
    
    Ep = np.zeros(Nt)
    Ek = np.zeros(Nt)
    x_l = np.zeros(Nt)
    F_r = np.zeros(Nt)    
    cont = np.zeros(Nt)
    p = np.zeros(Nt)
    
    x_l = Amp_Vibr*np.sin(Freq_Vibr*dt*np.arange(Nt)+1.5*np.pi)+Amp_Vibr
    vx_l = Amp_Vibr*Freq_Vibr*np.cos(Freq_Vibr*dt*np.arange(Nt)+1.5*np.pi)
    
    
    wall_l = np.where(ind_wall==1)
    wall_r = np.where(ind_wall==2)
    wall = np.where(ind_wall>0)
    
    vx = np.zeros(N)
    vy = np.zeros(N)    
    
    ax_old = np.zeros(N)
    ay_old = np.zeros(N)
    
    x = np.array(x_ini)
    y = np.array(y_ini)
    
    for nt in np.arange(Nt):
        Ly = Ly0+dLy_all[nt]
        x = x+vx*dt+ax_old*dt**2/2;  # first step in Verlet integration
        y = y+vy*dt+ay_old*dt**2/2;
        
        Fx = np.zeros(N)
        Fy = np.zeros(N) 
        
        x[wall_l] = x_l[nt]
        vx[wall_l] = vx_l[nt]
        Fx, Fy, Fl_now, Fr_now, Ep_now, cont_now, p_now = force_Xfixed(Fx, Fy, N, x, y, D0, x_l[nt], Lx0, Ly, ind_wall)

        F_r[nt] = Fr_now+sum(Fx[wall_r])        
        Ep[nt] = Ep_now
        cont[nt] = cont_now
        p[nt] = p_now
        
        Fx_all = Fx-B*vx
        Fy_all = Fy-B*vy
                        
        ax = np.divide(Fx_all, m0)
        ay = np.divide(Fy_all, m0)
        vx = vx+(ax_old+ax)*dt/2;  # second step in Verlet integration
        vy = vy+(ay_old+ay)*dt/2;        
        
        ax[wall] = 0
        ay[wall] = 0
        vx[wall] = 0
        vy[wall] = 0
        
        ax_old = ax;
        ay_old = ay;
        Ek[nt] = sum(0.5*np.multiply(m0,np.multiply(vx, vx)+np.multiply(vy, vy)))
    
    for ii in np.arange(len(nt_rec)-1):
        Ek_now = np.append(Ek_now, np.mean(Ek[nt_rec[ii]:nt_rec[ii+1]]))
        Ep_now = np.append(Ep_now, np.mean(Ep[nt_rec[ii]:nt_rec[ii+1]]))

    CB_ratio = min(cont)/max(cont)
    print ("freq=%f, cont_min/cont_max=%f\n" %(Freq_Vibr, CB_ratio))
    
    nt_dLy = np.arange(0, Nt, 100)
    return nt_dLy, dLy_all[nt_dLy], F_r, nt_rec, Ek_now, Ep_now


def MD_VibrBot_FSignal_Collision(beta, Nt, N, x_ini, y_ini, D0, m0, Lx, Freq_Vibr, Amp_Vibr, F_scheme, num_gap):    
    dt = D0[0]/40
    Nt = int(Nt)
    #Nt = int(5e7) 
    #Nt = int(1e4) 
    
    F_max = 0.01
    F_min = 1e-8
    nt_transition = int(Nt/num_gap/20)
    F_inc = np.linspace(F_min, F_max, nt_transition)
    F_dec = np.linspace(F_max, F_min, nt_transition)
    
    if F_scheme == 1:
        F_all = np.ones(Nt)*F_max
    elif F_scheme == 0:
        F_all = np.ones(Nt)*F_min
        F_all[0:nt_transition] = F_dec        
    elif F_scheme == 2:
        F_all = np.ones(Nt)*F_max
        nt_F = np.linspace(0, Nt, num_gap+1)
        nt_F = nt_F.astype(int)
        for ii in np.arange(1, num_gap):
            nt1 = nt_F[ii]-int(nt_transition/2)
            nt2 = nt_F[ii]+int(nt_transition/2)
            if ii%2 == 1:
                F_all[nt_F[ii]:nt_F[ii+1]] = F_min
                F_all[nt1:nt2] = F_dec
            else:
                F_all[nt1:nt2] = F_inc
    
    nt_rec = np.linspace(0, Nt, int(Nt/5e4*num_gap/5)+1)
    Ek_now = np.array(0)
    Ep_now = np.array(0)        
    nt_rec = nt_rec.astype(int)
    
    Ep = np.zeros(Nt)
    Ek = np.zeros(Nt)
    y_up = np.zeros(Nt)    
    F_up = np.zeros(Nt)
    cont = np.zeros(Nt)
    p = np.zeros(Nt)
    
    y_bot = Amp_Vibr*np.sin(Freq_Vibr*dt*np.arange(Nt)+1.5*np.pi)+Amp_Vibr
    vy_bot = Amp_Vibr*Freq_Vibr*np.cos(Freq_Vibr*dt*np.arange(Nt)+1.5*np.pi)
    
    vx = np.zeros(N+1)
    vy = np.zeros(N+1)    
    
    ax_old = np.zeros(N+1)
    ay_old = np.zeros(N+1)
    
    x = np.array(x_ini)
    y = np.array(y_ini)
    
    #t_start = time.time()
    
    for nt in np.arange(Nt):
        x = x+vx*dt+ax_old*dt**2/2;  # first step in Verlet integration
        y = y+vy*dt+ay_old*dt**2/2;
        y_up[nt] = y[N]
        
        Fx = np.zeros(N)
        Fy = np.zeros(N) 

        Fx, Fy, Fup_now, Fbot_now, Ep_now, cont_now, p_now, cont_up_now = force_YFixed_collision_ConstP(beta, Fx, Fy, N, x, y, vx, vy, D0[0:N], Lx, y_bot[nt], vy_bot[nt], y[N])

        F_up[nt] = Fup_now-F_all[nt]        
        Ep[nt] = Ep_now
        cont[nt] = cont_now
        p[nt] = p_now
        
        Fx_all = np.append(Fx,0)
        Fy_all = np.append(Fy, F_up[nt])
        
        ax = np.divide(Fx_all, m0)
        ay = np.divide(Fy_all, m0)
        vx = vx+(ax_old+ax)*dt/2;  # second step in Verlet integration
        vy = vy+(ay_old+ay)*dt/2;        
        ax_old = ax;
        ay_old = ay;
        Ek_up = 0.5*m0[N]*(vx[N]**2+vy[N]**2)
        Ek[nt] = sum(0.5*np.multiply(m0,np.multiply(vx, vx)+np.multiply(vy, vy)))-Ek_up
        
    
    for ii in np.arange(len(nt_rec)-1):
        Ek_now = np.append(Ek_now, np.mean(Ek[nt_rec[ii]:nt_rec[ii+1]]))
        Ep_now = np.append(Ep_now, np.mean(Ep[nt_rec[ii]:nt_rec[ii+1]]))

    #t_end = time.time()
    #print ("time=%.3e" %(t_end-t_start))
    print ("freq=%f, cont_min=%d, cont_max=%d, cont_ave=%f\n" %(Freq_Vibr, min(cont), max(cont), np.mean(cont)))
    
    nt_F = np.arange(0, Nt, 100)
    return nt_F, F_all[nt_F], y_up, nt_rec, Ek_now, Ep_now

def MD_SPSignal(mark_collision, beta, Nt, N, x_ini, y_ini,D0, m0, Lx, Ly, Freq_Vibr, Amp_Vibr, ind_in, ind_out, ind_fix, dr_scheme, num_gap, mark_vibrY, dr_one, dr_two):
    
    dt = D0[0]/40
    Nt = int(Nt)
    d_ini = D0[0]
    d0 = 0.1    
    
    dr_all = np.zeros(Nt)+dr_one
    
    if abs(dr_scheme) <= 2:
        nt_dr = np.linspace(0, Nt, 3)
        nt_dr = nt_dr.astype(int)        
        dr_all[nt_dr[1]:nt_dr[2]] = dr_two
        num_gap = 5
    elif dr_scheme == 3 or dr_scheme == 4:
        nt_dr = np.linspace(0, Nt, num_gap+1)
        nt_dr = nt_dr.astype(int)
        for ii in np.arange(1, num_gap, 2):            
            dr_all[nt_dr[ii]:nt_dr[ii+1]] = dr_two

    D_fix = d_ini+dr_all*d_ini
        
    nt_rec = np.linspace(0, Nt, int(Nt/5e4*num_gap/5)+1)
    Ek_rec = np.array(0)
    Ep_rec = np.array(0)
    cont_rec = np.array(0)
    
    nt_rec = nt_rec.astype(int)
    
    Ep = np.zeros(Nt)
    Ek = np.zeros(Nt)   
    cont = np.zeros(Nt)
    p = np.zeros(Nt)
    
    x_out = np.zeros(Nt)
    y_out = np.zeros(Nt)
    
    if mark_vibrY == 1:
        y_in = Amp_Vibr*np.sin(Freq_Vibr*dt*np.arange(Nt))+y_ini[ind_in]
        vy_in = Amp_Vibr*Freq_Vibr*np.cos(Freq_Vibr*dt*np.arange(Nt))
    else:
        x_in = Amp_Vibr*np.sin(Freq_Vibr*dt*np.arange(Nt))+x_ini[ind_in]
        vx_in = Amp_Vibr*Freq_Vibr*np.cos(Freq_Vibr*dt*np.arange(Nt))

    #y_bot = np.zeros(Nt)
    
    vx = np.zeros(N)
    vy = np.zeros(N)    
    
    ax_old = np.zeros(N)
    ay_old = np.zeros(N)
    
    x = np.array(x_ini)
    y = np.array(y_ini)
    
    #t_start = time.time()

    for nt in np.arange(Nt):
        
        D0[ind_fix] = D_fix[nt]
        
        x = x+vx*dt+ax_old*dt**2/2;  # first step in Verlet integration
        y = y+vy*dt+ay_old*dt**2/2;
        
        if mark_vibrY == 1:
            y[ind_in] = y_in[nt]
            x[ind_in] = x_ini[ind_in]
            vy[ind_in] = vy_in[nt]
            vx[ind_in] = 0
        else:            
            x[ind_in] = x_in[nt]
            y[ind_in] = y_ini[ind_in]
            vx[ind_in] = vx_in[nt]
            vy[ind_in] = 0
        
        Fx = np.zeros(N)
        Fy = np.zeros(N) 

        if mark_collision == 1:
            Fx, Fy, Ep_now, cont_now, p_now = force_YFixed_collision_ConstV(beta, Fx, Fy, N, x, y, vx, vy, D0, Lx, 0, Ly)
            Fx_all = Fx
            Fy_all = Fy
        else:
            Fx, Fy, Fup_now, Fbot_now, Ep_now, cont_now, p_now, cont_up = force_YFixed(Fx, Fy, N, x, y, D0, Lx, 0, Ly)                
            Fx_all = Fx-beta*vx
            Fy_all = Fy-beta*vy
            
        
        Ep[nt] = Ep_now
        cont[nt] = cont_now
        p[nt] = p_now                

        x_out[nt] = x[ind_out]
        y_out[nt] = y[ind_out]
        
        ax = np.divide(Fx_all, m0)
        ay = np.divide(Fy_all, m0)
        vx = vx+(ax_old+ax)*dt/2;  # second step in Verlet integration
        vy = vy+(ay_old+ay)*dt/2;        
        ax_old = ax;
        ay_old = ay;
        Ek[nt] = sum(0.5*np.multiply(m0,np.multiply(vx, vx)+np.multiply(vy, vy)))
        
        if nt % 2000 == 0:
            print ("nt = %d, Ek = %.2e, cont = %.2e" %(nt, Ek[nt], cont[nt]))
     
    
    for ii in np.arange(len(nt_rec)-1):
        Ek_rec = np.append(Ek_rec, np.mean(Ek[nt_rec[ii]:nt_rec[ii+1]]))
        Ep_rec = np.append(Ep_rec, np.mean(Ep[nt_rec[ii]:nt_rec[ii+1]]))
        cont_rec = np.append(cont_rec, np.mean(cont[nt_rec[ii]:nt_rec[ii+1]]))

    #t_end = time.time()
    #print ("time=%.3e" %(t_end-t_start))
    print ("freq=%f, cont_min=%d, cont_max=%d, cont_ave=%f\n" %(Freq_Vibr, min(cont), max(cont), np.mean(cont)))

    nt_dr = np.arange(0, Nt, 100)
    
    if mark_vibrY == 1:
        xy_out = y_out
    else:
        xy_out = x_out
    return nt_dr, dr_all[nt_dr], xy_out, nt_rec, Ek_rec, Ep_rec, cont_rec


def MD_YFixed_equi_SP_modecheck(Nt, N, x0, y0, D0, m0, Lx, Ly, T_set, V_em, n_em, ind_out):
    
    dt = min(D0)/40
    Nt = int(Nt)
    Ep = np.zeros(Nt)
    Ek = np.zeros(Nt)
    cont = np.zeros(Nt)
    x_out = np.zeros(Nt)
    y_out = np.zeros(Nt)
    
    nt_rec = np.linspace(0, Nt, int(Nt/5e4)+1)
    nt_rec = nt_rec.astype(int)
    Ek_rec = []
    Ep_rec = []
    cont_rec = []

    vx = np.zeros(N)
    vy = np.zeros(N)
    for ii in np.arange(n_em):
        ind1 = 2*np.arange(N)
        ind2 = ind1+1
        vx += V_em[ind1, ii]
        vy += V_em[ind2, ii]
        
    T_rd = sum(0.5*np.multiply(m0,np.multiply(vx, vx)+np.multiply(vy, vy)))
    vx = vx*np.sqrt(N*T_set/T_rd)
    vy = vy*np.sqrt(N*T_set/T_rd)
       
    ax_old = np.zeros(N)
    ay_old = np.zeros(N)
    
    x = np.array(x0)
    y = np.array(y0)
    
    #t_start = time.time()
    for nt in np.arange(Nt):
        x = x+vx*dt+ax_old*dt**2/2;  # first step in Verlet integration
        y = y+vy*dt+ay_old*dt**2/2;
        x_out[nt] = x[ind_out]
        y_out[nt] = y[ind_out]

        Fx = np.zeros(N)
        Fy = np.zeros(N) 
        Fx, Fy, Fup_now, Fbot_now, Ep_now, cont_now, p_now, cont_up = force_YFixed(Fx, Fy, N, x, y, D0, Lx, 0, Ly)                                       
        Ep[nt] = Ep_now
        cont[nt] = cont_now
        
        ax = np.divide(Fx, m0);
        ay = np.divide(Fy, m0);
        vx = vx+(ax_old+ax)*dt/2;  # second step in Verlet integration
        vy = vy+(ay_old+ay)*dt/2;        
        ax_old = ax;
        ay_old = ay;
        Ek[nt] = sum(0.5*np.multiply(m0,np.multiply(vx, vx)+np.multiply(vy, vy)))
        
    
    for ii in np.arange(len(nt_rec)-1):
        Ek_rec.append(np.mean(Ek[nt_rec[ii]:nt_rec[ii+1]]))
        Ep_rec.append(np.mean(Ep[nt_rec[ii]:nt_rec[ii+1]]))
        cont_rec.append(np.mean(cont[nt_rec[ii]:nt_rec[ii+1]]))
        
    nt_rec = (nt_rec[1:] + nt_rec[:-1]) / 2
    CB_ratio = min(cont)/max(cont)    
    print ("cont_min/cont_max=%f\n" %(CB_ratio))
    Freq_Vibr = 0    
    freq_x, fft_x = FFT_Fup(int(Nt/2), x_out[int(Nt/2):Nt], dt, Freq_Vibr)
    freq_y, fft_y = FFT_Fup(int(Nt/2), y_out[int(Nt/2):Nt], dt, Freq_Vibr)
    ind1 = freq_x<30
    ind2 = freq_y<30
    return freq_x[ind1], freq_y[ind2], fft_x[ind1], fft_y[ind2], np.mean(cont), nt_rec, Ek_rec, Ep_rec, cont_rec


def MD_YFixed_SPVibr_SP_modecheck(Nt, N, x0, y0, D0, m0, Lx, Ly, T_set, ind_in, ind_out, mark_vibrY):
  
    dt = min(D0)/40
    Nt = int(Nt)
    Ep = np.zeros(Nt)
    Ek = np.zeros(Nt)
    cont = np.zeros(Nt)
    x_out = np.zeros(Nt)
    y_out = np.zeros(Nt)
    
    nt_rec = np.linspace(0, Nt, int(Nt/5e4)+1)
    nt_rec = nt_rec.astype(int)
    Ek_rec = np.array(0)
    Ep_rec = np.array(0)
    cont_rec = np.array(0)
 

    vx = np.zeros(N)
    vy = np.zeros(N)
    if mark_vibrY == 1:
        vy[ind_in] = 1
        vy_mc = sum(np.multiply(vy,m0))/sum(m0)
        vy = vy-vy_mc
    else:
        vx[ind_in] = 1
        vx_mc = sum(np.multiply(vx,m0))/sum(m0)
        vx = vx-vx_mc
        
    T_rd = sum(0.5*np.multiply(m0,np.multiply(vx, vx)+np.multiply(vy, vy)))
    if mark_vibrY == 1:
        vy = vy*np.sqrt(N*T_set/T_rd)
        print("|vy|_Max=%.3e, |vy|_Min=%.3e" %(max(abs(vy)), min(abs(vy))))
    else:
        vx = vx*np.sqrt(N*T_set/T_rd)
        print("|vx|_Max=%.3e, |vx|_Min=%.3e" %(max(abs(vx)), min(abs(vx))))
    
    
    ax_old = np.zeros(N)
    ay_old = np.zeros(N)
    
    x = np.array(x0)
    y = np.array(y0)
    
    #t_start = time.time()
    mark_CB = 0
    for nt in np.arange(Nt):
        x = x+vx*dt+ax_old*dt**2/2;  # first step in Verlet integration
        y = y+vy*dt+ay_old*dt**2/2;
        x_out[nt] = x[ind_out]
        y_out[nt] = y[ind_out]

        Fx = np.zeros(N)
        Fy = np.zeros(N) 
        Fx, Fy, Fup_now, Fbot_now, Ep_now, cont_now, p_now, cont_up = force_YFixed(Fx, Fy, N, x, y, D0, Lx, 0, Ly)                                       
        Ep[nt] = Ep_now
        cont[nt] = cont_now
        if mark_CB == 0 and cont_now<cont[0]:
            print("nt_CB=%d" % nt)
            mark_CB = 1
        
        ax = np.divide(Fx, m0);
        ay = np.divide(Fy, m0);
        vx = vx+(ax_old+ax)*dt/2;  # second step in Verlet integration
        vy = vy+(ay_old+ay)*dt/2;        
        ax_old = ax;
        ay_old = ay;
        Ek[nt] = sum(0.5*np.multiply(m0,np.multiply(vx, vx)+np.multiply(vy, vy)))
        
    
    for ii in np.arange(len(nt_rec)-1):
        Ek_rec = np.append(Ek_rec, np.mean(Ek[nt_rec[ii]:nt_rec[ii+1]]))
        Ep_rec = np.append(Ep_rec, np.mean(Ep[nt_rec[ii]:nt_rec[ii+1]]))
        cont_rec = np.append(cont_rec, np.mean(cont[nt_rec[ii]:nt_rec[ii+1]]))

    CB_ratio = min(cont)/max(cont)    
    print ("cont_min/cont_max=%f\n" %(CB_ratio))
    Freq_Vibr = 0 
    freq_x, fft_x = FFT_Fup(int(Nt/2), x_out[int(Nt/2):Nt], dt, Freq_Vibr)
    freq_y, fft_y = FFT_Fup(int(Nt/2), y_out[int(Nt/2):Nt], dt, Freq_Vibr)
    ind1 = freq_x<30
    ind2 = freq_y<30
    return freq_x[ind1], freq_y[ind2], fft_x[ind1], fft_y[ind2], cont_rec, nt_rec, Ek_rec, Ep_rec

#181105
def MD_YFixed_SPVibr_vCorr_modecheck(Nt_MD, Nt_FFT, N, x0, y0, D0, m0, Lx, Ly, T_set, ind_in, ind_out, mark_vibrY):

    N = int(N)
    Nt_FFT = int(Nt_FFT)
    Nt_MD = int(Nt_MD)
    dt = min(D0)/40
    Nt = Nt_MD+Nt_FFT    
    Ep = np.zeros(Nt)
    Ek = np.zeros(Nt)
    cont = np.zeros(Nt)
    
    mark_FFT = np.zeros(Nt)
    mark_FFT[Nt_MD:Nt] = 1
    
    nt_rec = np.linspace(0, Nt, int(Nt/5e4)+1)
    nt_rec = nt_rec.astype(int)
    Ek_rec = []
    Ep_rec = []
    cont_rec = []

    vx = np.zeros(N)
    vy = np.zeros(N)
    if mark_vibrY == 1:
        vy[ind_in] = 1
        vy_mc = sum(np.multiply(vy,m0))/sum(m0)
        vy = vy-vy_mc
    else:
        vx[ind_in] = 1
        vx_mc = sum(np.multiply(vx,m0))/sum(m0)
        vx = vx-vx_mc
        
    T_rd = sum(0.5*np.multiply(m0,np.multiply(vx, vx)+np.multiply(vy, vy)))
    if mark_vibrY == 1:
        vy = vy*np.sqrt(N*T_set/T_rd)
        print("|vy|_Max=%.3e, |vy|_Min=%.3e" %(max(abs(vy)), min(abs(vy))))
    else:
        vx = vx*np.sqrt(N*T_set/T_rd)
        print("|vx|_Max=%.3e, |vx|_Min=%.3e" %(max(abs(vx)), min(abs(vx))))
    
    ax_old = np.zeros(N)
    ay_old = np.zeros(N)
    
    x = np.array(x0)
    y = np.array(y0)
    
    #t_start = time.time()
    for nt in np.arange(Nt):
        x = x+vx*dt+ax_old*dt**2/2;  # first step in Verlet integration
        y = y+vy*dt+ay_old*dt**2/2;

        Fx = np.zeros(N)
        Fy = np.zeros(N)                        
        Fx, Fy, Fup_now, Fbot_now, Ep_now, cont_now, p_now, cont_up = force_YFixed(Fx, Fy, N, x, y, D0, Lx, 0, Ly)                                       
        
        Ep[nt] = Ep_now
        cont[nt] = cont_now
        
        ax = np.divide(Fx, m0);
        ay = np.divide(Fy, m0);
        vx = vx+(ax_old+ax)*dt/2;  # second step in Verlet integration
        vy = vy+(ay_old+ay)*dt/2;        
        ax_old = ax;
        ay_old = ay;
        Ek[nt] = sum(0.5*np.multiply(m0,np.multiply(vx, vx)+np.multiply(vy, vy)))
        
        if mark_FFT[nt] == 1:
            if mark_FFT[nt-1] == 0:
                nt_ref = nt
                vx_rec = np.zeros([Nt_FFT, N])
                vy_rec = np.zeros([Nt_FFT, N])
            nt_delta = nt-nt_ref 
            vx_rec[nt_delta] = vx
            vy_rec[nt_delta] = vy
            if nt_delta == Nt_FFT-1:
                freq_now, fft_now = FFT_vCorr(Nt_FFT, N, vx_rec, vy_rec, dt)
                print ("Nt_End="+str(nt))
                   
    for ii in np.arange(len(nt_rec)-1):
        Ek_rec.append(np.mean(Ek[nt_rec[ii]:nt_rec[ii+1]]))
        Ep_rec.append(np.mean(Ep[nt_rec[ii]:nt_rec[ii+1]]))
        cont_rec.append(np.mean(cont[nt_rec[ii]:nt_rec[ii+1]]))
        
    CB_ratio = min(cont)/max(cont)    
    print ("cont_min/cont_max=%f" %(CB_ratio))
                
    return freq_now, fft_now, (nt_rec[:-1]+nt_rec[1:])/2, Ek_rec, Ep_rec, cont_rec

def MD_YFixed_ConstV(B, Nt, N, x0, y0, D0, m0, Lx, Ly):
  
    dt = min(D0)/40
    Nt = int(Nt)
    Ep = np.zeros(Nt)
    Ek = np.zeros(Nt)
    cont = np.zeros(Nt)
    
    nt_rec = np.linspace(0, Nt, int(Nt/1e3)+1)
    nt_rec = nt_rec.astype(int)
    Ek_rec = []
    Ep_rec = []
    cont_rec = []

    vx = np.zeros(N)
    vy = np.zeros(N)    
    ax_old = np.zeros(N)
    ay_old = np.zeros(N)
    
    x = np.array(x0)
    y = np.array(y0)
    
    #t_start = time.time()
    mark_CB = 0
    for nt in np.arange(Nt):
        x = x+vx*dt+ax_old*dt**2/2;  # first step in Verlet integration
        y = y+vy*dt+ay_old*dt**2/2;

        Fx = np.zeros(N)
        Fy = np.zeros(N) 
        Fx, Fy, Fup_now, Fbot_now, Ep_now, cont_now, p_now, cont_up = force_YFixed(Fx, Fy, N, x, y, D0, Lx, 0, Ly)                                       
        Fx = Fx-B*vx
        Fy = Fy-B*vy
        
        Ep[nt] = Ep_now
        cont[nt] = cont_now
        if mark_CB == 0 and cont_now<cont[0]:
            print("nt_CB=%d" % nt)
            mark_CB = 1
        
        ax = np.divide(Fx, m0);
        ay = np.divide(Fy, m0);
        vx = vx+(ax_old+ax)*dt/2;  # second step in Verlet integration
        vy = vy+(ay_old+ay)*dt/2;        
        ax_old = ax;
        ay_old = ay;
        Ek[nt] = sum(0.5*np.multiply(m0,np.multiply(vx, vx)+np.multiply(vy, vy)))
            
    for ii in np.arange(len(nt_rec)-1):
        Ek_rec.append(np.mean(Ek[nt_rec[ii]:nt_rec[ii+1]]))
        Ep_rec.append(np.mean(Ep[nt_rec[ii]:nt_rec[ii+1]]))
        cont_rec.append(np.mean(cont[nt_rec[ii]:nt_rec[ii+1]]))
    nt_rec = (nt_rec[0:-1]+nt_rec[1:])/2

    CB_ratio = min(cont)/max(cont)    
    print ("cont_min/cont_max=%f\n" %(CB_ratio))
    print ("Ek_last=%.3e" % Ek[-1])

    return x, y, nt_rec, Ek_rec, Ep_rec, cont_rec


def MD_Vibr3Part_ConstV(B, Nt, N, x_ini, y_ini, D0, m0, L, Freq_Vibr, Amp_Vibr, ind_in_all, ind_out, mark_vibrY, eigen_mode_now):
    
    dt = D0[0]/40
    Nt = int(Nt)
    
    nt_rec = np.linspace(0, Nt, int(Nt/5e4)+1)    
    
    nt_rec = nt_rec.astype(int)
    
    Ep = np.zeros(Nt)
    Ek = np.zeros(Nt)   
    cont = np.zeros(Nt)
    p = np.zeros(Nt)
    
    x_out = np.zeros(Nt)
    y_out = np.zeros(Nt)
    
    
    num_in = ind_in_all.size
    Phase_Vibr = np.sin(Freq_Vibr*dt*np.arange(Nt))
    Amp_Vibr_all = np.zeros(num_in)
    for i_in in np.arange(num_in):
        ind_in = ind_in_all[i_in]
        if mark_vibrY == 0:            
            Amp_Vibr_all[i_in] = eigen_mode_now[2*ind_in]
        elif mark_vibrY == 1:
            Amp_Vibr_all[i_in] = eigen_mode_now[2*ind_in+1]
    print(ind_in_all)
    print(Amp_Vibr_all)  
    Amp_Vibr_all = Amp_Vibr_all*Amp_Vibr/max(np.abs(Amp_Vibr_all))
    print(Amp_Vibr_all)    

    vx = np.zeros(N)
    vy = np.zeros(N)    
    
    ax_old = np.zeros(N)
    ay_old = np.zeros(N)
    
    x = np.array(x_ini)
    y = np.array(y_ini)
    
    #t_start = time.time()

    for nt in np.arange(Nt):
        x = x+vx*dt+ax_old*dt**2/2;  # first step in Verlet integration
        y = y+vy*dt+ay_old*dt**2/2;
        
        for i_in in np.arange(num_in):
            ind_in = ind_in_all[i_in]
            if mark_vibrY == 0:
                x[ind_in] = Phase_Vibr[nt]*Amp_Vibr_all[i_in]+x_ini[ind_in]                
                y[ind_in] = y_ini[ind_in]
            elif mark_vibrY == 1:
                x[ind_in] = x_ini[ind_in]
                y[ind_in] = Phase_Vibr[nt]*Amp_Vibr_all[i_in]+y_ini[ind_in]      
        
        Fx = np.zeros(N)
        Fy = np.zeros(N) 

        Fx, Fy, Fup_now, Fbot_now, Ep_now, cont_now, p_now, cont_up = force_YFixed(Fx, Fy, N, x, y, D0, L[0], 0, L[1])
        
        Ep[nt] = Ep_now
        cont[nt] = cont_now
        p[nt] = p_now
        
        Fx_all = Fx-B*vx
        Fy_all = Fy-B*vy

        x_out[nt] = x[ind_out]
        y_out[nt] = y[ind_out]
        
        ax = np.divide(Fx_all, m0)
        ay = np.divide(Fy_all, m0)
        vx = vx+(ax_old+ax)*dt/2;  # second step in Verlet integration
        vy = vy+(ay_old+ay)*dt/2;        
        ax_old = ax;
        ay_old = ay;
        Ek[nt] = sum(0.5*np.multiply(m0,np.multiply(vx, vx)+np.multiply(vy, vy)))
    
    Ek_now = np.array(0)
    Ep_now = np.array(0)
    cont_now = np.array(0)
    for ii in np.arange(len(nt_rec)-1):
        Ek_now = np.append(Ek_now, np.mean(Ek[nt_rec[ii]:nt_rec[ii+1]]))
        Ep_now = np.append(Ep_now, np.mean(Ep[nt_rec[ii]:nt_rec[ii+1]]))
        cont_now = np.append(cont_now, np.mean(cont[nt_rec[ii]:nt_rec[ii+1]]))
    cont_now[0] = cont_now[1]

    #t_end = time.time()
    #print ("time=%.3e" %(t_end-t_start))
    CB_ratio = min(cont)/max(cont)
    print ("freq=%f, cont_min/cont_max=%f\n" %(Freq_Vibr, CB_ratio))
    
    if mark_vibrY == 0:
        x_in = Phase_Vibr*Amp_Vibr
        freq_fft, fft_in = FFT_Fup(int(Nt/2), x_in[int(Nt/2):Nt], dt, Freq_Vibr)        
    elif mark_vibrY == 1:
        y_in = Phase_Vibr*Amp_Vibr
        freq_fft, fft_in = FFT_Fup(int(Nt/2), y_in[int(Nt/2):Nt], dt, Freq_Vibr)

    freq_fft, fft_x_out = FFT_Fup(int(Nt/2), x_out[int(Nt/2):Nt], dt, Freq_Vibr)
    freq_fft, fft_y_out = FFT_Fup(int(Nt/2), y_out[int(Nt/2):Nt], dt, Freq_Vibr)
       
    return freq_fft, fft_in, fft_x_out, fft_y_out, np.mean(cont), nt_rec, Ek_now, Ep_now, cont_now


def MD_dPhiSignal(mark_collision, beta, Nt, N, x_ini, y_ini, d0, phi0, m0, Lx, Ly, Freq_Vibr, Amp_Vibr, ind_in, ind_out, dphi_scheme, dphi_on, dphi_off, num_gap, mark_vibrY):
    
    dt = d0/40
    Nt = int(Nt)
    
    
    if dphi_scheme == 1:
        nt_dphi = np.linspace(0, Nt, 3)
        nt_dphi = nt_dphi.astype(int)        
        dphi_all = np.zeros(Nt)+dphi_on
        dphi_all[nt_dphi[1]:nt_dphi[2]] = dphi_off
    elif dphi_scheme == -1:
        nt_dphi = np.linspace(0, Nt, 3)
        nt_dphi = nt_dphi.astype(int)        
        dphi_all = np.zeros(Nt)+dphi_off
        dphi_all[nt_dphi[1]:nt_dphi[2]] = dphi_on                
    else:            
        dphi_all = np.zeros(Nt)+dphi_on
        nt_dphi = np.linspace(0, Nt, num_gap+1)
        nt_dphi = nt_dphi.astype(int)
        for ii in np.arange(1, num_gap, 2):            
            dphi_all[nt_dphi[ii]:nt_dphi[ii+1]] = dphi_off


    D_ini = np.zeros(N)+d0
    
    nt_rec = np.linspace(0, Nt, int(Nt/5e4*num_gap/5)+1)
    Ek_rec = np.array(0)
    Ep_rec = np.array(0)
    cont_rec = np.array(0)
    
    nt_rec = nt_rec.astype(int)
    
    Ep = np.zeros(Nt)
    Ek = np.zeros(Nt)   
    cont = np.zeros(Nt)
    p = np.zeros(Nt)
    
    x_out = np.zeros(Nt)
    y_out = np.zeros(Nt)
    
    if mark_vibrY == 1:
        y_in = Amp_Vibr*np.sin(Freq_Vibr*dt*np.arange(Nt))+y_ini[ind_in]
        vy_in = Amp_Vibr*Freq_Vibr*np.cos(Freq_Vibr*dt*np.arange(Nt))
    else:
        x_in = Amp_Vibr*np.sin(Freq_Vibr*dt*np.arange(Nt))+x_ini[ind_in]
        vx_in = Amp_Vibr*Freq_Vibr*np.cos(Freq_Vibr*dt*np.arange(Nt))

    #y_bot = np.zeros(Nt)
    
    vx = np.zeros(N)
    vy = np.zeros(N)    
    
    ax_old = np.zeros(N)
    ay_old = np.zeros(N)
    
    x = np.array(x_ini)
    y = np.array(y_ini)
    
    #t_start = time.time()

    for nt in np.arange(Nt):
        
        D0 = D_ini*np.sqrt(1+dphi_all[nt]/phi0)
        #if np.mod(nt,100000) == 0:
            #print(D0[3])
        
        x = x+vx*dt+ax_old*dt**2/2;  # first step in Verlet integration
        y = y+vy*dt+ay_old*dt**2/2;
        
        if mark_vibrY == 1:
            y[ind_in] = y_in[nt]
            x[ind_in] = x_ini[ind_in]
            vy[ind_in] = vy_in[nt]
            vx[ind_in] = 0
        else:            
            x[ind_in] = x_in[nt]
            y[ind_in] = y_ini[ind_in]
            vx[ind_in] = vx_in[nt]
            vy[ind_in] = 0
        
        Fx = np.zeros(N)
        Fy = np.zeros(N) 

        if mark_collision == 1:
            Fx, Fy, Ep_now, cont_now, p_now = force_YFixed_collision_ConstV(beta, Fx, Fy, N, x, y, vx, vy, D0, Lx, 0, Ly)
            Fx_all = Fx
            Fy_all = Fy
        else:
            Fx, Fy, Fup_now, Fbot_now, Ep_now, cont_now, p_now, cont_up = force_YFixed(Fx, Fy, N, x, y, D0, Lx, 0, Ly)                
            Fx_all = Fx-beta*vx
            Fy_all = Fy-beta*vy
            
        
        Ep[nt] = Ep_now
        cont[nt] = cont_now
        p[nt] = p_now                

        x_out[nt] = x[ind_out]
        y_out[nt] = y[ind_out]
        
        ax = np.divide(Fx_all, m0)
        ay = np.divide(Fy_all, m0)
        vx = vx+(ax_old+ax)*dt/2;  # second step in Verlet integration
        vy = vy+(ay_old+ay)*dt/2;        
        ax_old = ax;
        ay_old = ay;
        Ek[nt] = sum(0.5*np.multiply(m0,np.multiply(vx, vx)+np.multiply(vy, vy)))
    
    for ii in np.arange(len(nt_rec)-1):
        Ek_rec = np.append(Ek_rec, np.mean(Ek[nt_rec[ii]:nt_rec[ii+1]]))
        Ep_rec = np.append(Ep_rec, np.mean(Ep[nt_rec[ii]:nt_rec[ii+1]]))
        cont_rec = np.append(cont_rec, np.mean(cont[nt_rec[ii]:nt_rec[ii+1]]))

    #t_end = time.time()
    #print ("time=%.3e" %(t_end-t_start))
    print ("freq=%f, cont_min=%d, cont_max=%d, cont_ave=%f\n" %(Freq_Vibr, min(cont), max(cont), np.mean(cont)))

    nt_dphi = np.arange(0, Nt, 100)
    
    if mark_vibrY == 1:
        xy_out = y_out
    else:
        xy_out = x_out
    return nt_dphi, dphi_all[nt_dphi], xy_out, nt_rec[1:], Ek_rec[1:], Ep_rec[1:], cont_rec[1:]


def Damping_calc(Damp_scheme, B, N, x, y, vx, vy, Lx, Ly):
    
    Fx_damp = np.zeros(N)
    Fy_damp = np.zeros(N)
    if Damp_scheme == 1:
        Fx_damp = -B*vx
        Fy_damp = -B*vy
    if Damp_scheme == 2:
        Fx_damp = -B*vx*np.abs(vx)*5e5
        Fy_damp = -B*vy*np.abs(vy)*5e5
    if Damp_scheme == 3:
        Fx_damp = -B*vx/np.sqrt(np.abs(vx))*np.sqrt(2e-6)
        Fy_damp = -B*vy/np.sqrt(np.abs(vy))*np.sqrt(2e-6)
    if Damp_scheme == 4:
        Fx_damp = -B*vx*np.exp(-5e4*np.abs(vx)+1)*0.1
        Fy_damp = -B*vy*np.exp(-5e4*np.abs(vy)+1)*0.1
    if Damp_scheme == 5:
        Fx_damp = -B*vx*np.exp(-5e5*np.abs(vx)+1)
        Fy_damp = -B*vy*np.exp(-5e5*np.abs(vy)+1)
    if Damp_scheme == 6:
        Fx_damp = -B*vx*np.exp(-5e6*np.abs(vx)+1)*10
        Fy_damp = -B*vy*np.exp(-5e6*np.abs(vy)+1)*10
    if Damp_scheme == 7:
        Fx_damp = -B*vx*np.exp(-5e7*np.abs(vx)+1)*100
        Fy_damp = -B*vy*np.exp(-5e7*np.abs(vy)+1)*100
    return Fx_damp, Fy_damp

def Force_FixedPos_calc(k, N, x, y, x0, y0, D0, vx, vy, Lx, Ly):
    Fx_damp = np.zeros(N)
    Fy_damp = np.zeros(N)
    Ep = 0
    for nn in np.arange(N):        
        dy = y[nn]-y0[nn]
        dy = dy-round(dy/Ly)*Ly
        Dmn = 0.5*D0[nn]
        dx = x[nn]-x0[nn]
        dx = dx-round(dx/Lx)*Lx
        dmn = np.sqrt(dx**2+dy**2)
        if (dmn > 0):
            F = -k*(dmn/Dmn/Dmn)/dmn
            Fx_damp[nn] += F*dx
            Fy_damp[nn] += F*dy
            Ep += (1/2)*k*(dmn/Dmn)**2   

    return Fx_damp, Fy_damp, Ep

def MD_FilterCheck_Periodic_Equi_vCorr(Nt_damp, Nt_FFT, num_period, Damp_scheme, B, N, x0, y0, D0, m0, L, T_set, V_em, n_em):
    
    if Damp_scheme < 0:
        return
    N = int(N)
    Nt_FFT = int(Nt_FFT)
    Nt_damp = int(Nt_damp)
    dt = min(D0)/40
    Nt_period = int(2*Nt_damp+Nt_FFT)
    Nt = Nt_period*num_period
    Ep = np.zeros(Nt)
    Ek = np.zeros(Nt)
    cont = np.zeros(Nt)
    mark_damp = np.zeros(Nt)
    mark_FFT = np.zeros(Nt)
    for ii in np.arange(num_period):
        if ii > 0:
            t1 = ii*Nt_period
            t2 = t1+Nt_damp
            mark_damp[t1:t2] = 1
        t3 = ii*Nt_period+2*Nt_damp
        t4 = t3+Nt_FFT
        mark_FFT[t3:t4] = 1
    
    nt_rec = np.linspace(0, Nt, int(Nt/1e3)+1)
    nt_rec = nt_rec.astype(int)
    Ek_rec = []
    Ep_rec = []
    cont_rec = []
    num_FFT = 0

    vx = np.zeros(N)
    vy = np.zeros(N)
    for ii in np.arange(n_em):
        ind1 = 2*np.arange(N)
        ind2 = ind1+1
        vx += V_em[ind1, ii]
        vy += V_em[ind2, ii]
        
    T_rd = sum(0.5*np.multiply(m0,np.multiply(vx, vx)+np.multiply(vy, vy)))
    vx = vx*np.sqrt(N*T_set/T_rd)
    vy = vy*np.sqrt(N*T_set/T_rd)
    
    ax_old = np.zeros(N)
    ay_old = np.zeros(N)
    
    x = np.array(x0)
    y = np.array(y0)
    
    #t_start = time.time()
    for nt in np.arange(Nt):
        x = x+vx*dt+ax_old*dt**2/2;  # first step in Verlet integration
        y = y+vy*dt+ay_old*dt**2/2;

        Fx = np.zeros(N)
        Fy = np.zeros(N)                        
        Fx, Fy, Ep_now, cont_now, p_now = force_Regular(Fx, Fy, N, x, y, D0, L[0], L[1])
        
        if mark_damp[nt] == 1:
            Fx_damp, Fy_damp = Damping_calc(Damp_scheme, B, N, x, y, vx, vy, L[0], L[1])                     
            Fx = Fx + Fx_damp
            Fy = Fy + Fy_damp
        
        Ep[nt] = Ep_now
        cont[nt] = cont_now
        
        ax = np.divide(Fx, m0);
        ay = np.divide(Fy, m0);
        vx = vx+(ax_old+ax)*dt/2;  # second step in Verlet integration
        vy = vy+(ay_old+ay)*dt/2;        
        ax_old = ax;
        ay_old = ay;
        Ek[nt] = sum(0.5*np.multiply(m0,np.multiply(vx, vx)+np.multiply(vy, vy)))
        
        if mark_FFT[nt] == 1:
            if mark_FFT[nt-1] == 0:
                nt_ref = nt
                vx_rec = np.zeros([Nt_FFT, N])
                vy_rec = np.zeros([Nt_FFT, N])
            nt_delta = nt-nt_ref 
            vx_rec[nt_delta] = vx
            vy_rec[nt_delta] = vy
            if nt_delta == Nt_FFT-1:
                num_FFT += 1
                freq_now, fft_now = FFT_vCorr(Nt_FFT, N, vx_rec, vy_rec, dt)
                if num_FFT == 1:
                    fft_all = np.array([fft_now])
                    freq_all = np.array([freq_now])
                    len_fft_ref = len(fft_now)
                    len_freq_ref = len(freq_now)
                else:
                    fft_add = np.zeros(len_fft_ref)
                    freq_add = np.zeros(len_freq_ref)
                    len_fft_now = len(fft_now)
                    len_freq_now = len(freq_now)
                    if len_fft_now >= len_fft_ref:
                        fft_add[0:len_fft_ref] = fft_now[0:len_fft_ref]
                    else:
                        fft_add[0:len_fft_now] = fft_now[0:len_fft_now]
                        fft_add[len_fft_now:] = fft_now[len_fft_now]
                    
                    if len_freq_now >= len_freq_ref:
                        freq_add[0:len_freq_ref] = freq_now[0:len_freq_ref]
                    else:
                        freq_add[0:len_freq_now] = freq_now[0:len_freq_now]
                        freq_add[len_freq_now:] = freq_now[len_freq_now]
                    
                    fft_all = np.append(fft_all, [fft_add], axis=0)
                    freq_all = np.append(freq_all, [freq_add], axis=0)

                print("FFT_iteration: %d" % num_FFT)
                print("Ek_ave: %e" %(np.mean(Ek[nt_ref:nt])))
                ind1 = m0>5
                ind2 = m0<5
                print("|vx|_ave(heavy):%e" % np.mean(np.abs(vx[ind1])))
                print("|vx|_ave(light):%e" % np.mean(np.abs(vx[ind2])))
                    
    for ii in np.arange(len(nt_rec)-1):
        Ek_rec.append(np.mean(Ek[nt_rec[ii]:nt_rec[ii+1]]))
        Ep_rec.append(np.mean(Ep[nt_rec[ii]:nt_rec[ii+1]]))
        cont_rec.append(np.mean(cont[nt_rec[ii]:nt_rec[ii+1]]))
        
    CB_ratio = min(cont)/max(cont)    
    print ("cont_min/cont_max=%f" %(CB_ratio))
                
    return freq_all, fft_all, (nt_rec[:-1]+nt_rec[1:])/2, Ek_rec, Ep_rec, cont_rec


def MD_FilterCheck_Periodic_Equi_vCorr_Seperate(Nt_damp, Nt_FFT, num_period, Damp_scheme, k, B, N, x0, y0, D0, m0, L, T_set, V_em, n_em):
    
    # for damping scheme = -1 (fixed spring at initial position)
    if Damp_scheme != -1:
        return
    N = int(N)
    Nt_FFT = int(Nt_FFT)
    Nt_damp = int(Nt_damp)
    dt = min(D0)/40
    Nt = Nt_damp*num_period+Nt_FFT
    if num_period == 0:
        Nt = Nt_damp+Nt_FFT        
    Ep = np.zeros(Nt)
    Ek = np.zeros(Nt)
    cont = np.zeros(Nt)
    mark_FFT = np.zeros(Nt)
    
    t1 = Nt_damp * num_period
    if num_period == 0:
        t1 = Nt_damp 
    t2 = t1 + Nt_FFT
    mark_FFT[t1:t2] = 1
    
    nt_rec = np.linspace(0, Nt, int(Nt/1e3)+1)
    nt_rec = nt_rec.astype(int)
    Ek_rec = []
    Ep_rec = []
    cont_rec = []

    vx = np.zeros(N)
    vy = np.zeros(N)
    for ii in np.arange(n_em):
        ind1 = 2*np.arange(N)
        ind2 = ind1+1
        vx += V_em[ind1, ii]
        vy += V_em[ind2, ii]
        
    T_rd = sum(0.5*np.multiply(m0,np.multiply(vx, vx)+np.multiply(vy, vy)))
    vx = vx*np.sqrt(N*T_set/T_rd)
    vy = vy*np.sqrt(N*T_set/T_rd)
    
    ax_old = np.zeros(N)
    ay_old = np.zeros(N)
    
    x = np.array(x0)
    y = np.array(y0)
    
    #t_start = time.time()
    for nt in np.arange(Nt):
        x = x+vx*dt+ax_old*dt**2/2;  # first step in Verlet integration
        y = y+vy*dt+ay_old*dt**2/2;

        Fx = np.zeros(N)
        Fy = np.zeros(N)                        
        Fx, Fy, Ep_now, cont_now, p_now = force_Regular(Fx, Fy, N, x, y, D0, L[0], L[1])
        
        # always have damping exceot num_period = 0
        if num_period > 0:
            Fx_damp, Fy_damp, Ep_fix = Force_FixedPos_calc(k, N, x, y, x0, y0, D0, vx, vy, L[0], L[1])
            if (B > 0):
                Fx_damp += -B*vx
                Fy_damp += -B*vy
        elif num_period == 0:
            Fx_damp = 0
            Fy_damp = 0
            Ep_fix = 0
        
        
            
        Ep_now += Ep_fix           
        Fx = Fx + Fx_damp
        Fy = Fy + Fy_damp
        
        Ep[nt] = Ep_now
        cont[nt] = cont_now
        
        ax = np.divide(Fx, m0);
        ay = np.divide(Fy, m0);
        vx = vx+(ax_old+ax)*dt/2;  # second step in Verlet integration
        vy = vy+(ay_old+ay)*dt/2;        
        ax_old = ax;
        ay_old = ay;
        Ek[nt] = sum(0.5*np.multiply(m0,np.multiply(vx, vx)+np.multiply(vy, vy)))
        
        if mark_FFT[nt] == 1:
            if mark_FFT[nt-1] == 0:
                nt_ref = nt
                vx_rec = np.zeros([Nt_FFT, N])
                vy_rec = np.zeros([Nt_FFT, N])
            nt_delta = nt-nt_ref 
            vx_rec[nt_delta] = vx
            vy_rec[nt_delta] = vy
            if nt_delta == Nt_FFT-1:
                freq_now, fft_now = FFT_vCorr(Nt_FFT, N, vx_rec, vy_rec, dt)
                print ("Nt_End="+str(nt))
                   
    for ii in np.arange(len(nt_rec)-1):
        Ek_rec.append(np.mean(Ek[nt_rec[ii]:nt_rec[ii+1]]))
        Ep_rec.append(np.mean(Ep[nt_rec[ii]:nt_rec[ii+1]]))
        cont_rec.append(np.mean(cont[nt_rec[ii]:nt_rec[ii+1]]))
        
    CB_ratio = min(cont)/max(cont)    
    print ("cont_min/cont_max=%f" %(CB_ratio))
                
    return freq_now, fft_now, (nt_rec[:-1]+nt_rec[1:])/2, Ek_rec, Ep_rec, cont_rec



def MD_Periodic_Equi_vDistr(Nt_MD, Nt_rec, N, x0, y0, D0, m0, L, T_set, V_em, n_em):
    N = int(N)
    Nt_MD = int(Nt_MD)
    Nt_rec = int(Nt_rec)
    dt = min(D0)/40
    Nt = Nt_MD+Nt_rec
    Ep = np.zeros(Nt)
    Ek = np.zeros(Nt)
    cont = np.zeros(Nt)
    
    nt_rec = np.linspace(0, Nt, int(Nt/1e3)+1)
    nt_rec = nt_rec.astype(int)
    Ek_rec = []
    Ep_rec = []
    cont_rec = []
    
    ind1 = m0>5
    ind2 = m0<5
    vx_light = []
    vx_heavy = []
    vy_light = []
    vy_heavy = []
    
    vx = np.zeros(N)
    vy = np.zeros(N)
    for ii in np.arange(n_em):
        ind1 = 2*np.arange(N)
        ind2 = ind1+1
        vx += V_em[ind1, ii]
        vy += V_em[ind2, ii]
        
    T_rd = sum(0.5*np.multiply(m0,np.multiply(vx, vx)+np.multiply(vy, vy)))
    vx = vx*np.sqrt(N*T_set/T_rd)
    vy = vy*np.sqrt(N*T_set/T_rd)
    
    ax_old = np.zeros(N)
    ay_old = np.zeros(N)
    
    x = np.array(x0)
    y = np.array(y0)    
    
    #t_start = time.time()
    for nt in np.arange(Nt):
        x = x+vx*dt+ax_old*dt**2/2;  # first step in Verlet integration
        y = y+vy*dt+ay_old*dt**2/2;

        Fx = np.zeros(N)
        Fy = np.zeros(N)                        
        Fx, Fy, Ep_now, cont_now, p_now = force_Regular(Fx, Fy, N, x, y, D0, L[0], L[1])
                      
        Ep[nt] = Ep_now
        cont[nt] = cont_now
        
        ax = np.divide(Fx, m0);
        ay = np.divide(Fy, m0);
        vx = vx+(ax_old+ax)*dt/2;  # second step in Verlet integration
        vy = vy+(ay_old+ay)*dt/2;        
        ax_old = ax;
        ay_old = ay;
        Ek[nt] = sum(0.5*np.multiply(m0,np.multiply(vx, vx)+np.multiply(vy, vy)))
        
        if nt >= Nt_MD:
            vx_light.extend(vx[ind2])
            vy_light.extend(vy[ind2])
            vx_heavy.extend(vx[ind1])
            vy_heavy.extend(vy[ind1])
                                
    for ii in np.arange(len(nt_rec)-1):
        Ek_rec.append(np.mean(Ek[nt_rec[ii]:nt_rec[ii+1]]))
        Ep_rec.append(np.mean(Ep[nt_rec[ii]:nt_rec[ii+1]]))
        cont_rec.append(np.mean(cont[nt_rec[ii]:nt_rec[ii+1]]))
        
    CB_ratio = min(cont)/max(cont)    
    print ("cont_min/cont_max=%f" %(CB_ratio))
    
    return nt_rec, Ek_rec, Ep_rec, cont_rec, vx_light, vx_heavy, vy_light, vy_heavy

def Output_resonator_1D(Nt, x_drive, x0, m0, w0, dt):
    dx = x_drive - x0
    k = w0**2*m0
               
    x = 0
    vx = 0
    ax_old = 0  
    Nt = int(Nt)
    x_rec = np.zeros(Nt)
    
    for nt in np.arange(Nt):
        x = x+vx*dt+ax_old*dt**2/2;  # first step in Verlet integration
        x_rec[nt] = x  
        Fx = k*(dx[nt]-x)                
        ax = Fx/m0;
        vx = vx+(ax_old+ax)*dt/2;  # second step in Verlet integration       
        ax_old = ax;              

    freq_fft, fft_x_rec = FFT_Fup(int(Nt/2), x_rec[int(Nt/2):Nt], dt, w0)
    return freq_fft, fft_x_rec

def MD_Periodic_vCorr(Nt, N, x0, y0, D0, m0, vx0, vy0, L, T_set):
    
    dt = min(D0)/40
    Nt = int(Nt)
    Ep = np.zeros(Nt)
    Ek = np.zeros(Nt)
    cont = np.zeros(Nt)
    vx_rec = np.zeros([int(Nt/2), N])
    vy_rec = np.zeros([int(Nt/2), N])
 
    vx = vx0
    vy = vy0    
        
    T_rd = sum(0.5*np.multiply(m0,np.multiply(vx, vx)+np.multiply(vy, vy)))
    vx = vx*np.sqrt(N*T_set/T_rd)
    vy = vy*np.sqrt(N*T_set/T_rd)
    
    ax_old = np.zeros(N)
    ay_old = np.zeros(N)
    
    x = np.array(x0)
    y = np.array(y0)
    
    for nt in np.arange(Nt):
        x = x+vx*dt+ax_old*dt**2/2;  # first step in Verlet integration
        y = y+vy*dt+ay_old*dt**2/2;

        Fx = np.zeros(N)
        Fy = np.zeros(N)                        
        Fx, Fy, Ep_now, cont_now, p_now = force_Regular(Fx, Fy, N, x, y, D0, L[0], L[1])
        Ep[nt] = Ep_now
        cont[nt] = cont_now
        
        ax = np.divide(Fx, m0);
        ay = np.divide(Fy, m0);
        vx = vx+(ax_old+ax)*dt/2;  # second step in Verlet integration
        vy = vy+(ay_old+ay)*dt/2;        
        ax_old = ax;
        ay_old = ay;
        Ek[nt] = sum(0.5*np.multiply(m0,np.multiply(vx, vx)+np.multiply(vy, vy)))
        if (nt >= Nt/2):
            vx_rec[int(nt-Nt/2)] = vx
            vy_rec[int(nt-Nt/2)] = vy
        
    CB_ratio = min(cont)/max(cont)    
    print ("cont_min/cont_max=%f\n" %(CB_ratio))
    
    freq_now, fft_now = FFT_vCorr(int(Nt/2), N, vx_rec, vy_rec, dt)        
    return freq_now, fft_now, np.mean(cont)  

def MD_Period_ConstV_SD(Nt, N, x0, y0, D0, m0, Lx, Ly):    
    
    dt = D0[0]/40
    Nt = int(Nt)
    Ep = np.zeros(Nt)
    F_tot = np.zeros(Nt)
    
    vx = np.zeros(N)
    vy = np.zeros(N)    
    x = np.array(x0)
    y = np.array(y0)
    
    #t_start = time.time()
    for nt in np.arange(Nt):        
        Fx = np.zeros(N)
        Fy = np.zeros(N)        
        Fx, Fy, Ep_now, cont_now, p_now = force_Regular(Fx, Fy, N, x, y, D0, Lx, Ly)
        Ep[nt] = Ep_now
        vx = 0.1*np.divide(Fx, m0)
        vy = 0.1*np.divide(Fy, m0)
        x += vx*dt
        y += vy*dt
        F_tot[nt] = sum(np.absolute(Fx)+np.absolute(Fy))
    #t_end = time.time()
    print ("F_tot=%.3e" %(F_tot[nt]))
    #print ("time=%.3e" %(t_end-t_start))

    return x, y, p_now


def Force_FixedPos_YFixed_calc(k, N, x, y, x0, y0, D0, vx, vy, Lx, Ly):
    Fx_damp = np.zeros(N)
    Fy_damp = np.zeros(N)
    Ep = 0
    for nn in np.arange(N):        
        dy = y[nn]-y0[nn]
        Dmn = 0.5*D0[nn]
        dx = x[nn]-x0[nn]
        dx = dx-round(dx/Lx)*Lx
        dmn = np.sqrt(dx**2+dy**2)
        if (dmn > 0):
            F = -k*(dmn/Dmn/Dmn)/dmn
            Fx_damp[nn] += F*dx
            Fy_damp[nn] += F*dy
            Ep += (1/2)*k*(dmn/Dmn)**2   

    return Fx_damp, Fy_damp, Ep

def MD_VibrSP_ConstV_Yfixed_FixSpr(k, B, Nt, N, x_ini, y_ini, D0, m0, L, Freq_Vibr, Amp_Vibr, ind_in, ind_out):
    
    mark_vibrY = 0
    mark_resonator = 1
    dt = D0[0]/40
    Nt = int(Nt)
    
    nt_rec = np.linspace(0, Nt, int(Nt/5e4)+1)    
    
    nt_rec = nt_rec.astype(int)
    
    Ep = np.zeros(Nt)
    Ek = np.zeros(Nt)   
    cont = np.zeros(Nt)
    p = np.zeros(Nt)
    
    x_out = np.zeros(Nt)
    y_out = np.zeros(Nt)
    
    if mark_vibrY == 0:
        x_in = Amp_Vibr*np.sin(Freq_Vibr*dt*np.arange(Nt))+x_ini[ind_in]
    elif mark_vibrY == 1:
        y_in = Amp_Vibr*np.sin(Freq_Vibr*dt*np.arange(Nt))+y_ini[ind_in]

    #y_bot = np.zeros(Nt)
    
    vx = np.zeros(N)
    vy = np.zeros(N)    
    
    ax_old = np.zeros(N)
    ay_old = np.zeros(N)
    
    x = np.array(x_ini)
    y = np.array(y_ini)
    
    #t_start = time.time()

    for nt in np.arange(Nt):
        x = x+vx*dt+ax_old*dt**2/2;  # first step in Verlet integration
        y = y+vy*dt+ay_old*dt**2/2;
        
        if mark_vibrY == 0:
            x[ind_in] = x_in[nt]
            y[ind_in] = y_ini[ind_in]
        elif mark_vibrY == 1:
            x[ind_in] = x_ini[ind_in]
            y[ind_in] = y_in[nt]        
        
        Fx = np.zeros(N)
        Fy = np.zeros(N) 

        Fx, Fy, Fup_now, Fbot_now, Ep_now, cont_now, p_now, cont_up = force_YFixed(Fx, Fy, N, x, y, D0, L[0], 0, L[1])
        
        Fx_damp, Fy_damp, Ep_fix = Force_FixedPos_YFixed_calc(k, N, x, y, x_ini, y_ini, D0, vx, vy, L[0], L[1])
        #Fx_damp = 0; Fy_damp = 0; Ep_fix = 0
        Fx_damp += -B*vx
        Fy_damp += -B*vy
        
        Ep[nt] = Ep_now + Ep_fix
        cont[nt] = cont_now
        p[nt] = p_now
        
        Fx_all = Fx+Fx_damp
        Fy_all = Fy+Fy_damp

        x_out[nt] = x[ind_out]
        y_out[nt] = y[ind_out]
        
        ax = np.divide(Fx_all, m0)
        ay = np.divide(Fy_all, m0)
        vx = vx+(ax_old+ax)*dt/2;  # second step in Verlet integration
        vy = vy+(ay_old+ay)*dt/2;        
        ax_old = ax;
        ay_old = ay;
        Ek[nt] = sum(0.5*np.multiply(m0,np.multiply(vx, vx)+np.multiply(vy, vy)))
    
    
    
    Ek_now = []
    Ep_now = []
    cont_now = []
    for ii in np.arange(len(nt_rec)-1):
        Ek_now.append(np.mean(Ek[nt_rec[ii]:nt_rec[ii+1]]))
        Ep_now.append(np.mean(Ep[nt_rec[ii]:nt_rec[ii+1]]))
        cont_now.append(np.mean(cont[nt_rec[ii]:nt_rec[ii+1]]))
    nt_rec = (nt_rec[1:] + nt_rec[:-1]) / 2

    #t_end = time.time()
    #print ("time=%.3e" %(t_end-t_start))
    CB_ratio = min(cont)/max(cont)
    print ("freq=%f, cont_min/cont_max=%f, Ek_mean=%.3e, Ep_mean=%.3e\n" %(Freq_Vibr, CB_ratio, np.mean(Ek), np.mean(Ep)))
    
    if mark_vibrY == 0:
        freq_fft, fft_in = FFT_Fup(int(Nt/2), x_in[int(Nt/2):Nt], dt, Freq_Vibr)        
    elif mark_vibrY == 1:
        freq_fft, fft_in = FFT_Fup(int(Nt/2), y_in[int(Nt/2):Nt], dt, Freq_Vibr)
    
    if mark_resonator == 0:
        freq_fft, fft_x_out = FFT_Fup(int(Nt/2), x_out[int(Nt/2):Nt], dt, Freq_Vibr)
        freq_fft, fft_y_out = FFT_Fup(int(Nt/2), y_out[int(Nt/2):Nt], dt, Freq_Vibr)
    elif mark_resonator == 1:
        freq_fft, fft_x_out = Output_resonator_1D(Nt, x_out[0:Nt], x_ini[ind_out], m0[ind_out], Freq_Vibr, dt)
        freq_fft, fft_y_out = Output_resonator_1D(Nt, y_out[0:Nt], y_ini[ind_out], m0[ind_out], Freq_Vibr, dt)
       
        
    return freq_fft, fft_in, fft_x_out, fft_y_out, np.mean(cont), nt_rec, Ek_now, Ep_now, cont_now

def MD_VibrSP_ConstV_Yfixed_FixSpr2(k, B, Nt, N, x_ini, y_ini, D0, m0, L, Freq_Vibr, Amp_Vibr1, ind_in1, Amp_Vibr2, ind_in2, ind_out):
    
    mark_vibrY = 0
    mark_resonator = 0
    dt = D0[0]/40
    Nt = int(Nt)
    
    nt_rec = np.linspace(0, Nt, int(Nt/5e4)+1)    
    
    nt_rec = nt_rec.astype(int)
    
    Ep = np.zeros(Nt)
    Ek = np.zeros(Nt)   
    cont = np.zeros(Nt)
    p = np.zeros(Nt)
    
    x_out = np.zeros(Nt)
    y_out = np.zeros(Nt)
    
    if mark_vibrY == 0:
        x_in1 = Amp_Vibr1*np.sin(Freq_Vibr*dt*np.arange(Nt))+x_ini[ind_in1]
        x_in2 = Amp_Vibr2*np.sin(Freq_Vibr*dt*np.arange(Nt))+x_ini[ind_in2]
    elif mark_vibrY == 1:
        y_in1 = Amp_Vibr1*np.sin(Freq_Vibr*dt*np.arange(Nt))+y_ini[ind_in1]
        y_in2 = Amp_Vibr2*np.sin(Freq_Vibr*dt*np.arange(Nt))+y_ini[ind_in2]

    #y_bot = np.zeros(Nt)
    
    vx = np.zeros(N)
    vy = np.zeros(N)    
    
    ax_old = np.zeros(N)
    ay_old = np.zeros(N)
    
    x = np.array(x_ini)
    y = np.array(y_ini)
    
    #t_start = time.time()

    for nt in np.arange(Nt):
        x = x+vx*dt+ax_old*dt**2/2;  # first step in Verlet integration
        y = y+vy*dt+ay_old*dt**2/2;
        
        if mark_vibrY == 0:
            x[ind_in1] = x_in1[nt]
            y[ind_in1] = y_ini[ind_in1]
            x[ind_in2] = x_in2[nt]
            y[ind_in2] = y_ini[ind_in2]
        elif mark_vibrY == 1:
            x[ind_in1] = x_ini[ind_in1]
            y[ind_in1] = y_in1[nt]
            x[ind_in2] = x_ini[ind_in2]
            y[ind_in2] = y_in2[nt]
        
        Fx = np.zeros(N)
        Fy = np.zeros(N) 

        Fx, Fy, Fup_now, Fbot_now, Ep_now, cont_now, p_now, cont_up = force_YFixed(Fx, Fy, N, x, y, D0, L[0], 0, L[1])
        
        Fx_damp, Fy_damp, Ep_fix = Force_FixedPos_YFixed_calc(k, N, x, y, x_ini, y_ini, D0, vx, vy, L[0], L[1])
        #Fx_damp = 0; Fy_damp = 0; Ep_fix = 0
        Fx_damp += -B*vx
        Fy_damp += -B*vy
        
        Ep[nt] = Ep_now + Ep_fix
        cont[nt] = cont_now
        p[nt] = p_now
        
        Fx_all = Fx+Fx_damp
        Fy_all = Fy+Fy_damp

        x_out[nt] = x[ind_out]
        y_out[nt] = y[ind_out]
        
        ax = np.divide(Fx_all, m0)
        ay = np.divide(Fy_all, m0)
        vx = vx+(ax_old+ax)*dt/2;  # second step in Verlet integration
        vy = vy+(ay_old+ay)*dt/2;        
        ax_old = ax;
        ay_old = ay;
        Ek[nt] = sum(0.5*np.multiply(m0,np.multiply(vx, vx)+np.multiply(vy, vy)))
    
    
    
    Ek_now = []
    Ep_now = []
    cont_now = []
    for ii in np.arange(len(nt_rec)-1):
        Ek_now.append(np.mean(Ek[nt_rec[ii]:nt_rec[ii+1]]))
        Ep_now.append(np.mean(Ep[nt_rec[ii]:nt_rec[ii+1]]))
        cont_now.append(np.mean(cont[nt_rec[ii]:nt_rec[ii+1]]))
    nt_rec = (nt_rec[1:] + nt_rec[:-1]) / 2

    #t_end = time.time()
    #print ("time=%.3e" %(t_end-t_start))
    CB_ratio = min(cont)/max(cont)
    #print ("freq=%f, cont_min/cont_max=%f, Ek_mean=%.3e, Ep_mean=%.3e\n" %(Freq_Vibr, CB_ratio, np.mean(Ek), np.mean(Ep)))
    
    if mark_vibrY == 0:
        freq_fft, fft_in1 = FFT_Fup(int(Nt/2), x_in1[int(Nt/2):Nt], dt, Freq_Vibr)
        freq_fft, fft_in2 = FFT_Fup(int(Nt/2), x_in2[int(Nt/2):Nt], dt, Freq_Vibr)        
    elif mark_vibrY == 1:
        freq_fft, fft_in1 = FFT_Fup(int(Nt/2), y_in1[int(Nt/2):Nt], dt, Freq_Vibr)
        freq_fft, fft_in2 = FFT_Fup(int(Nt/2), y_in2[int(Nt/2):Nt], dt, Freq_Vibr)
    
    if mark_resonator == 0:
        freq_fft, fft_x_out = FFT_Fup(int(Nt/2), x_out[int(Nt/2):Nt], dt, Freq_Vibr)
        freq_fft, fft_y_out = FFT_Fup(int(Nt/2), y_out[int(Nt/2):Nt], dt, Freq_Vibr)
    elif mark_resonator == 1:
        freq_fft, fft_x_out = Output_resonator_1D(Nt, x_out[0:Nt], x_ini[ind_out], m0[ind_out], Freq_Vibr, dt)
        freq_fft, fft_y_out = Output_resonator_1D(Nt, y_out[0:Nt], y_ini[ind_out], m0[ind_out], Freq_Vibr, dt)
       
        
    return freq_fft, fft_in1, fft_in2, fft_x_out, fft_y_out, np.mean(cont), nt_rec, Ek_now, Ep_now, cont_now

def MD_VibrSP_ConstV_Yfixed_FixSpr3(k, B, Nt, N, x_ini, y_ini, D0, m0, L, Freq_Vibr1, Amp_Vibr1, ind_in1, Freq_Vibr2, Amp_Vibr2, ind_in2, ind_out):
    
    mark_vibrY = 0
    mark_resonator = 0
    dt = D0[0]/40
    Nt = int(Nt)
    
    nt_rec = np.linspace(0, Nt, int(Nt/5e4)+1)    
    
    nt_rec = nt_rec.astype(int)
    
    Ep = np.zeros(Nt)
    Ek = np.zeros(Nt)   
    cont = np.zeros(Nt)
    p = np.zeros(Nt)
    
    x_out = np.zeros(Nt)
    y_out = np.zeros(Nt)
    
    if mark_vibrY == 0:
        x_in1 = Amp_Vibr1*np.sin(Freq_Vibr1*dt*np.arange(Nt))+x_ini[ind_in1]
        x_in2 = Amp_Vibr2*np.sin(Freq_Vibr2*dt*np.arange(Nt))+x_ini[ind_in2]
    elif mark_vibrY == 1:
        y_in1 = Amp_Vibr1*np.sin(Freq_Vibr1*dt*np.arange(Nt))+y_ini[ind_in1]
        y_in2 = Amp_Vibr2*np.sin(Freq_Vibr2*dt*np.arange(Nt))+y_ini[ind_in2]

    #y_bot = np.zeros(Nt)
    
    vx = np.zeros(N)
    vy = np.zeros(N)    
    
    ax_old = np.zeros(N)
    ay_old = np.zeros(N)
    
    x = np.array(x_ini)
    y = np.array(y_ini)
    
    #t_start = time.time()

    for nt in np.arange(Nt):
        x = x+vx*dt+ax_old*dt**2/2;  # first step in Verlet integration
        y = y+vy*dt+ay_old*dt**2/2;
        
        if mark_vibrY == 0:
            x[ind_in1] = x_in1[nt]
            y[ind_in1] = y_ini[ind_in1]
            x[ind_in2] = x_in2[nt]
            y[ind_in2] = y_ini[ind_in2]
        elif mark_vibrY == 1:
            x[ind_in1] = x_ini[ind_in1]
            y[ind_in1] = y_in1[nt]
            x[ind_in2] = x_ini[ind_in2]
            y[ind_in2] = y_in2[nt]
        
        Fx = np.zeros(N)
        Fy = np.zeros(N) 

        Fx, Fy, Fup_now, Fbot_now, Ep_now, cont_now, p_now, cont_up = force_YFixed(Fx, Fy, N, x, y, D0, L[0], 0, L[1])
        
        Fx_damp, Fy_damp, Ep_fix = Force_FixedPos_YFixed_calc(k, N, x, y, x_ini, y_ini, D0, vx, vy, L[0], L[1])
        #Fx_damp = 0; Fy_damp = 0; Ep_fix = 0
        Fx_damp += -B*vx
        Fy_damp += -B*vy
        
        Ep[nt] = Ep_now + Ep_fix
        cont[nt] = cont_now
        p[nt] = p_now
        
        Fx_all = Fx+Fx_damp
        Fy_all = Fy+Fy_damp

        x_out[nt] = x[ind_out]
        y_out[nt] = y[ind_out]
        
        ax = np.divide(Fx_all, m0)
        ay = np.divide(Fy_all, m0)
        vx = vx+(ax_old+ax)*dt/2;  # second step in Verlet integration
        vy = vy+(ay_old+ay)*dt/2;        
        ax_old = ax;
        ay_old = ay;
        Ek[nt] = sum(0.5*np.multiply(m0,np.multiply(vx, vx)+np.multiply(vy, vy)))
    
    
    
    Ek_now = []
    Ep_now = []
    cont_now = []
    for ii in np.arange(len(nt_rec)-1):
        Ek_now.append(np.mean(Ek[nt_rec[ii]:nt_rec[ii+1]]))
        Ep_now.append(np.mean(Ep[nt_rec[ii]:nt_rec[ii+1]]))
        cont_now.append(np.mean(cont[nt_rec[ii]:nt_rec[ii+1]]))
    nt_rec = (nt_rec[1:] + nt_rec[:-1]) / 2

    #t_end = time.time()
    #print ("time=%.3e" %(t_end-t_start))
    CB_ratio = min(cont)/max(cont)
    print ("freq=%f, cont_min/cont_max=%f, Ek_mean=%.3e, Ep_mean=%.3e\n" %(Freq_Vibr1, CB_ratio, np.mean(Ek), np.mean(Ep)))
    
    if mark_vibrY == 0:
        freq_fft, fft_in1 = FFT_Fup(int(Nt/2), x_in1[int(Nt/2):Nt], dt, Freq_Vibr1)
        freq_fft, fft_in2 = FFT_Fup(int(Nt/2), x_in2[int(Nt/2):Nt], dt, Freq_Vibr2)       
    elif mark_vibrY == 1:
        freq_fft, fft_in1 = FFT_Fup(int(Nt/2), y_in1[int(Nt/2):Nt], dt, Freq_Vibr1)
        freq_fft, fft_in2 = FFT_Fup(int(Nt/2), y_in2[int(Nt/2):Nt], dt, Freq_Vibr2)
    
    if mark_resonator == 0:
        freq_fft, fft_x_out = FFT_Fup(int(Nt/2), x_out[int(Nt/2):Nt], dt, Freq_Vibr1)
        freq_fft, fft_y_out = FFT_Fup(int(Nt/2), y_out[int(Nt/2):Nt], dt, Freq_Vibr1)
    elif mark_resonator == 1:
        freq_fft, fft_x_out = Output_resonator_1D(Nt, x_out[0:Nt], x_ini[ind_out], m0[ind_out], Freq_Vibr1, dt)
        freq_fft, fft_y_out = Output_resonator_1D(Nt, y_out[0:Nt], y_ini[ind_out], m0[ind_out], Freq_Vibr1, dt)
       
        
    return freq_fft, fft_in1, fft_in2, fft_x_out, fft_y_out, np.mean(cont), nt_rec, Ek_now, Ep_now, cont_now

def MD_VibrSP_Force_ConstV(B, Nt, N, x_ini, y_ini, D0, m0, L, Freq_Vibr, Amp_Vibr, ind_in, ind_out, mark_vibrY, mark_resonator):
    
    dt = D0[0]/40
    Nt = int(Nt)
    
    nt_rec = np.linspace(0, Nt, int(Nt/5e4)+1)    
    
    nt_rec = nt_rec.astype(int)
    
    Ep = np.zeros(Nt)
    Ek = np.zeros(Nt)   
    cont = np.zeros(Nt)
    p = np.zeros(Nt)
    
    x_out = np.zeros(Nt)
    y_out = np.zeros(Nt)
    x_in = np.zeros(Nt)
    y_in = np.zeros(Nt)
    
    if mark_vibrY == 0:
        Fx_in = Amp_Vibr*np.sin(Freq_Vibr*dt*np.arange(Nt))
    elif mark_vibrY == 1:
        Fy_in = Amp_Vibr*np.sin(Freq_Vibr*dt*np.arange(Nt))

    #y_bot = np.zeros(Nt)
    
    vx = np.zeros(N)
    vy = np.zeros(N)    
    
    ax_old = np.zeros(N)
    ay_old = np.zeros(N)
    
    x = np.array(x_ini)
    y = np.array(y_ini)
    
    #t_start = time.time()

    for nt in np.arange(Nt):
        x = x+vx*dt+ax_old*dt**2/2;  # first step in Verlet integration
        y = y+vy*dt+ay_old*dt**2/2;
        x_in[nt] = x[ind_in]
        y_in[nt] = y[ind_in]        
                     
        Fx = np.zeros(N)
        Fy = np.zeros(N) 

        Fx, Fy, Fup_now, Fbot_now, Ep_now, cont_now, p_now, cont_up = force_YFixed(Fx, Fy, N, x, y, D0, L[0], 0, L[1])
        
        Ep[nt] = Ep_now
        cont[nt] = cont_now
        p[nt] = p_now
        
        Fx_all = Fx-B*vx
        Fy_all = Fy-B*vy
        
        if mark_vibrY == 0:
            Fx_all[ind_in] += Fx_in[nt]            
        elif mark_vibrY == 1:
            Fy_all[ind_in] += Fy_in[nt]    

        x_out[nt] = x[ind_out]
        y_out[nt] = y[ind_out]
        
        ax = np.divide(Fx_all, m0)
        ay = np.divide(Fy_all, m0)
        vx = vx+(ax_old+ax)*dt/2;  # second step in Verlet integration
        vy = vy+(ay_old+ay)*dt/2;        
        ax_old = ax;
        ay_old = ay;
        Ek[nt] = sum(0.5*np.multiply(m0,np.multiply(vx, vx)+np.multiply(vy, vy)))
    
    
    
    Ek_now = np.array(0)
    Ep_now = np.array(0)
    cont_now = np.array(0)
    for ii in np.arange(len(nt_rec)-1):
        Ek_now = np.append(Ek_now, np.mean(Ek[nt_rec[ii]:nt_rec[ii+1]]))
        Ep_now = np.append(Ep_now, np.mean(Ep[nt_rec[ii]:nt_rec[ii+1]]))
        cont_now = np.append(cont_now, np.mean(cont[nt_rec[ii]:nt_rec[ii+1]]))
    cont_now[0] = cont_now[1]

    #t_end = time.time()
    #print ("time=%.3e" %(t_end-t_start))
    CB_ratio = min(cont)/max(cont)
    print ("freq=%f, cont_min/cont_max=%f\n" %(Freq_Vibr, CB_ratio))
    
    if mark_vibrY == 0:
        freq_fft, fft_in = FFT_Fup(int(Nt/2), x_in[int(Nt/2):Nt], dt, Freq_Vibr)        
    elif mark_vibrY == 1:
        freq_fft, fft_in = FFT_Fup(int(Nt/2), y_in[int(Nt/2):Nt], dt, Freq_Vibr)
    
    if mark_resonator == 0:
        freq_fft, fft_x_out = FFT_Fup(int(Nt/2), x_out[int(Nt/2):Nt], dt, Freq_Vibr)
        freq_fft, fft_y_out = FFT_Fup(int(Nt/2), y_out[int(Nt/2):Nt], dt, Freq_Vibr)
    elif mark_resonator == 1:
        freq_fft, fft_x_out = Output_resonator_1D(Nt, x_out[0:Nt], x_ini[ind_out], m0[ind_out], Freq_Vibr, dt)
        freq_fft, fft_y_out = Output_resonator_1D(Nt, y_out[0:Nt], y_ini[ind_out], m0[ind_out], Freq_Vibr, dt)
       
    if Nt == 5e5:
        print(x[ind_out], y[ind_out])
        print(fft_x_out[100], fft_y_out[100])
        print(fft_in[100])
        
    return freq_fft, fft_in, fft_x_out, fft_y_out, np.mean(cont), nt_rec, Ek_now, Ep_now, cont_now



def MD_VibrSP_ConstV_ConfigCB(B, Nt, N, x_ini, y_ini, D0, m0, L, Freq_Vibr, Amp_Vibr, ind_in, ind_out, Nt_rec):
    
    mark_vibrY = 0
    dt = D0[0]/40
    Nt = int(Nt)
    
    nt_rec = np.linspace(0, Nt, int(Nt/5e4)+1)    
    
    nt_rec = nt_rec.astype(int)
    
    Ep = np.zeros(Nt)
    Ek = np.zeros(Nt)   
    cont = np.zeros(Nt)
    p = np.zeros(Nt)
    
    x_out = np.zeros(Nt)
    y_out = np.zeros(Nt)
    
    if mark_vibrY == 0:
        x_in = Amp_Vibr*np.sin(Freq_Vibr*dt*np.arange(Nt))+x_ini[ind_in]
    elif mark_vibrY == 1:
        y_in = Amp_Vibr*np.sin(Freq_Vibr*dt*np.arange(Nt))+y_ini[ind_in]

    #y_bot = np.zeros(Nt)
    
    vx = np.zeros(N)
    vy = np.zeros(N)    
    
    ax_old = np.zeros(N)
    ay_old = np.zeros(N)
    
    x = np.array(x_ini)
    y = np.array(y_ini)
    
    #t_start = time.time()

    for nt in np.arange(Nt):
        if nt == Nt_rec:
            x_rec = x[:]
            y_rec = y[:]
        x = x+vx*dt+ax_old*dt**2/2;  # first step in Verlet integration
        y = y+vy*dt+ay_old*dt**2/2;
        
        if mark_vibrY == 0:
            x[ind_in] = x_in[nt]
            y[ind_in] = y_ini[ind_in]
        elif mark_vibrY == 1:
            x[ind_in] = x_ini[ind_in]
            y[ind_in] = y_in[nt]        
        
        Fx = np.zeros(N)
        Fy = np.zeros(N) 

        Fx, Fy, Fup_now, Fbot_now, Ep_now, cont_now, p_now, cont_up = force_YFixed(Fx, Fy, N, x, y, D0, L[0], 0, L[1])
        
        Ep[nt] = Ep_now
        cont[nt] = cont_now
        p[nt] = p_now
        
        Fx_all = Fx-B*vx
        Fy_all = Fy-B*vy

        x_out[nt] = x[ind_out]
        y_out[nt] = y[ind_out]
        
        ax = np.divide(Fx_all, m0)
        ay = np.divide(Fy_all, m0)
        vx = vx+(ax_old+ax)*dt/2;  # second step in Verlet integration
        vy = vy+(ay_old+ay)*dt/2;        
        ax_old = ax;
        ay_old = ay;
        Ek[nt] = sum(0.5*np.multiply(m0,np.multiply(vx, vx)+np.multiply(vy, vy)))
    
    
    
    Ek_now = np.array(0)
    Ep_now = np.array(0)
    cont_now = np.array(0)
    for ii in np.arange(len(nt_rec)-1):
        Ek_now = np.append(Ek_now, np.mean(Ek[nt_rec[ii]:nt_rec[ii+1]]))
        Ep_now = np.append(Ep_now, np.mean(Ep[nt_rec[ii]:nt_rec[ii+1]]))
        cont_now = np.append(cont_now, np.mean(cont[nt_rec[ii]:nt_rec[ii+1]]))
    cont_now[0] = cont_now[1]

    CB_ratio = min(cont)/max(cont)
    print ("freq=%f, cont_min/cont_max=%f\n" %(Freq_Vibr, CB_ratio))    
        
    return x_rec, y_rec

def VL_YFixed_ConstV(N, x, y, D, Lx, VL_list, VL_counter_old, x_save, y_save, first_call):    
    
    r_factor = 1.2
    r_cut = np.amax(D)
    r_list = r_factor * r_cut
    r_list_sq = r_list**2
    r_skin_sq = ((r_factor - 1.0) * r_cut)**2

    if first_call == 0:
        dr_sq_max = 0.0
        for nn in np.arange(N):
            dy = y[nn] - y_save[nn]
            dx = x[nn] - x_save[nn]
            dx = dx - round(dx / Lx) * Lx
            dr_sq = dx**2 + dy**2
            if dr_sq > dr_sq_max:
                dr_sq_max = dr_sq
        if dr_sq_max < r_skin_sq:
            return VL_list, VL_counter_old, x_save, y_save

    VL_counter = 0
    
    for nn in np.arange(N):
        r_now = 0.5*D[nn]

        for mm in np.arange(nn+1, N):
            dy = y[mm]-y[nn]
            Dmn = 0.5*(D[mm]+D[nn])
            if abs(dy) < r_list:
                dx = x[mm]-x[nn]
                dx = dx - round(dx / Lx) * Lx
                if abs(dx) < r_list:
                    dmn_sq = dx**2 + dy**2
                    if dmn_sq < r_list_sq:
                        VL_list[VL_counter][0] = nn
                        VL_list[VL_counter][1] = mm
                        VL_counter += 1

    return VL_list, VL_counter, x, y


def MD_YFixed_ConstV_SP_SD_DiffK(Nt, N, x0, y0, D0, m0, Lx, Ly, k_list, k_type):    
    
    dt = D0[0] * np.sqrt(k_list[2]) / 20.0
    Nt = int(Nt)
    Ep = np.zeros(Nt)
    F_tot = np.zeros(Nt)
    
    vx = np.zeros(N)
    vy = np.zeros(N)    
    x = np.array(x0)
    y = np.array(y0)
    x_save = np.array(x0)
    y_save = np.array(y0)

    Fx = np.zeros(N)
    Fy = np.zeros(N)
    VL_list = np.zeros((N * 10, 2), dtype=int) 
    VL_counter = 0
    VL_list, VL_counter, x_save, y_save = VL_YFixed_ConstV(N, x, y, D0, Lx, VL_list, VL_counter, x_save, y_save, 1)

    t_start = time.time()
    for nt in np.arange(Nt):        
        Fx = np.zeros(N)
        Fy = np.zeros(N)
        VL_list, VL_counter, x_save, y_save = VL_YFixed_ConstV(N, x, y, D0, Lx, VL_list, VL_counter, x_save, y_save, 0)
        Fx, Fy, Fup_now, Fbot_now, Ep_now, cont_now, p_now, cont_up = force_YFixed_DiffK_VL(Fx, Fy, N, x, y, D0, Lx, 0, Ly, k_list, k_type, VL_list, VL_counter)
        Ep[nt] = Ep_now
        vx = 0.1 * Fx
        vy = 0.1 * Fy
        x += vx * dt
        y += vy * dt
        F_tot[nt] = sum(np.absolute(Fx) + np.absolute(Fy))
        # putting a threshold on total force
        if (F_tot[nt] < 1e-11):
            break
    print(nt)
    print(F_tot[nt])
    t_end = time.time()
    #print ("F_tot=%.3e" %(F_tot[nt]))
    #print ("time=%.3e" %(t_end-t_start))

    return x, y, p_now


def FIRE_YFixed_ConstV_DiffK(Nt, N, x0, y0, D0, m0, Lx, Ly, k_list, k_type):  

    dt_md = 0.01 * D0[0] * np.sqrt(k_list[2])
    N_delay = 20
    N_pn_max = 2000
    f_inc = 1.1
    f_dec = 0.5
    a_start = 0.15
    f_a = 0.99
    dt_max = 10.0 * dt_md
    dt_min = 0.05 * dt_md
    initialdelay = 1
    
    Nt = int(Nt)
    Ep = np.zeros(Nt)
    F_tot = np.zeros(Nt)
    
    vx = np.zeros(N)
    vy = np.zeros(N)    
    x = np.array(x0)
    y = np.array(y0)
    x_save = np.array(x0)
    y_save = np.array(y0)

    Fx = np.zeros(N)
    Fy = np.zeros(N)
    VL_list = np.zeros((N * 10, 2), dtype=int) 
    VL_counter = 0
    VL_list, VL_counter, x_save, y_save = VL_YFixed_ConstV(N, x, y, D0, Lx, VL_list, VL_counter, x_save, y_save, 1)
    Fx, Fy, Fup_now, Fbot_now, Ep_now, cont_now, p_now, cont_up = force_YFixed_DiffK_VL(Fx, Fy, N, x, y, D0, Lx, 0, Ly, k_list, k_type, VL_list, VL_counter)
        
    a_fire = a_start
    delta_a_fire = 1.0 - a_fire
    dt = dt_md
    dt_half = dt / 2.0

    N_pp = 0 # number of P being positive
    N_pn = 0 # number of P being negative
    ## FIRE
    for nt in np.arange(Nt):
        # FIRE update
        P = np.dot(vx, Fx) + np.dot(vy, Fy)
        
        if P > 0.0:
            N_pp += 1
            N_pn = 0
            if N_pp > N_delay:
                dt = min(f_inc * dt, dt_max)
                dt_half = dt / 2.0
                a_fire = f_a * a_fire
                delta_a_fire = 1.0 - a_fire
        else:
            N_pp = 0
            N_pn += 1
            if N_pn > N_pn_max:
                break
            if (initialdelay < 0.5) or (nt >= N_delay):
                if f_dec * dt > dt_min:
                    dt = f_dec * dt
                    dt_half = dt / 2.0
                a_fire = a_start
                delta_a_fire = 1.0 - a_fire
                x -= vx * dt_half
                y -= vy * dt_half
                vx = np.zeros(N)
                vy = np.zeros(N)

        # MD using Verlet method
        vx += Fx * dt_half
        vy += Fy * dt_half
        rsc_fire = np.sqrt(np.sum(vx**2 + vy**2)) / np.sqrt(np.sum(Fx**2 + Fy**2))
        vx = delta_a_fire * vx + a_fire * rsc_fire * Fx
        vy = delta_a_fire * vy + a_fire * rsc_fire * Fy
        x += vx * dt
        y += vy * dt

        Fx = np.zeros(N)
        Fy = np.zeros(N)
        VL_list, VL_counter, x_save, y_save = VL_YFixed_ConstV(N, x, y, D0, Lx, VL_list, VL_counter, x_save, y_save, 0)
        Fx, Fy, Fup_now, Fbot_now, Ep_now, cont_now, p_now, cont_up = force_YFixed_DiffK_VL(Fx, Fy, N, x, y, D0, Lx, 0, Ly, k_list, k_type, VL_list, VL_counter)
        Ep[nt] = Ep_now

        F_tot[nt] = sum(np.absolute(Fx) + np.absolute(Fy))
        # putting a threshold on total force
        if (F_tot[nt] < 1e-11):
            break

        vx += Fx * dt_half
        vy += Fy * dt_half

    #print(nt)
    #print(F_tot[nt])
    t_end = time.time()
    #print ("F_tot=%.3e" %(F_tot[nt]))
    #print ("time=%.3e" %(t_end-t_start))

    return x, y, p_now

def MD_VibrSP_ConstV_Yfixed_DiffK(k_list, k_type, B, Nt, N, x_ini, y_ini, D0, m0, L, Freq_Vibr, Amp_Vibr1, ind_in1, Amp_Vibr2, ind_in2, ind_out):
    
    Lx = L[0]
    Ly = L[1]

    mark_vibrY = 0
    mark_resonator = 0
    dt = D0[0]/40
    Nt = int(Nt)
    
    nt_rec = np.linspace(0, Nt, int(Nt/5e4)+1)    
    
    nt_rec = nt_rec.astype(int)
    
    Ep = np.zeros(Nt)
    Ek = np.zeros(Nt)   
    cont = np.zeros(Nt)
    p = np.zeros(Nt)
    F_tot = np.zeros(Nt)
    
    x_out = np.zeros(Nt)
    y_out = np.zeros(Nt)
    
    if mark_vibrY == 0:
        x_in1 = Amp_Vibr1*np.sin(Freq_Vibr*dt*np.arange(Nt))+x_ini[ind_in1]
        x_in2 = Amp_Vibr2*np.sin(Freq_Vibr*dt*np.arange(Nt))+x_ini[ind_in2]
    elif mark_vibrY == 1:
        y_in1 = Amp_Vibr1*np.sin(Freq_Vibr*dt*np.arange(Nt))+y_ini[ind_in1]
        y_in2 = Amp_Vibr2*np.sin(Freq_Vibr*dt*np.arange(Nt))+y_ini[ind_in2]
    
    vx = np.zeros(N)
    vy = np.zeros(N)    
    
    ax_old = np.zeros(N)
    ay_old = np.zeros(N)
    
    x = np.array(x_ini)
    y = np.array(y_ini)
    
    x_save = np.array(x_ini)
    y_save = np.array(y_ini)

    Fx = np.zeros(N)
    Fy = np.zeros(N)
    VL_list = np.zeros((N * 10, 2), dtype=int) 
    VL_counter = 0
    VL_list, VL_counter, x_save, y_save = VL_YFixed_ConstV(N, x, y, D0, Lx, VL_list, VL_counter, x_save, y_save, 1)
    Fx, Fy, Fup_now, Fbot_now, Ep_now, cont_now, p_now, cont_up = force_YFixed_DiffK_VL(Fx, Fy, N, x, y, D0, Lx, 0, Ly, k_list, k_type, VL_list, VL_counter)


    for nt in np.arange(Nt):
        x = x+vx*dt+ax_old*dt**2/2  # first step in Verlet integration
        y = y+vy*dt+ay_old*dt**2/2
        
        if mark_vibrY == 0:
            x[ind_in1] = x_in1[nt]
            y[ind_in1] = y_ini[ind_in1]
            x[ind_in2] = x_in2[nt]
            y[ind_in2] = y_ini[ind_in2]
        elif mark_vibrY == 1:
            x[ind_in1] = x_ini[ind_in1]
            y[ind_in1] = y_in1[nt]
            x[ind_in2] = x_ini[ind_in2]
            y[ind_in2] = y_in2[nt]
        
        Fx = np.zeros(N)
        Fy = np.zeros(N)
        VL_list, VL_counter, x_save, y_save = VL_YFixed_ConstV(N, x, y, D0, Lx, VL_list, VL_counter, x_save, y_save, 0)
        Fx, Fy, Fup_now, Fbot_now, Ep_now, cont_now, p_now, cont_up = force_YFixed_DiffK_VL(Fx, Fy, N, x, y, D0, Lx, 0, Ly, k_list, k_type, VL_list, VL_counter)
        
        Ep[nt] = Ep_now
        cont[nt] = cont_now
        p[nt] = p_now
        
        Fx_all = Fx - B*vx
        Fy_all = Fy - B*vy

        x_out[nt] = x[ind_out]
        y_out[nt] = y[ind_out]
        
        ax = np.divide(Fx_all, m0)
        ay = np.divide(Fy_all, m0)
        vx = vx+(ax_old+ax)*dt/2  # second step in Verlet integration
        vy = vy+(ay_old+ay)*dt/2        
        ax_old = ax
        ay_old = ay
        Ek[nt] = sum(0.5*np.multiply(m0,np.multiply(vx, vx)+np.multiply(vy, vy)))
    
    
    
    Ek_now = []
    Ep_now = []
    cont_now = []
    for ii in np.arange(len(nt_rec)-1):
        Ek_now.append(np.mean(Ek[nt_rec[ii]:nt_rec[ii+1]]))
        Ep_now.append(np.mean(Ep[nt_rec[ii]:nt_rec[ii+1]]))
        cont_now.append(np.mean(cont[nt_rec[ii]:nt_rec[ii+1]]))
    nt_rec = (nt_rec[1:] + nt_rec[:-1]) / 2

    #t_end = time.time()
    #print ("time=%.3e" %(t_end-t_start))
    CB_ratio = min(cont)/max(cont)
    #print ("freq=%f, cont_min/cont_max=%f, Ek_mean=%.3e, Ep_mean=%.3e\n" %(Freq_Vibr, CB_ratio, np.mean(Ek), np.mean(Ep)))
    
    if mark_vibrY == 0:
        freq_fft, fft_in1 = FFT_Fup(int(Nt/2), x_in1[int(Nt/2):Nt], dt, Freq_Vibr)
        freq_fft, fft_in2 = FFT_Fup(int(Nt/2), x_in2[int(Nt/2):Nt], dt, Freq_Vibr)        
    elif mark_vibrY == 1:
        freq_fft, fft_in1 = FFT_Fup(int(Nt/2), y_in1[int(Nt/2):Nt], dt, Freq_Vibr)
        freq_fft, fft_in2 = FFT_Fup(int(Nt/2), y_in2[int(Nt/2):Nt], dt, Freq_Vibr)
    
    if mark_resonator == 0:
        freq_fft, fft_x_out = FFT_Fup(int(Nt/2), x_out[int(Nt/2):Nt], dt, Freq_Vibr)
        freq_fft, fft_y_out = FFT_Fup(int(Nt/2), y_out[int(Nt/2):Nt], dt, Freq_Vibr)
    elif mark_resonator == 1:
        freq_fft, fft_x_out = Output_resonator_1D(Nt, x_out[0:Nt], x_ini[ind_out], m0[ind_out], Freq_Vibr, dt)
        freq_fft, fft_y_out = Output_resonator_1D(Nt, y_out[0:Nt], y_ini[ind_out], m0[ind_out], Freq_Vibr, dt)
       
        
    return freq_fft, fft_in1, fft_in2, fft_x_out, fft_y_out, np.mean(cont), nt_rec, Ek_now, Ep_now, cont_now

def MD_VibrSP_ConstV_Yfixed_DiffK2(k_list, k_type, B, Nt, N, x_ini, y_ini, D0, m0, L, Freq_Vibr, Amp_Vibr1, ind_in1, Amp_Vibr2, ind_in2, ind_out):
    
    Lx = L[0]
    Ly = L[1]

    mark_vibrY = 0
    mark_resonator = 0
    dt = D0[0]/40
    Nt = int(Nt)
    
    nt_rec = np.linspace(0, Nt, int(Nt/5e4)+1)    
    
    nt_rec = nt_rec.astype(int)
    
    Ep = np.zeros(Nt)
    Ek = np.zeros(Nt)   
    cont = np.zeros(Nt)
    p = np.zeros(Nt)
    F_tot = np.zeros(Nt)
    
    x_out = np.zeros(Nt)
    y_out = np.zeros(Nt)
    
    if mark_vibrY == 0:
        x_in1 = Amp_Vibr1*np.sin(Freq_Vibr*dt*np.arange(Nt))+x_ini[ind_in1]
        x_in2 = Amp_Vibr2*np.sin(Freq_Vibr*dt*np.arange(Nt))+x_ini[ind_in2]
    elif mark_vibrY == 1:
        y_in1 = Amp_Vibr1*np.sin(Freq_Vibr*dt*np.arange(Nt))+y_ini[ind_in1]
        y_in2 = Amp_Vibr2*np.sin(Freq_Vibr*dt*np.arange(Nt))+y_ini[ind_in2]
    
    vx = np.zeros(N)
    vy = np.zeros(N)    
    
    ax_old = np.zeros(N)
    ay_old = np.zeros(N)
    
    x = np.array(x_ini)
    y = np.array(y_ini)
    
    x_save = np.array(x_ini)
    y_save = np.array(y_ini)

    Fx = np.zeros(N)
    Fy = np.zeros(N)
    VL_list = np.zeros((N * 10, 2), dtype=int) 
    VL_counter = 0
    VL_list, VL_counter, x_save, y_save = VL_YFixed_ConstV(N, x, y, D0, Lx, VL_list, VL_counter, x_save, y_save, 1)
    Fx, Fy, Fup_now, Fbot_now, Ep_now, cont_now, p_now, cont_up = force_YFixed_DiffK_VL(Fx, Fy, N, x, y, D0, Lx, 0, Ly, k_list, k_type, VL_list, VL_counter)


    for nt in np.arange(Nt):
        x = x+vx*dt+ax_old*dt**2/2  # first step in Verlet integration
        y = y+vy*dt+ay_old*dt**2/2
        
        if mark_vibrY == 0:
            x[ind_in1] = x_in1[nt]
            y[ind_in1] = y_ini[ind_in1]
            x[ind_in2] = x_in2[nt]
            y[ind_in2] = y_ini[ind_in2]
        elif mark_vibrY == 1:
            x[ind_in1] = x_ini[ind_in1]
            y[ind_in1] = y_in1[nt]
            x[ind_in2] = x_ini[ind_in2]
            y[ind_in2] = y_in2[nt]
        
        Fx = np.zeros(N)
        Fy = np.zeros(N)
        VL_list, VL_counter, x_save, y_save = VL_YFixed_ConstV(N, x, y, D0, Lx, VL_list, VL_counter, x_save, y_save, 0)
        Fx, Fy, Fup_now, Fbot_now, Ep_now, cont_now, p_now, cont_up = force_YFixed_DiffK_VL(Fx, Fy, N, x, y, D0, Lx, 0, Ly, k_list, k_type, VL_list, VL_counter)
        
        Ep[nt] = Ep_now
        cont[nt] = cont_now
        p[nt] = p_now
        
        Fx_all = Fx - B*vx
        Fy_all = Fy - B*vy

        x_out[nt] = x[ind_out]
        y_out[nt] = y[ind_out]
        
        ax = np.divide(Fx_all, m0)
        ay = np.divide(Fy_all, m0)
        vx = vx+(ax_old+ax)*dt/2  # second step in Verlet integration
        vy = vy+(ay_old+ay)*dt/2        
        ax_old = ax
        ay_old = ay
        Ek[nt] = sum(0.5*np.multiply(m0,np.multiply(vx, vx)+np.multiply(vy, vy)))
    
    
    
    Ek_now = []
    Ep_now = []
    cont_now = []
    for ii in np.arange(len(nt_rec)-1):
        Ek_now.append(np.mean(Ek[nt_rec[ii]:nt_rec[ii+1]]))
        Ep_now.append(np.mean(Ep[nt_rec[ii]:nt_rec[ii+1]]))
        cont_now.append(np.mean(cont[nt_rec[ii]:nt_rec[ii+1]]))
    nt_rec = (nt_rec[1:] + nt_rec[:-1]) / 2

    #t_end = time.time()
    #print ("time=%.3e" %(t_end-t_start))
    CB_ratio = min(cont)/max(cont)
    #print ("freq=%f, cont_min/cont_max=%f, Ek_mean=%.3e, Ep_mean=%.3e\n" %(Freq_Vibr, CB_ratio, np.mean(Ek), np.mean(Ep)))
    
    if mark_vibrY == 0:
        freq_fft, fft_in1 = FFT_Fup(int(Nt/2), x_in1[int(Nt/2):Nt], dt, Freq_Vibr)
        freq_fft, fft_in2 = FFT_Fup(int(Nt/2), x_in2[int(Nt/2):Nt], dt, Freq_Vibr)        
    elif mark_vibrY == 1:
        freq_fft, fft_in1 = FFT_Fup(int(Nt/2), y_in1[int(Nt/2):Nt], dt, Freq_Vibr)
        freq_fft, fft_in2 = FFT_Fup(int(Nt/2), y_in2[int(Nt/2):Nt], dt, Freq_Vibr)
    
    if mark_resonator == 0:
        freq_fft, fft_x_out = FFT_Fup(int(Nt/2), x_out[int(Nt/2):Nt], dt, Freq_Vibr)
        freq_fft, fft_y_out = FFT_Fup(int(Nt/2), y_out[int(Nt/2):Nt], dt, Freq_Vibr)
    elif mark_resonator == 1:
        freq_fft, fft_x_out = Output_resonator_1D(Nt, x_out[0:Nt], x_ini[ind_out], m0[ind_out], Freq_Vibr, dt)
        freq_fft, fft_y_out = Output_resonator_1D(Nt, y_out[0:Nt], y_ini[ind_out], m0[ind_out], Freq_Vibr, dt)
       
        
    return x_in1-x_ini[ind_in1], x_in2-x_ini[ind_in2], x_out-x_ini[ind_out]