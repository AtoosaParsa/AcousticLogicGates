"""

Simulator by Qikai Wu
https://journals.aps.org/pre/abstract/10.1103/PhysRevE.99.062901

Modified by Atoosa Parsa 

"""
import matplotlib.pyplot as plt

def Line_single(xdata, ydata, line_spec, xlabel, ylabel, mark_print, fn = '', xscale='linear', yscale='linear'):
    fig, ax1 = plt.subplots()
    fig.set_size_inches(3.5,3.5*3/4)
    pos1 = ax1.get_position()
    pos2 = [pos1.x0 + 0.12, pos1.y0 + 0.05,  pos1.width-0.1, pos1.height] 
    ax1.set_position(pos2)
    #ax1.tick_params(labelsize=10)
    for axis in ['top','bottom','left','right']:
        ax1.spines[axis].set_linewidth(2)       
    plt.ylabel(ylabel, fontsize=12)
    plt.xlabel(xlabel, fontsize=12)
    plt.xscale(xscale), plt.yscale(yscale)
    plt.style.use('default')
    ax1.plot(xdata, ydata, line_spec)
    if mark_print == 1:
        fig.savefig(fn, dpi = 300)
    fig.show()
        
def Line_multi(xdata, ydata, line_spec, xlabel, ylabel, xscale='linear', yscale='linear'):
    fig, ax1 = plt.subplots()
    fig.set_size_inches(3.5,3.5*3/4)
    ax1.tick_params(labelsize=10)
    for axis in ['top','bottom','left','right']:
        ax1.spines[axis].set_linewidth(2)       
    ax1.set_ylabel(ylabel, fontsize=12)
    plt.xlabel(xlabel, fontsize=12)
    plt.xscale(xscale), plt.yscale(yscale)
    plt.style.use('default')
    for ii in range(len(xdata)):
        ax1.plot(xdata[ii], ydata[ii], line_spec[ii])
    plt.show() 
    
def Line_yy(xdata, ydata, line_spec, xlabel, ylabel, xscale='linear', yscale='linear'):
    fig, ax1 = plt.subplots()
    fig.set_size_inches(3.5,3.5*3/4)
    ax1.tick_params(labelsize=10)
    for axis in ['top','bottom','left','right']:
        ax1.spines[axis].set_linewidth(2)       
    ax1.set_ylabel(ylabel[0], fontsize=12)
    plt.xlabel(xlabel, fontsize=12)
    plt.xscale(xscale), plt.yscale(yscale)
    plt.style.use('default')
    ax1.plot(xdata[0], ydata[0], line_spec[0])
    ax2 = ax1.twinx()
    ax2.set_ylabel(ylabel[1], fontsize=12)
    ax2.plot(xdata[1], ydata[1], line_spec[1])
    plt.show() 
    