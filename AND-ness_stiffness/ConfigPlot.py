"""

Simulator by Qikai Wu
https://journals.aps.org/pre/abstract/10.1103/PhysRevE.99.062901

Modified by Atoosa Parsa 

"""

import numpy as np

def ConfigPlot_DiffSize(N, x, y, D, L, mark_print):
    
    import matplotlib.pyplot as plt
    from matplotlib.patches import Ellipse
    Dmin = min(D)
    fig, ax = plt.subplots(subplot_kw={'aspect': 'equal'})
    
    ells = []
    D_all = []
    for i in range(N):
        x_now = x[i]%L[0]
        y_now = y[i]%L[1]
        for k in range(-1, 2):
            for l in range(-1, 2):                        
                e = Ellipse((x_now+k*L[0], y_now+l*L[1]), D[i],D[i],0)
                ells.append(e)
                D_all.append(D[i])
                    
    i = 0
    for e in ells:
        ax.add_artist(e)
        e.set_clip_box(ax.bbox)
        e.set_alpha(0.3)
        if D_all[i] > Dmin:
            e.set_facecolor('C1')
        else:
            e.set_facecolor('C0')
        i += 1
                
    ax.set_xlim(0, L[0])
    ax.set_ylim(0, L[1])
    plt.show() 
    if mark_print == 1:
        fig.savefig('/Users/Hightoutou/Desktop/fig.png', dpi = 300)
        
        
        
        
def ConfigPlot_DiffMass(N, x, y, D, L, m, mark_print, hn):
    
    import matplotlib.pyplot as plt
    from matplotlib.patches import Ellipse
    m_min = min(m)
    m_max = max(m)
    fig, ax = plt.subplots(subplot_kw={'aspect': 'equal'})
    ells = []
    m_all = []
    for i in range(N):
        x_now = x[i]%L[0]
        y_now = y[i]%L[1]
        for k in range(-1, 2):
            for l in range(-1, 2):                        
                e = Ellipse((x_now+k*L[0], y_now+l*L[1]), D[i],D[i],0)
                ells.append(e)
                m_all.append(m[i])

    i = 0
    for e in ells:
        ax.add_artist(e)
        e.set_clip_box(ax.bbox)
        e.set_facecolor('C0')
        
        e.set_alpha(0.2+(m_all[i]-m_min)/(m_max-m_min)*0.8)
        e.set_alpha(0.2+(m_all[i]-m_min)/(m_max-m_min)*0.3)
 
        i += 1
                
    ax.set_xlim(0, L[0])
    ax.set_ylim(0, L[1])

    plt.show() 
    if mark_print == 1:
        fig.savefig(hn, dpi = 300)
    return fig

def ConfigPlot_DiffMass2(N, x, y, D, L, m, mark_print, hn, in1, in2, out):
    
    import matplotlib.pyplot as plt
    from matplotlib.patches import Ellipse
    m_min = min(m)
    m_max = max(m)
    fig, ax = plt.subplots(subplot_kw={'aspect': 'equal'})
    ells = []
    m_all = []
    for i in range(N):
        x_now = x[i]%L[0]
        y_now = y[i]%L[1]
        for k in range(-1, 2):
            for l in range(-1, 2):                        
                e = Ellipse((x_now+k*L[0], y_now+l*L[1]), D[i],D[i],0)
                if i==in1:
                    e.set_edgecolor((0, 1, 0))
                    e.set_linewidth(4)
                elif i==in2:
                    e.set_edgecolor((0, 0, 1))
                    e.set_linewidth(4)
                elif i==out:
                    e.set_edgecolor((1, 0, 0))
                    e.set_linewidth(4)
                ells.append(e)
                m_all.append(m[i])

    i = 0
    for e in ells:
        ax.add_artist(e)
        e.set_clip_box(ax.bbox)
        e.set_facecolor('C0')
        
        e.set_alpha(0.2+(m_all[i]-m_min)/(m_max-m_min)*0.8)
        e.set_alpha(0.2+(m_all[i]-m_min)/(m_max-m_min)*0.3)
 
        i += 1
                
    ax.set_xlim(0, L[0])
    ax.set_ylim(0, L[1])

    plt.show() 
    if mark_print == 1:
        fig.savefig(hn, dpi = 300)
    return fig

def ConfigPlot_DiffStiffness(N, x, y, D, L, m, mark_print, hn):
    
    import matplotlib.pyplot as plt
    from matplotlib.patches import Ellipse
    m_min = min(m)
    m_max = max(m)
    fig, ax = plt.subplots(subplot_kw={'aspect': 'equal'})
    ells = []
    m_all = []
    for i in range(N):
        x_now = x[i]%L[0]
        y_now = y[i]%L[1]
        for k in range(-1, 2):
            for l in range(-1, 2):                        
                e = Ellipse((x_now+k*L[0], y_now+l*L[1]), D[i],D[i],0)
                ells.append(e)
                m_all.append(m[i])

    i = 0
    for e in ells:
        ax.add_artist(e)
        e.set_clip_box(ax.bbox)
        e.set_facecolor('C2')
        
        e.set_alpha(0.2+(m_all[i]-m_min)/(m_max-m_min)*0.8)
        e.set_alpha(0.2+(m_all[i]-m_min)/(m_max-m_min)*0.3)
 
        i += 1
                
    ax.set_xlim(0, L[0])
    ax.set_ylim(0, L[1])

    plt.show() 
    if mark_print == 1:
        fig.savefig(hn, dpi = 300)
    return fig

def ConfigPlot_DiffStiffness2(N, x, y, D, L, m, mark_print, hn, in1, in2, out):
    
    import matplotlib.pyplot as plt
    from matplotlib.patches import Ellipse
    m_min = min(m)
    m_max = max(m)
    fig, ax = plt.subplots(subplot_kw={'aspect': 'equal'})
    ells = []
    m_all = []
    for i in range(N):
        x_now = x[i]%L[0]
        y_now = y[i]%L[1]
        for k in range(-1, 2):
            for l in range(-1, 2):                        
                e = Ellipse((x_now+k*L[0], y_now+l*L[1]), D[i],D[i],0)
                if i==in1:
                    e.set_edgecolor((0, 1, 0))
                    e.set_linewidth(4)
                elif i==in2:
                    e.set_edgecolor((0, 0, 1))
                    e.set_linewidth(4)
                elif i==out:
                    e.set_edgecolor((1, 0, 0))
                    e.set_linewidth(4)
                ells.append(e)
                m_all.append(m[i])

    i = 0
    for e in ells:
        ax.add_artist(e)
        e.set_clip_box(ax.bbox)
        e.set_facecolor('C2')
        
        e.set_alpha(0.2+(m_all[i]-m_min)/(m_max-m_min)*0.8)
        e.set_alpha(0.2+(m_all[i]-m_min)/(m_max-m_min)*0.3)
 
        i += 1
                
    ax.set_xlim(0, L[0])
    ax.set_ylim(0, L[1])

    plt.show() 
    if mark_print == 1:
        fig.savefig(hn, dpi = 300)
    return fig

def ConfigPlot_DiffStiffness3(N, x, y, D, L, m, mark_print, hn, in1, in2, out):
    
    import matplotlib.pyplot as plt
    from matplotlib.patches import Ellipse
    m_min = min(m)
    m_max = max(m)
    fig, ax = plt.subplots(subplot_kw={'aspect': 'equal'})
    ells = []
    m_all = []
    for i in range(N):
        x_now = x[i]%L[0]
        y_now = y[i]%L[1]
        for k in range(-1, 2):
            for l in range(-1, 2):                        
                e = Ellipse((x_now+k*L[0], y_now+l*L[1]), D[i],D[i],0)
                if i==in1:
                    e.set_edgecolor((0, 1, 0))
                    e.set_linewidth(4)
                    plt.scatter(x_now, y_now, marker='^', s=300, color=(0, 1, 0, 1))
                elif i==in2:
                    e.set_edgecolor((0, 0, 1))
                    e.set_linewidth(4)
                    plt.scatter(x_now, y_now, marker='s', s=300, color=(0, 0, 1, 1))
                elif i==out:
                    e.set_edgecolor((1, 0, 0))
                    e.set_linewidth(4)
                    plt.scatter(x_now, y_now, marker='*', s=300, color=(1, 0, 0, 1))
                ells.append(e)
                m_all.append(m[i])

    i = 0
    for e in ells:
        ax.add_artist(e)
        e.set_clip_box(ax.bbox)
        e.set_facecolor('C2')
        
        e.set_alpha(0.2+(m_all[i]-m_min)/(m_max-m_min)*0.8)
        e.set_alpha(0.2+(m_all[i]-m_min)/(m_max-m_min)*0.3)
 
        i += 1
                
    ax.set_xlim(0, L[0])
    ax.set_ylim(0, L[1])

    import matplotlib.lines as mlines
    red_star = mlines.Line2D([], [], color=(1, 0, 0), marker='*', linestyle='None',
                            markersize=15, label='Output')
    blue_square = mlines.Line2D([], [], color=(0, 0, 1), marker='s', linestyle='None',
                            markersize=15, label='Input 2')
    green_triangle = mlines.Line2D([], [], color=(0, 1, 0), marker='^', linestyle='None',
                            markersize=15, label='Input 1')

    plt.legend(handles=[red_star, green_triangle, blue_square], bbox_to_anchor=(1.315, 1), fontsize=14)

    plt.show() 
    if mark_print == 1:
        fig.savefig(hn, dpi = 300)
    return fig

def ConfigPlot_DiffMass_3D(N, x, y, z, D, L, m, mark_print):
    
    import numpy as np
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    
    m_min = min(m)
    m_max = max(m)
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_aspect('equal')
    
    sphes = []
    m_all = []
    for i in range(int(N/2)):
        x_now = x[i]%L[0]
        y_now = y[i]%L[1]
        
        z_now = z[i]%L[2]
        r_now = 0.5*D[i]
        #alpha_now = 0.2+(m_all[i]-m_min)/(m_max-m_min)*0.3
        alpha_now = 0.3
        
        pos1 = 0
        pos2 = 1        
        for j in range(pos1, pos2):
            for k in range(pos1, pos2):
                for l in range(pos1, pos2):
                    
                    u = np.linspace(0, 2 * np.pi, 100)
                    v = np.linspace(0, np.pi, 100)

                    x_plot = x_now+j*L[0]+r_now * np.outer(np.cos(u), np.sin(v))
                    y_plot = y_now+k*L[1]+r_now * np.outer(np.sin(u), np.sin(v))
                    z_plot = z_now+l*L[2]+r_now * np.outer(np.ones(np.size(u)), np.cos(v))
                    ymin = y_plot[y_plot>0].min()
                    ymax = y_plot[y_plot>0].max()
                    print (i, ymin, ymax)
                    ax.plot_surface(x_plot,y_plot,z_plot,rstride=4,cstride=4, color='C0',linewidth=0,alpha=alpha_now)
                    #sphes.append(e)
                    #m_all.append(m[i])

#    i = 0
#    for e in sphes:
#        ax.add_artist(e)
#        e.set_clip_box(ax.bbox)
#        e.set_facecolor('C0')
        
#        e.set_alpha(0.2+(m_all[i]-m_min)/(m_max-m_min)*0.8)
        #e.set_alpha(0.2+(m_all[i]-m_min)/(m_max-m_min)*0.3)
 
#        i += 1
                
    ax.set_xlim(0, L[0])
    ax.set_ylim(0, L[1])
    ax.set_zlim(0, L[2])

    plt.show() 
    if mark_print == 1:
        fig.savefig('/Users/Hightoutou/Desktop/fig.png', dpi = 300)
        
def ConfigPlot_YFixed_rec(N, x, y, D, Lx, y_top, y_bot, m, mark_order):
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.patches import Ellipse, Rectangle
    
    mark_print = 0
    m_min = min(m)
    m_max = max(m)
    if m_min == m_max:
        m_max *= 1.001
    fig, ax = plt.subplots(subplot_kw={'aspect': 'equal'})
    ells = []
    m_all = []
    for i in range(N):
        x_now = x[i]%Lx
        y_now = y[i]
        if mark_order==1:
            plt.text(x_now, y_now, str(i))
        for k in range(-1, 2):                      
            e = Ellipse((x_now+k*Lx, y_now), D[i],D[i],0)
            ells.append(e)
            m_all.append(m[i])

    i = 0
    for e in ells:
        ax.add_artist(e)
        e.set_clip_box(ax.bbox)
        e.set_facecolor('C0')        
        #e.set_alpha(0.2+(m_all[i]-m_min)/(m_max-m_min)*0.8)
        e.set_alpha(0.3+(m_all[i]-m_min)/(m_max-m_min)*0.3)
        i += 1
    
    #rect = Rectangle([0, y_top], Lx, 0.2*D[0], color='C0')
    #ax.add_patch(rect)
    
    for nn in np.arange(N):
        x1 = x[nn]%Lx
        d_up = y_top-y[nn]
        d_bot = y[nn]-y_bot
        r_now = 0.5*D[nn]
        
        if d_up<r_now:
            ax.plot([x1, x1], [y[nn], y[nn]+r_now], '-', color='w')
        if d_bot<r_now:
            ax.plot([x1, x1], [y[nn], y[nn]-r_now], '-', color='w')
            
        for mm in np.arange(nn+1, N):
            dy = y[mm]-y[nn]
            Dmn = 0.5*(D[mm]+D[nn])
            if abs(dy) < Dmn:
                x2 = x[mm]%Lx
                if x2>x1:
                    xl = x1
                    xr = x2
                    yl = y[nn]
                    yr = y[mm]
                else:
                    xl = x2
                    xr = x1
                    yl = y[mm]
                    yr = y[nn]
                dx0 = xr-xl
                dx = dx0-round(dx0/Lx)*Lx
                dmn = np.sqrt(dx**2+dy**2)
                if dmn < Dmn:
                    if dx0<Dmn:
                        ax.plot([xl, xr], [yl, yr], '-', color='w')
                    else:
                        ax.plot([xl, xr-Lx], [yl, yr], '-', color='w')
                        ax.plot([xl+Lx, xr], [yl, yr], '-', color='w')
                    
    
                
    ax.set_xlim(0, Lx)
    ax.set_ylim(y_bot, y_top)

    plt.show() 
    if mark_print == 1:
        fig.savefig('/Users/Hightoutou/Desktop/plot_test/fig'+str(int(ind_nt+1e4))+'.png', dpi = 150)
    
def ConfigPlot_DiffMass_SP(N, x, y, D, L, m, mark_print, ind_in, ind_out, ind_fix):
    
    import matplotlib.pyplot as plt
    from matplotlib.patches import Ellipse
    m_min = min(m)
    m_max = max(m)
    fig, ax = plt.subplots(subplot_kw={'aspect': 'equal'})
    ells = []
    m_all = []
    width = 2
    for i in range(N):
        x_now = x[i]%L[0]
        y_now = y[i]%L[1]
        for k in range(-1, 2):
            for l in range(-1, 2):                        
                e = Ellipse((x_now+k*L[0], y_now+l*L[1]), D[i],D[i],0)
                ells.append(e)
                m_all.append(m[i])
                
                
                ax.add_artist(e)
                e.set_clip_box(ax.bbox)
                e.set_facecolor('C0')                
                #e.set_alpha(0.2+(m[i]-m_min)/(m_max-m_min)*0.8)
                e.set_alpha(0.2+(m[i]-m_min)/(m_max-m_min)*0.3)
                if i == ind_in:
                    e.set_edgecolor('r')
                    e.set_linewidth(width)
                if i == ind_out:
                    e.set_edgecolor('b')
                    e.set_linewidth(width)
                if i == ind_fix:
                    e.set_edgecolor('k')
                    e.set_linewidth(width)
                       
                    
                
                
    ax.set_xlim(0, L[0])
    ax.set_ylim(0, L[1])

    plt.show() 
    if mark_print == 1:
        fig.savefig('/Users/Hightoutou/Desktop/fig.png', dpi = 300)
    
        

def ConfigPlot_DiffMass_FixLx(N, x, y, D, L, m, mark_print, ind_wall):
    
    import matplotlib.pyplot as plt
    from matplotlib.patches import Ellipse
    m_min = min(m)
    m_max = max(m)
    fig, ax = plt.subplots(subplot_kw={'aspect': 'equal'})
    ells = []
    m_all = []
    width = 2
    for i in range(N):
        x_now = x[i]
        y_now = y[i]%L[1]
        for l in range(-1, 2):                        
            e = Ellipse((x_now, y_now+l*L[1]), D[i],D[i],0)
            ells.append(e)
            m_all.append(m[i])
            if ind_wall[i] > 0:
                e.set_edgecolor('k')
                e.set_linewidth(width)

    i = 0
    for e in ells:
        ax.add_artist(e)
        e.set_clip_box(ax.bbox)
        e.set_facecolor('C0')
        
        e.set_alpha(0.2+(m_all[i]-m_min)/(m_max-m_min)*0.8)
        e.set_alpha(0.2+(m_all[i]-m_min)/(m_max-m_min)*0.3)
 
        i += 1
                
    ax.set_xlim(0, L[0])
    ax.set_ylim(0, L[1])

    plt.show() 
    if mark_print == 1:
        fig.savefig('/Users/Hightoutou/Desktop/fig.png', dpi = 300)
        
        
def ConfigPlot_DiffMass_SP_rec(N, x, y, D, L, m, mark_print, ind_in, ind_out, ind_fix):
    
    import matplotlib.pyplot as plt
    from matplotlib.patches import Ellipse
    m_min = min(m)
    m_max = max(m)
    fig, ax = plt.subplots(subplot_kw={'aspect': 'equal'})
    ells = []
    m_all = []
    width = 2
    for i in range(N):
        x_now = x[i]%L[0]
        y_now = y[i]%L[1]
        for k in range(-1, 2):
            for l in range(-1, 2):                        
                e = Ellipse((x_now+k*L[0], y_now+l*L[1]), D[i],D[i],0)
                ells.append(e)
                m_all.append(m[i])
                
                
                ax.add_artist(e)
                e.set_clip_box(ax.bbox)
                e.set_facecolor('C0')                
                #e.set_alpha(0.2+(m[i]-m_min)/(m_max-m_min)*0.8)
                e.set_alpha(0.2+(m[i]-m_min)/(m_max-m_min)*0.3)
                if i == ind_in:
                    e.set_edgecolor('r')
                    e.set_linewidth(width)
                if i == ind_out:
                    e.set_edgecolor('b')
                    e.set_linewidth(width)
                if i == ind_fix:
                    e.set_edgecolor('k')
                    e.set_linewidth(width)
    Lx = L[0]
    Ly = L[1]
    for nn in np.arange(N):
        x1 = x[nn]%Lx
        y1 = y[nn]%Ly
        for mm in np.arange(nn+1, N):
            x2 = x[mm]%Lx
            y2 = y[mm]%Ly            
            if x2>x1:
                xl = x1
                xr = x2
                yl = y1
                yr = y2
            else:
                xl = x2
                xr = x1
                yl = y2
                yr = y1            
            dx0 = xr-xl
            dx = dx0-round(dx0/Lx)*Lx
            
            if y2>y1:
                xd = x1
                xu = x2
                yd = y1
                yu = y2
            else:
                xd = x2
                xu = x1
                yd = y2
                yu = y1
            
            dy0 = yu-yd
            dy = dy0-round(dy0/Ly)*Ly
                        
            Dmn = 0.5*(D[mm]+D[nn])                                
            dmn = np.sqrt(dx**2+dy**2)
            if dmn < Dmn:
                if dx0<Dmn and dy0<Dmn:
                    ax.plot([xl, xr], [yl, yr], '-', color='w')
                else:
                    if dx0>Dmn and dy0>Dmn:
                        if yr>yl:
                            ax.plot([xl, xr-Lx], [yl, yr-Ly], '-', color='w')
                            ax.plot([xl+Lx, xr], [yl+Ly, yr], '-', color='w')
                        else:
                            ax.plot([xl, xr-Lx], [yl, yr+Ly], '-', color='w')
                            ax.plot([xl+Lx, xr], [yl-Ly, yr], '-', color='w')
                    else:
                        if dx0>Dmn:
                            ax.plot([xl, xr-Lx], [yl, yr], '-', color='w')
                            ax.plot([xl+Lx, xr], [yl, yr], '-', color='w')
                        if dy0>Dmn:
                            ax.plot([xd, xu], [yd, yu-Ly], '-', color='w')
                            ax.plot([xd, xu], [yd+Ly, yu], '-', color='w')

    ax.set_xlim(0, L[0])
    ax.set_ylim(0, L[1])

    plt.show() 
    if mark_print == 1:
        fig.savefig('/Users/Hightoutou/Desktop/fig.png', dpi = 300)  
    
    return fig

def ConfigPlot_EigenMode_DiffMass(N, x, y, D, L, m, em, mark_print, hn):
    
    import matplotlib.pyplot as plt
    from matplotlib.patches import Ellipse
    m_min = min(m)
    m_max = max(m)
    if m_min == m_max:
        m_max *= 1.001   
        
    fig, ax = plt.subplots(subplot_kw={'aspect': 'equal'})
    ells = []
    m_all = []
    for i in range(N):
        x_now = x[i]%L[0]
        y_now = y[i]%L[1]
        for k in range(-1, 2):
            for l in range(-1, 2):                        
                e = Ellipse((x_now+k*L[0], y_now+l*L[1]), D[i],D[i],0)
                ells.append(e)
                m_all.append(m[i])

    i = 0
    for e in ells:
        ax.add_artist(e)
        e.set_clip_box(ax.bbox)
        e.set_facecolor('C0')
        
        e.set_alpha(0.2+(m_all[i]-m_min)/(m_max-m_min)*0.8)
        e.set_alpha(0.2+(m_all[i]-m_min)/(m_max-m_min)*0.3)
 
        i += 1
                
    r_now = D[0]*0.5
    dr = np.zeros(N)    
    for i in range(N):
        dr[i] = np.sqrt(em[2*i]**2+em[2*i+1]**2)
    dr_max = max(dr)
    for i in range(N):
        ratio = dr[i]/dr_max*r_now/dr_max
        plt.arrow(x[i], y[i],em[2*i]*ratio, em[2*i+1]*ratio, head_width=0.005)
    
    
    ax.set_xlim(0, L[0])
    ax.set_ylim(0, L[1])
    
    

    plt.show() 
    if mark_print == 1:
        fig.savefig(hn, dpi = 300)
    return fig

def ConfigPlot_YFixed_SelfAssembly(N, Nl, x, y, theta, n, d1, d2, Lx, y_top, y_bot):
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.patches import Ellipse, Rectangle
    
    mark_order = 0
    
    fig, ax = plt.subplots(subplot_kw={'aspect': 'equal'})
    ells = []
    alpha_all = []
    alpha1 = 0.6
    alpha2 = 0.3
    for i in range(N):
        x_now = x[i]%Lx
        y_now = y[i]
        if mark_order==1:
            plt.text(x_now, y_now, str(i))
        alpha = alpha1 if i < Nl else alpha2
        for k in range(-1, 2):                      
            e = Ellipse((x_now+k*Lx, y_now), d1,d1,0)
            ells.append(e)
            alpha_all.append(alpha)
            if i >= Nl:
                for ind in range(n):
                    x_i = x_now+k*Lx+0.5*(d1+d2)*np.cos(theta[i]+ind*2*np.pi/n)
                    y_i = y_now+0.5*(d1+d2)*np.sin(theta[i]+ind*2*np.pi/n)
                    e = Ellipse((x_i, y_i), d2,d2,0)
                    ells.append(e)
                    alpha_all.append(alpha)

    i = 0
    for e in ells:
        ax.add_artist(e)
        e.set_clip_box(ax.bbox)
        e.set_facecolor('C0')        
        #e.set_alpha(0.2+(m_all[i]-m_min)/(m_max-m_min)*0.8)
        e.set_alpha(alpha_all[i])
        i += 1
                
    ax.set_xlim(0, Lx)
    ax.set_ylim(y_bot, y_top)

    plt.show() 
    
def ConfigPlot_YFixed_SelfAssembly_BumpyBd(N, n_col, Nl, x, y, theta, n, d0, d1, d2, Lx, y_top, y_bot):
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.patches import Ellipse, Rectangle
    
    mark_order = 0
    
    fig, ax = plt.subplots(subplot_kw={'aspect': 'equal'})
    ells = []
    alpha_all = []
    alpha1 = 0.6
    alpha2 = 0.3
    for i in range(n_col+1):
        x_now = i*d0
        e1 = Ellipse((x_now, y_bot), d0,d0,0)
        e2 = Ellipse((x_now, y_top), d0,d0,0)
        ells.append(e1)
        alpha_all.append(alpha1)
        ells.append(e2)
        alpha_all.append(alpha1)
        
    for i in range(N):
        x_now = x[i]%Lx
        y_now = y[i]
        if mark_order==1:
            plt.text(x_now, y_now, str(i))
        alpha = alpha1 if i < Nl else alpha2
        for k in range(-1, 2):                      
            e = Ellipse((x_now+k*Lx, y_now), d1,d1,0)
            ells.append(e)
            alpha_all.append(alpha)
            if i >= Nl:
                for ind in range(n):
                    x_i = x_now+k*Lx+0.5*(d1+d2)*np.cos(theta[i]+ind*2*np.pi/n)
                    y_i = y_now+0.5*(d1+d2)*np.sin(theta[i]+ind*2*np.pi/n)
                    e = Ellipse((x_i, y_i), d2,d2,0)
                    ells.append(e)
                    alpha_all.append(alpha)

    i = 0
    for e in ells:
        ax.add_artist(e)
        e.set_clip_box(ax.bbox)
        e.set_facecolor('C0')        
        #e.set_alpha(0.2+(m_all[i]-m_min)/(m_max-m_min)*0.8)
        e.set_alpha(alpha_all[i])
        i += 1
                
    ax.set_xlim(0, Lx)
    ax.set_ylim(y_bot, y_top)

    plt.show() 