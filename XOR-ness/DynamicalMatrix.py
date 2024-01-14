"""

Simulator by Qikai Wu
https://journals.aps.org/pre/abstract/10.1103/PhysRevE.99.062901

Modified by Atoosa Parsa 

"""
def DM_mass(N, x0, y0, D0, m0, L):
    
    import numpy as np
    
    Lx = L[0]
    Ly = L[1]
    M = np.zeros((2*N, 2*N))
    contactNum = 0

    for i in range(N):
        for j in range(i):
            dij = 0.5*(D0[i]+D0[j])
            dijsq = dij**2
            dx = x0[i]-x0[j]
            dx = dx-round(dx/Lx)*Lx
            dy = y0[i]-y0[j]
            dy = dy-round(dy/Ly)*Ly
            rijsq = dx**2+dy**2
            if rijsq<dijsq:
                contactNum += 1  
                rijmat = np.array([[dx*dx, dx*dy], [dx*dy, dy*dy]])
                rij = np.sqrt(rijsq)
                Mij1 = -rijmat/rijsq/dijsq
                Mij2 = -(1-rij/dij)*(rijmat/rijsq-[[1,0],[0,1]])/rij/dij
                Mij = Mij1+Mij2
                M[2*i:2*i+2,2*j:2*j+2] = Mij
                M[2*j:2*j+2,2*i:2*i+2] = Mij
                M[2*i:2*i+2,2*i:2*i+2] = M[2*i:2*i+2,2*i:2*i+2] - Mij
                M[2*j:2*j+2,2*j:2*j+2] = M[2*j:2*j+2,2*j:2*j+2] - Mij

    m_sqrt = np.zeros((2*N, 2*N))
    m_inv = np.zeros((2*N, 2*N))
    for i in range(N):
        m_sqrt[2*i, 2*i] = 1/np.sqrt(m0[i])
        m_sqrt[2*i+1, 2*i+1] = 1/np.sqrt(m0[i])
        m_inv[2*i, 2*i] = 1/m0[i]
        m_inv[2*i+1, 2*i+1] = 1/m0[i]

    #M = m_sqrt.dot(M).dot(m_sqrt)
    M = m_inv.dot(M)

    w,v = np.linalg.eig(M)
    
    return w,v

def DM_mass_3D(N, x0, y0, z0, D0, m0, L):
    
    import numpy as np
    
    Lx = L[0]
    Ly = L[1]
    Lz = L[2]
    M = np.zeros((3*N, 3*N))
    contactNum = 0

    for i in range(N):
        for j in range(i):
            dij = 0.5*(D0[i]+D0[j])
            dijsq = dij**2
            dx = x0[i]-x0[j]
            dx = dx-round(dx/Lx)*Lx
            dy = y0[i]-y0[j]
            dy = dy-round(dy/Ly)*Ly
            dz = z0[i]-z0[j]
            dz = dz-round(dz/Lz)*Lz
            rijsq = dx**2+dy**2+dz**2
            if rijsq<dijsq:
                contactNum += 1  
                rijmat = np.array([[dx*dx, dx*dy, dx*dz], [dy*dx, dy*dy, dy*dz], [dz*dx, dz*dy, dz*dz]])
                rij = np.sqrt(rijsq)
                Mij1 = -rijmat/rijsq/dijsq
                Mij2 = -(1-rij/dij)*(rijmat/rijsq-[[1,0,0],[0,1,0],[0,0,1]])/rij/dij
                Mij = Mij1+Mij2
                M[3*i:3*i+3,3*j:3*j+3] = Mij
                M[3*j:3*j+3,3*i:3*i+3] = Mij
                M[3*i:3*i+3,3*i:3*i+3] = M[3*i:3*i+3,3*i:3*i+3] - Mij
                M[3*j:3*j+3,3*j:3*j+3] = M[3*j:3*j+3,3*j:3*j+3] - Mij

    m_sqrt = np.zeros((3*N, 3*N))
    m_inv = np.zeros((3*N, 3*N))
    for i in range(N):
        m_sqrt[3*i, 3*i] = 1/np.sqrt(m0[i])
        m_sqrt[3*i+1, 3*i+1] = 1/np.sqrt(m0[i])
        m_sqrt[3*i+2, 3*i+2] = 1/np.sqrt(m0[i])
        m_inv[3*i, 3*i] = 1/m0[i]
        m_inv[3*i+1, 3*i+1] = 1/m0[i]
        m_inv[3*i+2, 3*i+2] = 1/m0[i]

    #M = m_sqrt.dot(M).dot(m_sqrt)
    M = m_inv.dot(M)

    w,v = np.linalg.eig(M)
    
    return w,v

def DM_mass_Yfixed(N, x0, y0, D0, m0, Lx, y_bot, y_top, k):
    
    import numpy as np
    
    M = np.zeros((2*N, 2*N))
    contactNum = 0

    for i in range(N):
        r_now = 0.5*D0[i]
        if y0[i]-y_bot<r_now or y_top-y0[i]<r_now:
            M[2*i+1, 2*i+1] = M[2*i+1, 2*i+1]+1/r_now/r_now
        for j in range(i):
            dij = 0.5*(D0[i]+D0[j])
            dijsq = dij**2
            dx = x0[i]-x0[j]
            dx = dx-round(dx/Lx)*Lx
            dy = y0[i]-y0[j]
            rijsq = dx**2+dy**2
            if rijsq<dijsq:
                contactNum += 1  
                rijmat = np.array([[dx*dx, dx*dy], [dx*dy, dy*dy]])
                rij = np.sqrt(rijsq)
                Mij1 = -rijmat/rijsq/dijsq
                Mij2 = -(1-rij/dij)*(rijmat/rijsq-[[1,0],[0,1]])/rij/dij
                Mij = Mij1+Mij2
                M[2*i:2*i+2,2*j:2*j+2] = Mij
                M[2*j:2*j+2,2*i:2*i+2] = Mij
                M[2*i:2*i+2,2*i:2*i+2] = M[2*i:2*i+2,2*i:2*i+2] - Mij
                M[2*j:2*j+2,2*j:2*j+2] = M[2*j:2*j+2,2*j:2*j+2] - Mij

    M = k*M
    m_sqrt = np.zeros((2*N, 2*N))
    m_inv = np.zeros((2*N, 2*N))
    for i in range(N):
        m_sqrt[2*i, 2*i] = 1/np.sqrt(m0[i])
        m_sqrt[2*i+1, 2*i+1] = 1/np.sqrt(m0[i])
        m_inv[2*i, 2*i] = 1/m0[i]
        m_inv[2*i+1, 2*i+1] = 1/m0[i]

    #M = m_sqrt.dot(M).dot(m_sqrt)
    M = m_inv.dot(M)

    w,v = np.linalg.eig(M)
    
    return w,v

def DM_mass_Xfixed(N, x0, y0, D0, m0, Ly):
    
    import numpy as np
    
    M = np.zeros((2*N, 2*N))
    contactNum = 0

    for i in range(N):
        for j in range(i):
            dij = 0.5*(D0[i]+D0[j])
            dijsq = dij**2
            dx = x0[i]-x0[j]            
            dy = y0[i]-y0[j]
            dy = dy-round(dy/Ly)*Ly
            rijsq = dx**2+dy**2
            if rijsq<dijsq:
                contactNum += 1  
                rijmat = np.array([[dx*dx, dx*dy], [dx*dy, dy*dy]])
                rij = np.sqrt(rijsq)
                Mij1 = -rijmat/rijsq/dijsq
                Mij2 = -(1-rij/dij)*(rijmat/rijsq-[[1,0],[0,1]])/rij/dij
                Mij = Mij1+Mij2
                M[2*i:2*i+2,2*j:2*j+2] = Mij
                M[2*j:2*j+2,2*i:2*i+2] = Mij
                M[2*i:2*i+2,2*i:2*i+2] = M[2*i:2*i+2,2*i:2*i+2] - Mij
                M[2*j:2*j+2,2*j:2*j+2] = M[2*j:2*j+2,2*j:2*j+2] - Mij

    m_sqrt = np.zeros((2*N, 2*N))
    m_inv = np.zeros((2*N, 2*N))
    for i in range(N):
        m_sqrt[2*i, 2*i] = 1/np.sqrt(m0[i])
        m_sqrt[2*i+1, 2*i+1] = 1/np.sqrt(m0[i])
        m_inv[2*i, 2*i] = 1/m0[i]
        m_inv[2*i+1, 2*i+1] = 1/m0[i]

    #M = m_sqrt.dot(M).dot(m_sqrt)
    M = m_inv.dot(M)

    w,v = np.linalg.eig(M)
    
    return w,v

def DM_mass_DiffK_Yfixed(N, x0, y0, D0, m0, Lx, y_bot, y_top, k_list, k_type):
    
    import numpy as np
    
    M = np.zeros((2*N, 2*N))
    contactNum = 0

    for i in range(N):
        r_now = 0.5*D0[i]
        if y0[i]-y_bot<r_now or y_top-y0[i]<r_now:
            M[2*i+1, 2*i+1] = M[2*i+1, 2*i+1] + k_list[k_type[i]] / r_now / r_now
        for j in range(i):
            dij = 0.5 * (D0[i] + D0[j])
            dijsq = dij**2
            dx = x0[i] - x0[j]
            dx = dx - round(dx / Lx) * Lx
            dy = y0[i] - y0[j]
            rijsq = dx**2 + dy**2
            if rijsq < dijsq:
                contactNum += 1  
                k = k_list[(k_type[i] ^ k_type[j]) + np.maximum(k_type[i], k_type[j])]
                rijmat = np.array([[dx*dx, dx*dy], [dx*dy, dy*dy]])
                rij = np.sqrt(rijsq)
                Mij1 = -k * rijmat / rijsq / dijsq
                Mij2 = -k * (1.0 - rij / dij) * (rijmat / rijsq - [[1,0],[0,1]]) / rij / dij
                Mij = Mij1 + Mij2
                M[2*i:2*i+2,2*j:2*j+2] = Mij
                M[2*j:2*j+2,2*i:2*i+2] = Mij
                M[2*i:2*i+2,2*i:2*i+2] = M[2*i:2*i+2,2*i:2*i+2] - Mij
                M[2*j:2*j+2,2*j:2*j+2] = M[2*j:2*j+2,2*j:2*j+2] - Mij

    m_sqrt = np.zeros((2*N, 2*N))
    m_inv = np.zeros((2*N, 2*N))
    for i in range(N):
        m_sqrt[2*i, 2*i] = 1/np.sqrt(m0[i])
        m_sqrt[2*i+1, 2*i+1] = 1/np.sqrt(m0[i])
        m_inv[2*i, 2*i] = 1/m0[i]
        m_inv[2*i+1, 2*i+1] = 1/m0[i]

    #M = m_sqrt.dot(M).dot(m_sqrt)

    w,v = np.linalg.eig(M)
    
    return w,v

def DM_mass_Zfixed_3D(N, x0, y0, z0, D0, m0, L):
    
    import numpy as np
    
    Lx = L[0]
    Ly = L[1]
    Lz = L[2]
    M = np.zeros((3*N, 3*N))
    contactNum = 0

    for i in range(N):
        for j in range(i):
            dij = 0.5*(D0[i]+D0[j])
            dijsq = dij**2
            dx = x0[i]-x0[j]
            dx = dx-round(dx/Lx)*Lx
            dy = y0[i]-y0[j]
            dy = dy-round(dy/Ly)*Ly
            dz = z0[i]-z0[j]
            rijsq = dx**2+dy**2+dz**2
            if rijsq<dijsq:
                contactNum += 1  
                rijmat = np.array([[dx*dx, dx*dy, dx*dz], [dy*dx, dy*dy, dy*dz], [dz*dz, dz*dy, dz*dz]])
                rij = np.sqrt(rijsq)
                Mij1 = -rijmat/rijsq/dijsq
                Mij2 = -(1-rij/dij)*(rijmat/rijsq-[[1,0,0],[0,1,0],[0,0,1]])/rij/dij
                Mij = Mij1+Mij2
                M[3*i:3*i+3,3*j:3*j+3] = Mij
                M[3*j:3*j+3,3*i:3*i+3] = Mij
                M[3*i:3*i+3,3*i:3*i+3] = M[3*i:3*i+3,3*i:3*i+3] - Mij
                M[3*j:3*j+3,3*j:3*j+3] = M[3*j:3*j+3,3*j:3*j+3] - Mij

    m_sqrt = np.zeros((3*N, 3*N))
    m_inv = np.zeros((3*N, 3*N))
    for i in range(N):
        m_sqrt[3*i, 3*i] = 1/np.sqrt(m0[i])
        m_sqrt[3*i+1, 3*i+1] = 1/np.sqrt(m0[i])
        m_sqrt[3*i+2, 3*i+2] = 1/np.sqrt(m0[i])
        m_inv[3*i, 3*i] = 1/m0[i]
        m_inv[3*i+1, 3*i+1] = 1/m0[i]
        m_inv[3*i+2, 3*i+2] = 1/m0[i]

    #M = m_sqrt.dot(M).dot(m_sqrt)
    M = m_inv.dot(M)

    w,v = np.linalg.eig(M)
    
    return w,v


def DM_mass_UpPlate(N, x0, y0, D0, m0, Lx, y_up, m_up):
    
    import numpy as np
    
    M = np.zeros((2*N+1, 2*N+1))
    contactNum = 0

    for i in range(N):
        for j in range(i):
            dij = 0.5*(D0[i]+D0[j])
            dijsq = dij**2
            dx = x0[i]-x0[j]
            dx = dx-round(dx/Lx)*Lx
            dy = y0[i]-y0[j]
            rijsq = dx**2+dy**2
            if rijsq<dijsq:
                contactNum += 1  
                rijmat = np.array([[dx*dx, dx*dy], [dx*dy, dy*dy]])
                rij = np.sqrt(rijsq)
                Mij1 = -rijmat/rijsq/dijsq
                Mij2 = -(1-rij/dij)*(rijmat/rijsq-[[1,0],[0,1]])/rij/dij
                Mij = Mij1+Mij2
                M[2*i:2*i+2,2*j:2*j+2] = Mij
                M[2*j:2*j+2,2*i:2*i+2] = Mij
                M[2*i:2*i+2,2*i:2*i+2] = M[2*i:2*i+2,2*i:2*i+2] - Mij
                M[2*j:2*j+2,2*j:2*j+2] = M[2*j:2*j+2,2*j:2*j+2] - Mij

    for i in range(N):
        r_now = 0.5*D0[i]
        if y0[i]<r_now:
            M[2*i+1, 2*i+1] = M[2*i+1, 2*i+1]+1/r_now**2
        if y_up-y0[i]<r_now:
            M[2*i+1, 2*i+1] = M[2*i+1, 2*i+1]+1/r_now**2
            M[2*N, 2*N] = M[2*N, 2*N]+1/r_now**2
            M[2*i+1, 2*N] = M[2*i+1, 2*N]-1/r_now**2
            M[2*N, 2*i+1] = M[2*N, 2*i+1]-1/r_now**2    
    
    m_sqrt = np.zeros((2*N+1, 2*N+1))
    m_inv = np.zeros((2*N+1, 2*N+1))
    for i in range(N):
        m_sqrt[2*i, 2*i] = 1/np.sqrt(m0[i])
        m_sqrt[2*i+1, 2*i+1] = 1/np.sqrt(m0[i])
        m_inv[2*i, 2*i] = 1/m0[i]
        m_inv[2*i+1, 2*i+1] = 1/m0[i]
    m_sqrt[2*N, 2*N] = 1/np.sqrt(m_up)
    m_inv[2*N, 2*N] = 1/m_up
    
    #M = m_sqrt.dot(M).dot(m_sqrt)
    M = m_inv.dot(M)

    w,v = np.linalg.eig(M)
    
    return w,v

def DM_mass_UpPlate_3D(N, x0, y0, z0, D0, m0, Lx, Ly, z_up, m_up):
    
    import numpy as np
    
    M = np.zeros((3*N+1, 3*N+1))
    contactNum = 0

    for i in range(N):
        for j in range(i):
            dij = 0.5*(D0[i]+D0[j])
            dijsq = dij**2
            dx = x0[i]-x0[j]
            dx = dx-round(dx/Lx)*Lx
            dy = y0[i]-y0[j]
            dy = dy-round(dy/Ly)*Ly
            dz = z0[i]-z0[j]
            rijsq = dx**2+dy**2+dz**2
            if rijsq<dijsq:
                contactNum += 1  
                rijmat = np.array([[dx*dx, dx*dy, dx*dz], [dy*dx, dy*dy, dy*dz], [dz*dz, dz*dy, dz*dz]])
                rij = np.sqrt(rijsq)
                Mij1 = -rijmat/rijsq/dijsq
                Mij2 = -(1-rij/dij)*(rijmat/rijsq-[[1,0,0],[0,1,0],[0,0,1]])/rij/dij
                Mij = Mij1+Mij2
                M[3*i:3*i+3,3*j:3*j+3] = Mij
                M[3*j:3*j+3,3*i:3*i+3] = Mij
                M[3*i:3*i+3,3*i:3*i+3] = M[3*i:3*i+3,3*i:3*i+3] - Mij
                M[3*j:3*j+3,3*j:3*j+3] = M[3*j:3*j+3,3*j:3*j+3] - Mij


    for i in range(N):
        r_now = 0.5*D0[i]
        if z0[i]<r_now:
            M[3*i+2, 3*i+2] = M[3*i+2, 3*i+2]+1/r_now**2
        if z_up-z0[i]<r_now:
            M[3*i+2, 3*i+2] = M[3*i+2, 3*i+2]+1/r_now**2
            M[3*N, 3*N] = M[3*N, 3*N]+1/r_now**2
            M[3*i+2, 3*N] = M[3*i+2, 3*N]-1/r_now**2
            M[3*N, 3*i+2] = M[3*N, 3*i+2]-1/r_now**2
            
    m_sqrt = np.zeros((3*N+1, 3*N+1))
    m_inv = np.zeros((3*N+1, 3*N+1))
    for i in range(N):
        m_sqrt[3*i, 3*i] = 1/np.sqrt(m0[i])
        m_sqrt[3*i+1, 3*i+1] = 1/np.sqrt(m0[i])
        m_sqrt[3*i+2, 3*i+2] = 1/np.sqrt(m0[i])
        m_inv[3*i, 3*i] = 1/m0[i]
        m_inv[3*i+1, 3*i+1] = 1/m0[i]
        m_inv[3*i+2, 3*i+2] = 1/m0[i]
    m_sqrt[3*N, 3*N] = 1/np.sqrt(m_up)
    m_inv[3*N, 3*N] = 1/m_up    
        
    #M = m_sqrt.dot(M).dot(m_sqrt)
    M = m_inv.dot(M)

    w,v = np.linalg.eig(M)
    
    return w,v
