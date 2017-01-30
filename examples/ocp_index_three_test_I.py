from DAEpy.solvers.ocp import ocp_solver

import numpy as np
import matplotlib.pyplot as plt

def f(x,y,t):

    return np.array([x[2], x[3], -y[0]*x[0], -y[0]*x[1]-9.81])

def g(x,t):

    return np.array([x[0]**2+x[1]**2-1.0])

def fx(x,y,t):

    out = np.zeros((4, 4, len(t)))
    out[0,2,:] = 1.0
    out[1,3,:] = 1.0
    out[2,0,:] = -y[0]
    out[3,1,:] = -y[0]

    return out

def fy(x,y,t):

    out = np.zeros((4, 1, len(t)))
    out[2,0,:] = -x[0]
    out[3,0,:] = -x[1]

    return out

def gx(x,t):

    out = np.zeros((1,4,len(t)))
    out[0,0,:] = 2*x[0]
    out[0,1,:] = 2*x[1]

    return out

def L(x,u,t):

    G = g(x,t)

    return np.einsum('ij...,ij...->j...', G, G)

def Lx(x,u,t):

    G = g(x,t)
    Gx = gx(x,t)

    return 2*np.einsum('i...,i...->...', G, Gx)

def Lu(x,u,t):

    nu = u.shape[0]
    nt = t.shape[0]

    return np.zeros((nu, nt))

if __name__ == "__main__":

    numt = 500
    numx = 4
    numy = 1

    xresult = np.empty((numx, numt), dtype=np.float64)
    yresult = np.empty((numy, numt), dtype=np.float64)

    t = np.linspace(0,3.0,numt, dtype=np.float64)
    x = np.zeros((numx, numt), dtype=np.float64)
    y = np.zeros((numy, numt), dtype=np.float64)

    #x = x_actual(t)
    #y = y_actual(t)

    x0 = np.array([1.0,0.0,0.0,0.0], dtype=np.float64)
    w = 1.0

    m = 1.0e2

    sol = ocp_solver(L, f, x, y, t, x0, m, Lx=Lx, Lu=Lu, fx=fx, fu=fy, verbose = 2, tol = 1e-4, max_nodes = 2000)

    _ , ax = plt.subplots(2)

    ax[0].set_xlabel("t")
    ax[0].set_ylabel("Differential Variables")
    ax[0].plot(sol.x, sol.y[:3].T)

    ax[1].set_xlabel("Collocation Residuals")
    ax[1].set_ylabel("RMS Residuals")
    ax[1].plot(sol.x[1:], sol.rms_residuals)
    """
    f, ax = plt.subplots(2)

    ax[0].set_xlabel("t")
    ax[0].set_ylabel("Absolute Error")
    ax[0].plot(sol.x, sol.y[:2].T - x_actual(sol.x).T)

    ax[1].set_xlabel("Times (Arbitrary Units)")
    ax[1].set_ylabel("Relative Error")
    ax[1].plot(sol.x[1:], (sol.y[:2].T - x_actual(sol.x).T)[1:]/x_actual(sol.x).T[1:])
    """
    plt.show()
