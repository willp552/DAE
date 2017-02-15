from DAEpy.solvers.ocp import ocp_solver

import numpy as np
import matplotlib.pyplot as plt

def f(x,y,t):

    return np.array([x[2], x[3], -y[0]*x[0], -y[0]*x[1]-9.81])

def g(x,t):

    return np.array([(x[0]**2+x[1]**2-1.0)*1000.0])

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

    t = np.linspace(0,0.1,numt, dtype=np.float64)
    x = np.zeros((numx, numt), dtype=np.float64)
    y = np.zeros((numy, numt), dtype=np.float64)

    #x = x_actual(t)
    #y = y_actual(t)

    x0 = np.array([1.0,0.0,0.0,0.0], dtype=np.float64)
    w = 1.0

    m = 1.0e5

    sol = ocp_solver(L, f, x, y, t, x0, m, 1e-5, Lx=Lx, Lu=Lu, fx=fx, fu=fy, verbosity = 2, is_linear = True)

    f, ax = plt.subplots(2)

    ax[0].set_xlabel("Times (Arbitrary Units)")
    ax[0].set_ylabel("Concentration \n (Arbitrary Units)")
    ax[0].plot(sol.t, sol.u.T)

    ax[1].set_xlabel("Times (Arbitrary Units)")
    ax[1].set_ylabel("RMS Residuals")
    #ax[1].plot(sol.t[1:], sol.rms_residuals)


    """
    f, ax = plt.subplots(2)

    ax[0].set_xlabel("Times (Arbitrary Units)")
    ax[0].set_ylabel("Absolute Error (Arbitrary Units)")
    ax[0].plot(sol.t, sol.x.T - c_exact(sol.t).T)

    ax[1].set_xlabel("Times (Arbitrary Units)")
    ax[1].set_ylabel("Relative Error")
    ax[1].plot(sol.t[1:], (sol.x.T - c_exact(sol.t).T)[1:]/c_exact(sol.t).T[1:])

    f, ax = plt.subplots(2)

    ax[0].set_xlabel("Times (Arbitrary Units)")
    ax[0].set_ylabel("Hamiltonian (Arbitrary Units)")
    ax[0].plot(sol.t, sol.h.T)

    ax[1].set_xlabel("Times (Arbitrary Units)")
    ax[1].set_ylabel("Hamiltonian Infinity Norm")
    ax[1].plot(sol.t, sol.hu_norm)
    """
    plt.show()
