from DAEpy.solvers.ocp import ocp_solver

import numpy as np
import matplotlib.pyplot as plt

def f(x,y,t):

    return np.array([-y[0], -y[0], y[0]], dtype=np.float64)

def g(x,y,t):

    return np.array([y[0]-x[0]*x[1]], dtype=np.float64)

def c_exact(t):

    return np.array([1.0/(1.0+t), 1.0/(1.0+t), t/(1.0+t)])

def L(x,u,t):

    G = g(x,u,t)

    return np.einsum('ij...,ij...->j...', G, G)

if __name__ == "__main__":

    numt = 100
    numx = 3
    numy = 1

    t = np.linspace(0,1,numt, dtype=np.float64)
    x = np.zeros((numx, numt), dtype=np.float64)
    y = np.zeros((numy, numt), dtype=np.float64)

    x0 = np.array([1.0,1.0,0.0], dtype=np.float64)
    w = 0.5
    m = 1.0e8

    sol = ocp_solver(L, f, x, y, t, x0, m, verbose = 2, tol = 1e-7)

    f, ax = plt.subplots(2)

    ax[0].set_xlabel("Times (Arbitrary Units)")
    ax[0].set_ylabel("Concentration \n (Arbitrary Units)")
    ax[0].plot(sol.x, sol.y[:3].T)

    ax[1].set_xlabel("Times (Arbitrary Units)")
    ax[1].set_ylabel("RMS Residuals")
    ax[1].plot(sol.x[1:], sol.rms_residuals)

    f, ax = plt.subplots(2)

    ax[0].set_xlabel("Times (Arbitrary Units)")
    ax[0].set_ylabel("Absolute Error (Arbitrary Units)")
    ax[0].plot(sol.x, sol.y[:3].T - c_exact(sol.x).T)

    ax[1].set_xlabel("Times (Arbitrary Units)")
    ax[1].set_ylabel("Relative Error")
    ax[1].plot(sol.x[1:], (sol.y[:3].T - c_exact(sol.x).T)[1:]/c_exact(sol.x).T[1:])

    plt.show()
