from DAEpy.solvers.ocf import dae_solver_I

import numpy as np
import matplotlib.pyplot as plt

"""
This is a simple demonstration of the proposed method for solving index 1
differential algebraic equations.

In this example a first order reaction is used, for which there is an
explicit solution.

                    A + B -> C

        dCa/dt = -r
        dCb/dt = -r
        dCc/dt = +r

        r - CaCb = 0

x = c with length 3
y = r with length 1
"""

def f(x,y,t):

    return np.array([-y[0], -y[0], y[0]])

def g(x,y,t):

    return np.array([y[0]-x[0]*x[1]])

def gy(x,y,t):

    tmp = np.array([[1.0]])

    return np.repeat(tmp[...,np.newaxis],len(t),-1)

def fy(x,y,t):

    tmp = np.array([[-1.0], [-1.0], [1.0]])

    return np.repeat(tmp[...,np.newaxis],len(t),-1)

def gx(x,y,t):

    return np.array([[-x[1], -x[0], [0]*len(t)]])

def fx(x,y,t):

    tmp = np.array([[0,0,0],[0,0,0],[0,0,0]])

    return np.repeat(tmp[..., np.newaxis],len(t),-1)

if __name__ == "__main__":

    numt = 100
    numx = 3
    numy = 1

    t = np.linspace(0,1,100)
    x = np.zeros((numx, numt))
    y = np.zeros((numy, numt))

    x0 = np.array([1.0,1.0,0.0])
    w = 0.5
    m = 1000

    sol = dae_solver_I(f, g, x, y, t, x0, m, w, verbose = 2, tol = 1e-5)

    f, ax = plt.subplots(2)

    ax[0].set_xlabel("t")
    ax[0].set_ylabel("Concentration")
    ax[0].plot(sol.x, sol.y[:3].T)

    ax[1].set_xlabel("t")
    ax[1].set_ylabel("RMS Residuals")
    ax[1].plot(sol.x[1:], sol.rms_residuals)

    plt.show()
