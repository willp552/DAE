from DAEpy.solvers import dae_solver_one

import pdb
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

    return np.array([-y[0], -y[0], y[0]], dtype=np.float64)

def g(x,y,t):

    return np.array([y[0]-x[0]*x[1]], dtype=np.float64)

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

def c_exact(t):

    return np.array([1.0/(1.0+t), 1.0/(1.0+t), t/(1.0+t)])

if __name__ == "__main__":

    numt = 100
    numx = 3
    numy = 1

    t = np.linspace(0,1,100, dtype=np.float64)
    x = np.zeros((numx, numt), dtype=np.float64)
    y = np.zeros((numy, numt), dtype=np.float64)

    x0 = np.array([1.0,1.0,0.0], dtype=np.float64)
    w = 0.5
    m = 1.0e9

    sol = dae_solver_one(f, g, x, y, t, x0, m, w, verbose = 2, tol = 1e-5)

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
    #ax[1].plot(sol.x, (sol.y[:3].T - c_exact(sol.x).T)/c_exact(sol.x).T)

    plt.show()

    #pdb.set_trace()
