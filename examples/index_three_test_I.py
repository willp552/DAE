
from DAEpy.solvers import dae_solver_three

import pdb
import numpy as np
import matplotlib.pyplot as plt

def f(x,y,z,t):

    return np.array([-z[0]*y[0], -z[0]*y[1]-9.81])

def g(x,y,t):

    return np.array([x[0], x[1]])

def h(y,t):

    return np.array([y[0]**2 + y[1]**2 - 1.0])

if __name__ == "__main__":

    numt = 5
    numx = 2
    numy = 2
    numz = 1

    t = np.linspace(0,1,numt, dtype=np.float64)

    x = np.zeros((numx, numt), dtype=np.float64)
    y = np.zeros((numy, numt), dtype=np.float64)
    z = np.zeros((numz, numt), dtype=np.float64)

    x0 = np.array([0.0,0.0], dtype=np.float64)
    y0 = np.array([1.0,0.0], dtype=np.float64)

    w = 0.5
    m = 1.0e5

    sol = dae_solver_three(f, g, h, x, y, z, t, x0, y0, m, w, verbose = 2, tol = 1e-5)

    f, ax = plt.subplots(2)

    ax[0].set_xlabel("Times (Arbitrary Units)")
    ax[0].set_ylabel("Concentration \n (Arbitrary Units)")
    ax[0].plot(sol.x, sol.y[:4].T)

    ax[1].set_xlabel("Times (Arbitrary Units)")
    ax[1].set_ylabel("RMS Residuals")
    ax[1].plot(sol.x[1:], sol.rms_residuals)

    plt.show()
