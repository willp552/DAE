from DAEpy.solvers import dae_solver_two

import pdb
import numpy as np
import matplotlib.pyplot as plt

def f(x,y,t):

    return np.array([0.52*y[0]-0.48*x[1],
                     y[0]])

def g(x,t):

    return np.array([x[0]-0.52*t*x[1]-np.exp(-t)])

def fx(x,y,t):

    temp = np.array([[0.0,-0.48],[0.0,0.0]])

    return np.repeat(temp[:,:,np.newaxis],len(t),-1)

def fy(x,y,t):

    temp = np.array([[0.52],[1.0]])

    return np.repeat(temp[:,:,np.newaxis],len(t),-1)

def gx(x,t):

    return np.array([[[1.0]*len(t), -0.52*t]])

if __name__ == "__main__":

    numt = 100
    numx = 2
    numy = 1
    x0 = np.array([1.0,1.0], dtype=np.float64)

    t = np.linspace(0,0.5,numt, dtype=np.float64)

    w = 0.5
    m = 1.0e6

    xresult = np.empty((numx, numt), dtype=np.float64)
    yresult = np.empty((numy, numt), dtype=np.float64)

    for i in range(numt-1):

        ts = np.linspace(t[i],t[i+1],numt, dtype=np.float64)
        xs = np.zeros((numx, numt), dtype=np.float64)
        ys = np.zeros((numy, numt), dtype=np.float64)

        sol = dae_solver_two(f, g, xs, ys, ts, x0, m, w, gx, fy, fx, verbose = 2, tol = 1e-7, max_nodes = 10000)

        soli = sol.sol(t[i])
        solx, soly, _ = np.split(soli, np.cumsum([numx, numy]))
        x0 = sol.sol(t[i+1])[:numx]

        xresult[:, i] = solx
        yresult[:, i] = soly

    f, ax = plt.subplots(2)

    ax[0].set_xlabel("Times (Arbitrary Units)")
    ax[0].set_ylabel("Concentration \n (Arbitrary Units)")
    ax[0].plot(t, xresult.T)

    plt.show()
