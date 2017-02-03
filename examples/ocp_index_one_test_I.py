from DAEpy.solvers.ocp import ocp_solver

import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import pdb

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

def L(x,u,t):

    G = g(x,u,t)

    return np.einsum('ij...,ij...->j...', G, G)

def Lx(x,u,t):

    G = g(x,u,t)
    Gx = gx(x,u,t)

    return 2*np.einsum('i...,i...->...', G, Gx)

def Lu(x,u,t):

    G = g(x,u,t)
    Gu = gy(x,u,t)

    return 2*np.einsum('i...,i...->...', G, Gu)

if __name__ == "__main__":

    """
    Parse the command line arguments.
    """

    parser = argparse.ArgumentParser(description="Parse example arguments.")
    parser.add_argument('--folder', default=None, type=str, dest='folder')
    args = parser.parse_args()

    """
    Perform the calculations.
    """

    numt = 500
    numx = 3
    numy = 1

    t = np.linspace(0,1,numt, dtype=np.float64)
    x = np.zeros((numx, numt), dtype=np.float64)
    y = np.zeros((numy, numt), dtype=np.float64)

    x0 = np.array([1.0,1.0,0.0], dtype=np.float64)
    w = 0.5
    m = 1.0e5

    sol = ocp_solver(L, f, x, y, t, x0, m, Lx=Lx, Lu=Lu, fx=fx, fu=fy, verbose = 2, tol = 1e-4, max_nodes = 10000)

    plt.rc('text', usetex=True)

    f1, ax1 = plt.subplots(2,figsize=(10,7))

    ax1[0].set_xlabel("t")
    ax1[0].set_ylabel("Concentrations")
    ax1[0].plot(sol.t, sol.x.T)

    ax1[1].set_xlabel("t")
    ax1[1].set_ylabel("RMS Residuals")
    ax1[1].plot(sol.t[1:], sol.rms_residuals)
    ax1[1].set_ylim([0,2e-8])

    f2, ax2 = plt.subplots(2,figsize=(10,7))

    ax2[0].set_xlabel("t")
    ax2[0].set_ylabel("Absolute Error")
    ax2[0].plot(sol.t, sol.x.T - c_exact(sol.t).T)

    ax2[1].set_xlabel("t")
    ax2[1].set_ylabel("Relative Error")
    ax2[1].plot(sol.t[1:], (sol.x.T - c_exact(sol.t).T)[1:]/c_exact(sol.t).T[1:])

    f3, ax3 = plt.subplots(2,figsize=(10,7))

    ax3[0].set_xlabel("t")
    ax3[0].set_ylabel(r"$H$")
    ax3[0].plot(sol.t, sol.h.T)

    ax3[1].set_xlabel("t")
    ax3[1].set_ylabel(r"$ ||\nabla H ||_{\infty}$")
    ax3[1].plot(sol.t, sol.hu_norm)

    if args.folder:
        f1.savefig(os.path.join(args.folder, "index_one_I_variables.png"))
        f2.savefig(os.path.join(args.folder, "index_one_I_errors.png"))
        f3.savefig(os.path.join(args.folder, "index_one_I_hamiltonian.png"))
    else:
        print("Showing")
        plt.show()
