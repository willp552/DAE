from DAEpy.solvers.ocp import ocp_solver

import os
import argparse
import numpy as np
import matplotlib.pyplot as plt

def f(x,y,t):

    return np.array([-0.5*(1+t)*x[1], y[0]])

def g(x,t):

    return np.array([x[0]-0.5*t*x[1]-np.exp(-t)])

def fx(x,y,t):

    temp = np.array([[0.0,0.0],[0.0,0.0]])
    temp = np.repeat(temp[:,:,np.newaxis],len(t),-1)

    temp[0,1,:] = -0.5*(1+t)

    return temp

def fy(x,y,t):

    temp = np.array([[0.0],[1.0]])

    return np.repeat(temp[:,:,np.newaxis],len(t),-1)

def gx(x,t):

    return np.array([[[1.0]*len(t), -0.5*t]])


def x_actual(t):

    return np.array([(1+0.5*t)*np.exp(-t), np.exp(-t)])

def L(x,u,t):

    G = g(x,t)

    return 0.5*np.einsum('ij...,ij...->j...', G, G)

def Lx(x,u,t):

    G = g(x,t)
    Gx = gx(x,t)

    return 2*np.einsum('i...,i...->...', G, Gx)

def Lu(x,u,t):

    nu = u.shape[0]
    nt = t.shape[0]

    return np.zeros((nu, nt))

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

    numt = 50
    numx = 2
    numy = 1

    xresult = np.empty((numx, numt), dtype=np.float64)
    yresult = np.empty((numy, numt), dtype=np.float64)

    t = np.linspace(0,3.0,numt, dtype=np.float64)
    x = np.zeros((numx, numt), dtype=np.float64)
    y = np.zeros((numy, numt), dtype=np.float64)

    #x = x_actual(t)
    #y = y_actual(t)

    x0 = np.array([1.0,1.0], dtype=np.float64)
    w = 1.0

    m = 1.0e8

    sol = ocp_solver(L, f, x, y, t, x0, m, Lx=Lx, Lu=Lu, fx=fx, fu=fy, verbose = 2, tol = 1e-4, max_nodes = 10000)

    plt.rc('text', usetex=True)

    f1, ax1 = plt.subplots(2,figsize=(10,7))

    ax1[0].set_xlabel("t")
    ax1[0].set_ylabel("Differential Variables)")
    ax1[0].plot(sol.t, sol.x.T)

    ax1[1].set_xlabel("t")
    ax1[1].set_ylabel("RMS Residuals")
    ax1[1].plot(sol.t[1:], sol.rms_residuals)

    f2, ax2 = plt.subplots(2,figsize=(10,7))

    ax2[0].set_xlabel("t")
    ax2[0].set_ylabel("Absolute Error")
    ax2[0].plot(sol.t, sol.x.T - x_actual(sol.t).T)

    ax2[1].set_xlabel("t")
    ax2[1].set_ylabel("Relative Error")
    ax2[1].plot(sol.t[1:], (sol.x.T - x_actual(sol.t).T)[1:]/x_actual(sol.t).T[1:])

    f3, ax3 = plt.subplots(2,figsize=(10,7))

    ax3[0].set_xlabel("t")
    ax3[0].set_ylabel(r"$H$")
    ax3[0].plot(sol.t, sol.h.T)

    ax3[1].set_xlabel("t")
    ax3[1].set_ylabel(r"$ ||\nabla H ||_{\infty}$")
    ax3[1].plot(sol.t, sol.hu_norm)

    if args.folder:
        f1.savefig(os.path.join(args.folder, "index_two_I_variables.png"))
        f2.savefig(os.path.join(args.folder, "index_two_I_errors.png"))
        f3.savefig(os.path.join(args.folder, "index_two_I_hamiltonian.png"))
    else:
        print("Showing")
        plt.show()
