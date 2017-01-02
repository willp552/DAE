"""
This is a simple demonstration of the proposed method for solving index 1
differential algebraic equations.
"""

import numpy as np
import scikits.bvp1lg.colnew as colnew
import matplotlib.pyplot as plt
import pdb

mu = 100.0
a1 = 0.5
a2 = 0.5

numx = 3
numy = 1

dgs = [1] * (2*numx + numy)

x0 = np.array([1.0,1.0,0.0])

bndry = np.array([0.0,1.0,0.0,0.0,0.0,0.0,0.0])
tol = [1e-6] * (2*numx + numy)

def f(x,y,t):

    return np.array([-y[0], -y[0], y[0]])

def g(x,y,t):

    return np.array([y[0]-x[0]*x[1]])

def g_y(x,y,t):

    tmp = np.array([[1.0]])

    return np.repeat(tmp[...,np.newaxis],len(t),-1)

def f_y(x,y,t):

    tmp = np.array([[-1.0], [-1.0], [1.0]])

    return np.repeat(tmp[...,np.newaxis],len(t),-1)

def g_x(x,y,t):

    return np.array([[-x[1], -x[0], [0]*len(t)]])

def f_x(x,y,t):

    tmp = np.array([[0,0,0],[0,0,0],[0,0,0]])

    return np.repeat(tmp[..., np.newaxis],len(t),-1)

def ode(t,X):

    """
    output: [x,y,l] where dim(x) = numx, dim(y) = numy, dim(l) = numx.
    """

    nx = len(t)

    # Parse the variables.
    x = X[:numx]
    y = X[numx:numx+numy]
    l = X[numx+numy:numx*2+numy]

    # Initialise the output
    out = np.empty((numx + numx + numy, nx))

    F   = f(x,y,t)
    G   = g(x,y,t)
    G_y = g_y(x,y,t)
    G_x = g_x(x,y,t)
    F_y = f_y(x,y,t)
    F_x = f_x(x,y,t)

    mult = lambda x, y: np.einsum("ji...,j...->i...", x, y)

    # The output

    out[:numx] = F
    out[numx:numx+numy] = - mu * (a2 * mult(G_y,G) + mult(F_y,l))
    out[numx+numy:numx*2+numy] = -a2 * mult(G_x,G) - mult(F_x,l)

    return out

def bdy(X):

    t = bndry

    # Parse the variables.
    x = X[:numx]
    y = X[numx:numx+numy]
    l = X[numx+numy:numx*2+numy]

    F   = f(x,y,t)
    G   = g(x,y,t)
    G_y = g_y(x,y,t)
    G_x = g_x(x,y,t)
    F_y = f_y(x,y,t)
    F_x = f_x(x,y,t)

    out = np.empty(numx + numx + numy)

    mult = lambda x, y: np.einsum("ji...,j...->i...", x, y)

    out[:numx] = x[:,0]-x0
    out[numx:numx+numy] = - mu * (a2 * mult(G_y,G) + mult(F_y,l))[:,0]
    out[numx+numy:numx*2+numy] = l[:,1] - a1 * mult(G_x,G)[:,1]

    return out

if __name__ == "__main__":

    solution = colnew.solve(bndry, dgs, ode, bdy, tolerances = tol)
    plt.plot(solution.mesh, solution(solution.mesh)[:,:3])
    plt.show()
