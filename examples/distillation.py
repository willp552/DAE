from DAEpy.solvers.ocp import ocp_solver

import numpy as np
import matplotlib.pyplot as plt

"""
********************************************************************************
                                    Data
                                    ----

Components:
   1 - Benzene
   2 - Toluene
   3 - Paraxylene

********************************************************************************
"""

# Heat Capcities:
CP = np.array([134800, 155960, 181700])

# Antoine Coefficients:
A = np.array([4.726, 4.078, 4.146])
B = np.array([1660.7, 1343.9, 1474.4])
C = np.array([-1.46, -53.77, -55.37])

"""
********************************************************************************
                            Design Specification
                            --------------------
********************************************************************************
"""

# Number of variables
N_COMPS = 3
N_TRAYS = 10

# Feed flow rate:
F = 100 #kmol/hr
D = 50  #kmol/hr

# Equilibrium constraints:
K = np.ones(N_TRAYS)

def dynamics(xv,yv,t):
    algs = np.split(yv, np.cumsum([N_COMPS*N_TRAYS,
                                    N_COMPS*N_TRAYS,
                                    N_COMPS*N_TRAYS,
                                    N_COMPS*N_TRAYS]))

    V, L, x, y = [np.reshape(a, (N_COMPS,N_TRAYS,len(t))) for a in algs]

    # Middle tray component balances:
    comp_balance = V[:,2:]*y[:,2:] - V[:,1:-1]*y[:,1:-1] + L[:,:-2]*x[:,:-2] - L[:,1:-1]*x[:,1:-1]

    # Top tray component balances
    top_tray = V[:,1]*y[:,1] - L[:,0]*x[:,0] - D*y[:,0]

    # Bottom tray component balances
    bottom_tray = L[:,-2]*x[:,-2] - V[:,-1]*y[-1] - (F - D)*x[:,-1]


def constraints(xv,yv,t):

    algs = np.split(yv, np.cumsum([N_COMPS*N_TRAYS,
                                    N_COMPS*N_TRAYS,
                                    N_COMPS*N_TRAYS,
                                    N_COMPS*N_TRAYS]))

    V, L, x, y = [np.reshape(a, (N_COMPS,N_TRAYS,len(t))) for a in algs]

    # Mole fraction constraint.
    y_con = np.sum(y, axis=0)
    x_con = np.sum(x, axis=0)
