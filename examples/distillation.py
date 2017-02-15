from DAEpy.solvers.ocp import ocp_solver

import os
import pdb
import argparse
import numpy as np
import matplotlib.pyplot as plt

PROPERTIES = {

'benzene' : {'cp' : 134800, 'rho' : 876, 'mr' : 78.11e-3,
             'antoine' : {'A': 4.726, 'B': 1660.7, 'C': -1.46}},
'toluene' : {'cp' : 155960, 'rho' : 867, 'mr' : 92.14e-3,
             'antoine' : {'A': 4.078, 'B': 1343.9, 'C': -53.77}},
'p-xylene' : {'cp' : 181700, 'rho' : 861, 'mr' : 106.17e-3,
             'antoine' : {'A': 4.146, 'B': 1474.4, 'C': -55.37}}
}

class Column(object):

    def __init__(self, components, n_trays, pressure, feed_rate, dist_rate, feed_comp, feed_tray, T_top, T_bottom):

        self._register_components(components)

        # Number of trays in the column. Each tray is treated as an Equilibrium
        # stage and so the reboiler is included in this number.
        self.n_trays  = n_trays

        # The pressure of the column is assumed to be constant.
        self.pressure = pressure # Pa

        # Feed conditions and disillate rate.
        self.F  = feed_rate #kmol/hr
        self.Fx = feed_comp
        self.D  = dist_rate #kmol/hr

        # Tray of the column on which the feed is placed.
        self.f_tray = feed_tray

        self.T_top = T_top # K
        self.T_bot = T_bottom # K

    def _register_components(self, components):

        """
        Registers the properties for the component.
        """

        n_components = len(components)

        self.n_comps = n_components

        self.cp  = np.empty((n_components))
        self.rho = np.empty((n_components))
        self.mr  = np.empty((n_components))
        self.eq  = [0]*n_components

        for i,c in enumerate(components):

            self.cp[i]  = PROPERTIES[c]['cp']
            self.rho[i] = PROPERTIES[c]['rho']
            self.mr[i]  = PROPERTIES[c]['mr']
            self.eq[i]  = PROPERTIES[c]['antoine']

    def _antoine(self, T, comp):

        """
        Returns the staturation pressure values at the given temperatures.
        """

        if type(T) == np.ndarray:
            cof = self.eq[comp]
            psat = np.vectorize(lambda x: cof['A']+cof['B']/(x+cof['C']))
            out = psat(T)
        else:
            out = 10.0**(A+B/(val+C))

        return out

    def _mass_balance(self, V, L, x, y, t=None):

        """

        Solves for the holdup on each tray of the column.

        Parameters
        ----------

        V : array_like, shape (n_trays, m)
            Vapour flow rates on each of the trays.
        L : array_like, shape (n_trays, m)
            Liquid flow rates on each of the trays.
        x : array_like, shape (n_trays, n_comps, m)
            Liquid mole fractions out of each tray.
        y : array_like, shape (n_trays, n_comps, m)
            Vapour mole fractions out of each tray.

        Returns
        -------
        d(xM)/dt : ndarray shape (n_trays, n_comp, m)
            Rate of accumulation of holdup in each each tray.
        """

        F  = self.F
        D  = self.D
        Fx = self.Fx
        B  = F - D

        if t is not None:
            m = len(t)
            dxMdt = np.empty((self.n_trays, self.n_comps, m))
        else:
            dxMdt = np.empty((self.n_trays, self.n_comps))

        # First we balance the middle column ignoring the feed tray. Then we
        # balance the feed tray (indexed 0), with our known distillated rate.
        # Finally we balance the reboiler and feed tray.

        for i in range(self.n_comps):

            dxMdt[1:-1, i] = V[2:]*x[2:,i] - V[1:-1]*y[1:-1,i] + L[:-2]*x[:-2,i] - L[1:-1]*x[1:-1,i]

            dxMdt[0, i] = V[1]*y[1,i] - L[0]*x[0,i] - D*y[0,i]

            dxMdt[-1, i] = L[2]*y[2,i] - V[-1]*y[-1,i] - B*x[-1,i]

            dxMdt[self.f_tray, i] += Fx[i] * F

        return dxMdt

    def _equilibrium(self, x, y, T, P, t=None):

        """
        Solves for the equilibrium on each tray.

        Parameters
        ----------

        x : array_like, shape (n_trays, n_comps, m)
            Liquid mole fractions out of each tray.
        y : array_like, shape (n_trays, n_comps, m)
            Vapour mole fractions out of each tray.
        T : array_like, shape (n_tray, m)
            Temperature of the trays.
        P : scalar
            Pressure of the column.

        Returns
        -------
        y - Kx : ndarray shape (n_trays, n_comp, m)
            Error in the equilibrium.
        """

        if t is not None:
            m  = len(t)
            eq = np.empty((self.n_trays, self.n_comps, m))
        else:
            eq = np.empty((self.n_trays, self.n_comps))

        for i in range(self.n_comps):
            K = self._antoine(T, i)/P
            eq[:, i] = y[:, i] - K*x[:, i]
        return eq

    def _mole_fraction(self, x, t=None):

        """
        Solves for the mole fraction.

        Parameters
        ----------

        x : array_like, shape (n_trays, n_comps, m)
            Liquid mole fractions out of each tray.

        Returns
        -------
        sum(x)-1.0 : ndarray shape (n_trays, m)
            Sum of the mole fractions.
        """

        if t is not None:
            sum_x = np.empty((self.n_trays, len(t)))
        else:
            sum_x = np.empty((self.n_trays))

        sum_x = np.sum(x, axis=1) - 1.0

        return sum_x

    def _weir(self, hu, x, y, L, t=None):

        """
        Solves for the weir flow.

        Parameters
        ----------

        hu : array_like, shape (n_trays, n_comps, m)
            Holdup in each tray of the column
        x : array_like, shape (n_trays, n_comps, m)
            Liquid mole fractions out of each tray.
        L : array_like, shape (n_trays, m)
            Liquid flow rates on each of the trays.

        Returns
        -------
        out : ndarray shape (n_trays, n_comps, m)
            Error in the tray holdup.
        """

        # Temporary variables.
        A = 7.36
        hw = 0.05
        l = 1.36

        # Calculate the mean density and average molecular weight, assuming
        # and ideal mixture.
        rho = np.einsum('ij...,j...->i...', x, self.rho)
        Mw  = np.einsum('ij...,j...->i...', x, self.mr)

        Mk  = rho * A *(hw + 1.41*L*Mw/(rho*l*9.81**0.5))/Mw
        
        if t is not None:
            out = np.empty((self.n_trays, self.n_comps, len(t)))
        else:
            out = np.empty((self.n_trays, self.n_comps))

        for i in range(self.n_comps):
            out[:,i] = hu[:,i] - x[:,i]*Mk

        return out

    def derivatives(self, dv, av, t):

        # Algebraic variables are in the format:
        # 1 - L, N_TRAYS
        # 2 - V, N_TRAYS
        # 3 - x, N_COMPS*N_TRAYS
        # 4 - y, N_COMPS*N_TRAYS

        algs = np.split(av, np.cumsum([self.n_trays,
                                       self.n_trays,
                                       self.n_comps*self.n_trays]))

        L = np.reshape(algs[0], (self.n_trays, len(t)), order='F')
        V = np.reshape(algs[1], (self.n_trays, len(t)), order='F')
        x = np.reshape(algs[2], (self.n_trays, self.n_comps, len(t)),order='F')
        y = np.reshape(algs[3], (self.n_trays, self.n_comps, len(t)),order='F')


        dxMdt = self._mass_balance(V, L, x, y, t)

        return np.reshape(dxMdt,(self.n_trays*self.n_comps, len(t)), order='F')

    def constraints(self, dv, av, t):

        # Algebraic variables are in the format:
        # 1 - L, N_TRAYS
        # 2 - V, N_TRAYS
        # 3 - x, N_COMPS*N_TRAYS
        # 4 - y, N_COMPS*N_TRAYS

        algs = np.split(av, np.cumsum([self.n_trays,
                                       self.n_trays,
                                       self.n_comps*self.n_trays]))

        L = np.reshape(algs[0], (self.n_trays, len(t)), order='F')
        V = np.reshape(algs[1], (self.n_trays, len(t)), order='F')
        x = np.reshape(algs[2], (self.n_trays, self.n_comps, len(t)),order='F')
        y = np.reshape(algs[3], (self.n_trays, self.n_comps, len(t)),order='F')

        # Differential variables are in the format:
        # 1 - hu, N_COMPS*N_TRAYS

        hu = np.reshape(dv, (self.n_trays, self.n_comps, len(t)), order='F')

        T = np.linspace(self.T_top, self.T_bot, self.n_trays)
        T = np.repeat(T[...,np.newaxis], len(t), axis=-1)

        el  = self._equilibrium(x, y, T, self.pressure, t)
        mfx = self._mole_fraction(x, t)
        mfy = self._mole_fraction(y, t)
        w   = self._weir(hu, x, y, L, t)

        el = np.reshape(el, (self.n_trays*self.n_comps, len(t)), order='F')
        w  = np.reshape(w, (self.n_trays*self.n_comps, len(t)), order='F')

        cons = np.concatenate((el, mfx, mfy, w))

        return np.einsum('ij...,ij...->j...', cons, cons)

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

    components = ["benzene", "toluene", "p-xylene"]
    n_trays    = 10
    pressure   = 1e5
    feed_rate  = 100
    dist_rate  = 50
    feed_comp  = np.array([0.15, 0.20, 0.65])
    feed_tray  = 5
    T_top      = 367
    T_bottom   = 416

    model = Column(components, n_trays, pressure, feed_rate, dist_rate, feed_comp, feed_tray, T_top, T_bottom)

    numt = 50
    numx = model.n_comps*model.n_trays
    numy = 2*(1+model.n_comps)*model.n_trays

    t = np.linspace(0, 5,numt, dtype=np.float64)
    x = np.random.uniform(1.0, 10, (numx, numt))
    y = np.random.uniform(1.0, 10, (numy, numt))

    x0 = np.ones((numx), dtype=np.float64)
    w = 0.5
    m = 1.0e5

    sol = ocp_solver(model.constraints, model.derivatives, x, y, t, x0, m, verbose = 2, tol = 1e-4, max_nodes = 10000)

    plt.rc('text', usetex=True)

    f1, ax1 = plt.subplots(2,figsize=(10,7))

    ax1[0].set_xlabel("t")
    ax1[0].set_ylabel("Holdups")
    ax1[0].plot(sol.t, sol.x.T)

    ax1[1].set_xlabel("t")
    ax1[1].set_ylabel("RMS Residuals")
    ax1[1].plot(sol.t[1:], sol.rms_residuals)
    ax1[1].set_ylim([0,2e-8])

    f2, ax2 = plt.subplots(2,figsize=(10,7))

    ax2[0].set_xlabel("t")
    ax2[0].set_ylabel(r"$H$")
    ax2[0].plot(sol.t, sol.h.T)

    ax2[1].set_xlabel("t")
    ax2[1].set_ylabel(r"$ ||\nabla H ||_{\infty}$")
    ax2[1].plot(sol.t, sol.hu_norm)

    if args.folder:
        f1.savefig(os.path.join(args.folder, "index_one_I_variables.png"))
        f2.savefig(os.path.join(args.folder, "index_one_I_hamiltonian.png"))
    else:
        print("Showing")
        plt.show()
