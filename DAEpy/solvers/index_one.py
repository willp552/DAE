from scipy.integrate import solve_bvp

import numpy as np
import pdb

def estimate_jacobians(func):

    """Esimtates the jacobians by forward differencing.

    Returns
    -------
    fx : callable
        The jacobian of the function with respects to x. An element (i,j,k)
        corresponds to df_i(x_q,y_q,t_q) / d (x_q)_j. fx returns an ndarray
        of shape (nf, nx, m)
    fy : callable
        The jacobian of the function with respects to y. An element (i,j,k)
        corresponds to df_i(x_q,y_q,t_q) / d (y_q)_j. fy returns an ndarray
        of shape (nf, ny, m)
    """

    def fx(x, y, t, f0=None):
        eps = np.finfo(float).eps
        nx, m = x.shape
        if f0 == None:
            f0 = func(x,y,t)

        nf = f0.shape[0]
        xdtype = x.dtype
        dfdx = np.empty((nf, nx, m), dtype=xdtype)

        hx = eps**0.5 * (1 + np.abs(x))

        for i in range(nx):
            x_new = x.copy()
            x_new[i] += hx[i]
            hi = x_new[i] - x[i]
            f_new = func(x_new, y, t)
            dfdx[:,i,:] = (f_new-f0)/hi

        return dfdx

    def fy(x, y, t, f0=None):

        eps = np.finfo(float).eps
        ny, m = y.shape
        if f0 == None:
            f0 = func(x,y,t)
        nf = f0.shape[0]
        ydtype = y.dtype
        dfdy = np.empty((nf, ny, m))

        hy = eps**0.5 * (1 + np.abs(y))

        for i in range(ny):
            y_new = y.copy()
            y_new[i] += hy[i]
            hi = y_new[i] - y[i]
            f_new = func(x, y_new, t)
            dfdy[:,i,:] = (f_new-f0)/hi

        return dfdy

    return fx, fy

def parse_jacobians(f, g, fx, fy, gx, gy):

    """Parse the Jacobians and evalate numerically if necessary.
    """

    if fx == None or fy == None:
        fx, fy = estimate_jacobians(f)

    if gx == None or gy == None:
        gx, gy = estimate_jacobians(g)

    return fx, fy, gx, gy

def vmp(x, y):
    """
    Returns the vector matrix product. A_T.x
    """
    return np.einsum("ji...,j...->i...", x, y)


def parse_functions(f, g, gy, fy, gx, fx, x0, ta, tb, m, w, numx, numy):

    """Merges the functions and their Jacobians in a single unified ODE with
    boundary conditions.
    """

    def ode(t, z):

        # Parse the variables
        x, y, l = np.split(z, np.cumsum([numx, numy]))

        # Evaluate functions
        F   = f(x,y,t)
        G   = g(x,y,t)
        Gy  = gy(x,y,t)
        Gx  = gx(x,y,t)
        Fy  = fy(x,y,t)
        Fx  = fx(x,y,t)

        dxdt = F
        dydt = - m * (vmp(Gy, G) + vmp(Fy, l))
        dldt =  np.zeros_like(- vmp(Gx,G) - vmp(Fx,l))

        return np.concatenate((dxdt,dydt,dldt))

    def bnd(za, zb):

        # Memory inefficient way of making compatable with the functions.
        z = np.vstack((za,zb)).T
        t = np.array([ta,tb])

        # Parse the variables
        x, y, l = np.split(z, np.cumsum([numx, numy]))

        # Evaluate functions
        F   = f(x,y,t)
        G   = g(x,y,t)
        Gy  = gy(x,y,t)
        Gx  = gx(x,y,t)
        Fy  = fy(x,y,t)
        Fx  = fx(x,y,t)

        bx = x[:,0]-x0
        by = - m * (vmp(Gy, G) + vmp(Fy, l))[:,0]
        bl = (1 - w) * l[:,1] - w * vmp(Gx,G)[:,1]

        return np.concatenate((bx,by,bl))

    return ode, bnd


def dae_solver_one(f, g, x, y, t, x0, m, w, gy=None, fy=None, gx=None, fx=None, *args, **kwargs):

    """Solve a differential algebraic equation using the optimal control
    formulation.

    This functional solves a system of differential alegraic equations (DAEs),
    order 1, by reformulating the solution as an optimal control problem. The
    system of DAEs should be in Hessenberg form, that is:

        dx/dt = f(x,y,t)    (1)
        g(x,y,t)=0          (2)

    Here t is a 1-dimensional independent variable, x is an nx-dimensional
    variable known are the differential variables and y is an ny-dimensional
    variable known as the algebraic variables.

    Parameters
    ----------
    f : callable
        Right-hand side of equation (1). The calling signature is ''f(x,y,t)'',
        where x, y and t are all ndarrays: ''t'' with shape (m,), ''x'' with
        shape (nx, m) and ''y'' with shape (ny, m). Returns a 1-dimensional
        array of length nx.
    g : callable
        Right-hand side of equation (2). The calling signature is ''g(x,y,t)'',
        where x, y and t are all ndarrays: ''t'' with shape (m,), ''x'' with
        shape (nx, m) and ''y'' with shape (ny, m). Returns a 1-dimensional
        array of length ny.
    x : array_like, shape (nx, m)
        Intial guess for the differential variable values at the mesh nodes, the
        i-th column corresponds to t[i].
    y : array_like, shape (ny, m)
        Intial guess for the algebraic variable values at the mesh nodes, the
        i-th column corresponds to t[i].
    t : array_like, shape (m,)
        Initial mesh, a strictly increasing sequence of real numbers with
        ''t[0]=a'' and ''t[-1]=b''.
    x0 : array_like, shape (nx,)
        Intial conditions for the differential variables.
    m : scalar
        Parameter value for the gradient descent.
    w : scalar
        Relative weighting for the optimal control.
    gy : callable
        Jacobian of f with respects to the algebraic variables (y). The calling
        signature is ''gy(x,y,t)'' where x, y and t are all ndarrays: ''t'' with
        shape (m,), ''x'' with shape (nx, m) and ''y'' with shape (ny, m).
        Returns a 2-dimensional array with shape (ny,ny).
    gx : callable
        Jacobian of f with respects to the algebraic variables (y). The calling
        signature is ''gx(x,y,t)'' where x, y and t are all ndarrays: ''t'' with
        shape (m,), ''x'' with shape (nx, m) and ''y'' with shape (ny, m).
        Returns a 2-dimensional array with shape (ny,nx).
    fy : callable
        Jacobian of f with respects to the algebraic variables (y). The calling
        signature is ''fy(x,y,t)'' where x, y and t are all ndarrays: ''t'' with
        shape (m,), ''x'' with shape (nx, m) and ''y'' with shape (ny, m).
        Returns a 2-dimensional array with shape (nx,ny).
    fx : callable
        Jacobian of f with respects to the algebraic variables (y). The calling
        signature is ''fx(x,y,t)'' where x, y and t are all ndarrays: ''t'' with
        shape (m,), ''x'' with shape (nx, m) and ''y'' with shape (ny, m).
        Returns a 2-dimensional array with shape (nx,nx).

    Returns
    -------
    Bunch object with the following fields defined:
    sol : PPoly
        Found solution for y as `scipy.interpolate.PPoly` instance, a C1
        continuous cubic spline.
    p : ndarray or None, shape (k,)
        Found parameters. None, if the parameters were not present in the
        problem.
    x : ndarray, shape (m,)
        Nodes of the final mesh.
    y : ndarray, shape (n, m)
        Solution values at the mesh nodes.
    yp : ndarray, shape (n, m)
        Solution derivatives at the mesh nodes.
    rms_residuals : ndarray, shape (m - 1,)
        RMS values of the relative residuals over each mesh interval (see the
        description of `tol` parameter).
    niter : int
        Number of completed iterations.
    status : int
        Reason for algorithm termination:
            * 0: The algorithm converged to the desired accuracy.
            * 1: The maximum number of mesh nodes is exceeded.
            * 2: A singular Jacobian encountered when solving the collocation
              system.
    """

    # Ensure that the inputs are numpy ndarrays

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    t = np.asarray(t)

    # Check the validity of the mesh and function inputs.

    if x.ndim != 2:
        raise ValueError("x should be two dimensional...")
    if y.ndim != 2:
        raise ValueError("y should be two dimensional...")
    if t.ndim != 1:
        raise ValueError("t should be one dimensional...")
    if x0.ndim != 1:
        raise ValueError("x0 should be one dimensional...")

    # Infer the dimensions from the input shapes.

    m  = t.shape[0]
    nx = x.shape[0]
    ny = y.shape[0]

    # Check the intial conditions for differential variables.

    if x0.shape[0] != nx:
        raise ValueError("Mismatch between the number of differential variables\
                          and the number of intial values specified...")

    # Parse the Jacobians

    fx, fy, gx, gy = parse_jacobians(f, g, fx, fy, gx, gy)

    # Infer the t range

    ta = t[0]
    tb = t[-1]

    # Parse the functions

    ode, bnd = parse_functions(f, g, gy, fy, gx, fx, x0, ta, tb, m, w, nx, ny)

    # Generate the inital function surface using a zero start more lambda.

    l = np.zeros_like(x)
    z = np.concatenate((x,y,l))

    solution = solve_bvp(ode, bnd, t, z, *args, **kwargs)

    return solution
