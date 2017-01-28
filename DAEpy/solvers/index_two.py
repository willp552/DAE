from .jacobians import estimate_jacobian, estimate_jacobians_many

from scipy.integrate import solve_bvp

import numpy as np
import pdb

def parse_jacobians(f, g, fx, fy, gx):

    """Parse the Jacobians and evalate numerically if necessary.
    """

    if fx == None or fy == None:
        fx, fy = estimate_jacobians_many(f, [0,1])

    if gx == None:
        gx = estimate_jacobian(g, 0)

    return fx, fy, gx

def vmp(x, y):
    """
    Returns the vector matrix product. A_T.x
    """
    return np.einsum("ji...,j...->i...", x, y)


def parse_functions(f, g, fy, gx, fx, x0, ta, tb, mu, w, numx, numy):

    """Merges the functions and their Jacobians in a single unified ODE with
    boundary conditions.
    """

    def ode(t, z):

        # Parse the variables
        x, y, l = np.split(z, np.cumsum([numx, numy]))

        # Evaluate functions
        F   = f(x,y,t)
        G   = g(x,t)
        Gx  = gx(x,t)
        Fy  = fy(x,y,t)
        Fx  = fx(x,y,t)

        dxdt = F
        dydt = - mu * vmp(Fy, l)
        dldt = - vmp(Gx,G) - vmp(Fx,l)
        return np.concatenate((dxdt,dydt,dldt))

    def bnd(za, zb):

        # Memory inefficient way of making compatable with the functions.
        z = np.vstack((za,zb)).T
        t = np.array([ta,tb])

        # Parse the variables
        x, y, l = np.split(z, np.cumsum([numx, numy]))

        # Evaluate functions
        F   = f(x,y,t)
        G   = g(x,t)
        Gx  = gx(x,t)
        Fy  = fy(x,y,t)
        Fx  = fx(x,y,t)

        bx = x[:,0]- x0
        by = vmp(Fy, l)[:,0]
        bl = l[:,-1] - vmp(Gx,G)[:,-1]

        return np.concatenate((bx,by,bl))

    return ode, bnd


def dae_solver_two(f, g, x, y, t, x0, mu, w, gx=None, fy=None, fx=None, *args, **kwargs):

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
        Right-hand side of equation (2). The calling signature is ''g(x,t)'',
        where x, y and t are all ndarrays: ''t'' with shape (m,) and ''x'' with
        shape (nx, m).
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

    fx, fy, gx = parse_jacobians(f, g, fx, fy, gx)

    # Infer the t range

    ta = t[0]
    tb = t[-1]

    # Parse the functions

    ode, bnd = parse_functions(f, g, fy, gx, fx, x0, ta, tb, mu, w, nx, ny)

    # Generate the inital function surface using a zero start more lambda.

    l = np.zeros_like(x)
    z = np.concatenate((x,y,l))

    solution = solve_bvp(ode, bnd, t, z, *args, **kwargs)

    return solution
