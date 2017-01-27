from .jacobians import estimate_jacobian, estimate_jacobians_many

from scipy.integrate import solve_bvp

import numpy as np
import pdb

def parse_jacobians(f, h, g, fx, fy, fz, gx, gy, hy):

    """Parse the Jacobians and evalate numerically if necessary.
    """

    if fx == None or fy == None or fz == None:
        fx, fy, fz = estimate_jacobians_many(f, [0,1,2])

    if gx == None or gy ==None:
        gx, gy = estimate_jacobians_many(g, [0,1])

    if hy == None:
        hy = estimate_jacobian(h, 0)

    return fx, fy, fz, gx, gy, hy

def vmp(x, y):
    """
    Returns the vector matrix product. A_T.x
    """
    return np.einsum("ji...,j...->i...", x, y)


def parse_functions(f, h, g, fx, fy, fz, gx, gy, hy, x0, y0, ta, tb, mu, w, nx, ny, nz):

    """Merges the functions and their Jacobians in a single unified ODE with
    boundary conditions.
    """

    def ode(t, z):

        # Parse the variables
        x, y, z, l1, l2 = np.split(z, np.cumsum([nx, ny, nz, nx]))

        # Evaluate functions
        F   = f(x,y,z,t)
        G   = g(x,y,t)
        H   = h(y,t)
        Fx  = fx(x,y,z,t)
        Fy  = fy(x,y,z,t)
        Fz  = fz(x,y,z,t)
        Gx  = gx(x,y,t)
        Gy  = gy(x,y,t)
        Hy  = hy(y, t)

        dxdt = F
        dydt = G
        dzdt  = - mu * vmp(Fz, l1)
        dl1dt = - vmp(Fx,l1) - vmp(Gx,l2)
        dl2dt = - vmp(Hy, H) - vmp(Fy, l1) - vmp(Gy, l2)

        return np.concatenate((dxdt,dydt,dzdt,dl1dt,dl2dt))

    def bnd(za, zb):

        # Memory inefficient way of making compatable with the functions.
        z = np.vstack((za,zb)).T
        t = np.array([ta,tb])

        # Parse the variables
        x, y, z, l1, l2 = np.split(z, np.cumsum([nx, ny, nz, nx]))

        # Evaluate functions
        F   = f(x,y,z,t)
        G   = g(x,y,t)
        H   = h(y,t)
        Fx  = fx(x,y,z,t)
        Fy  = fy(x,y,z,t)
        Fz  = fz(x,y,z,t)
        Gx  = gx(x,y,t)
        Gy  = gy(x,y,t)
        Hy  = hy(y, t)


        bx = x[:,0] - x0
        by = y[:,0] - y0
        bz = vmp(Fz, l1)[:,0]
        bl1 = l1[:,-1] - np.zeros_like(l1[:, -1])
        bl2 = l2[:,-1] - vmp(Hy,H)[:,-1]

        return np.concatenate((bx,by,bz,bl1,bl2))

    return ode, bnd


def dae_solver_three(f, g, h, x, y, z, t, x0, y0, mu, w, fx=None, fy=None, fz=None, gx=None, gy=None, hy=None, *args, **kwargs):

    """Solve a differential algebraic equation using the optimal control
    formulation.

    This functional solves a system of differential alegraic equations (DAEs),
    order 1, by reformulating the solution as an optimal control problem. The
    system of DAEs should be in Hessenberg form, that is:

        dx/dt = f(x,y,z,t)    (1)
        dy/dt = g(x,y,t)      (2)
        h(y,t)=0              (3)

    Here t is a 1-dimensional independent variable, x is an nx-dimensional
    variable known are the differential variables and y is an ny-dimensional
    variable known as the algebraic variables.

    Parameters
    ----------
    f : callable
        Right-hand side of equation (1). The calling signature is ''f(x,y,z,t)'', where x, y, z and t are all ndarrays: ''t'' with shape(m,), ''x'' with shape (nx, m), ''y'' with shape (ny, m) and ''z'' with shape (nz, m). Returns a 1-dimensional array of length nx.
    g : callable
        Right-hand side of equation (2). The calling signature is ''g(x,y,t)'',
        where x, y and t are all ndarrays: ''t'' with shape (m,), ''x'' with
        shape (nx, m) and ''y'' woth shape(ny, m). Returns an 1-dimensional array with length ny.
    h  : callable
        Right-hand side of equation (2). The calling signature is ''h(y,t)'',
        where x, y and t are all ndarrays: ''t'' with shape (m,) and ''y'' with
        shape (ny, m).
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
    y0 : array_like, shape (ny,)
        Intial conditions for the differential variables.
    m : scalar
        Parameter value for the gradient descent.
    w : scalar
        Relative weighting for the optimal control.

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
    z = np.asarray(z, dtype=float)
    t = np.asarray(t)

    # Check the validity of the mesh and function inputs.

    if x.ndim != 2:
        raise ValueError("x should be two dimensional...")
    if y.ndim != 2:
        raise ValueError("y should be two dimensional...")
    if z.ndim != 2:
        raise ValueError("y should be two dimensional...")
    if t.ndim != 1:
        raise ValueError("t should be one dimensional...")
    if x0.ndim != 1:
        raise ValueError("x0 should be one dimensional...")
    if y0.ndim != 1:
        raise ValueError("x0 should be one dimensional...")

    # Infer the dimensions from the input shapes.

    m  = t.shape[0]
    nx = x.shape[0]
    ny = y.shape[0]
    nz = z.shape[0]

    # Check the intial conditions for differential variables.

    if x0.shape[0] != nx:
        raise ValueError("Mismatch between the number of differential variables and the number of intial values specified...")
    if y0.shape[0] != ny:
        raise ValueError("Mismatch between the number of differential variables and the number of intial values specified...")

    # Parse the Jacobians

    fx, fy, fz, gx, gy, hy = parse_jacobians(f, h, g, fx, fy, fz, gx, gy, hy)

    # Infer the t range

    ta = t[0]
    tb = t[-1]

    # Parse the functions

    ode, bnd = parse_functions(f, h, g, fx, fy, fz, gx, gy, hy, x0, y0, ta, tb, mu, w, nx, ny, nz)

    # Generate the inital function surface using a zero start for lambda.
    # The concatenate function involves shuffling memory around to form a
    # contiguous block of memory which is very inefficient.

    l1 = np.zeros_like(x)
    l2 = np.zeros_like(y)
    z = np.concatenate((x,y,z,l1,l2))

    solution = solve_bvp(ode, bnd, t, z, *args, **kwargs)

    return solution
