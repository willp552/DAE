from DAEpy.solvers.derivatives import estimate_derivative
from scipy.integrate import solve_bvp

import numpy as np
import pdb

def construct_hamiltonian(L, f):

    """Constructs the hamiltonian from the Lagrangian and the differential
    equation.
    """
    def hamiltonian(x, u, l, t):

        # Evaluate functions
        Le = L(x, u, t)
        Fe = f(x, u, t)

        # Evaulate the hamiltonian
        H = Le + np.einsum('i...,i...->...', l, Fe)

        return H

    return hamiltonian

def parse_system(H, Hx, Hu, f, x0, ta, tb, m, phi, nx, nu):

    """Constructs the ode and boundary equations from the Hamiltonians.
    """

    def ode(t, z):

        # Parse the variables.
        x, u, l = np.split(z, np.cumsum([nx, nu]))

        # Evaluate the functions.
        He  = H(x, u, l, t)
        Hxe = Hx(x, u, l, t)
        Hue = Hu(x, u, l, t)

        dxdt = f(x, u, t)
        dydt = - m * Hue
        dldt = - Hxe

        return np.concatenate((dxdt,dydt,dldt))

    def bnd(za, zb):

        # Memory inefficient way of making compatable with the functions.
        z = np.vstack((za,zb)).T
        t = np.array([ta,tb])

        # Parse the variables.
        x, u, l = np.split(z, np.cumsum([nx, nu]))

        # Evaluate the functions.
        He  = H(x, u, l, t)
        Hxe = Hx(x, u, l, t)
        Hue = Hu(x, u, l, t)

        if phi:
            P = phi(x, t)[:,-1]
        else:
            P = np.zeros_like(l[:,-1])

        bx = x[:,0] - x0
        bu = Hue[:,0]
        bl = l[:,-1] + P

        return np.concatenate((bx,bu,bl))
    return ode, bnd

def ocp_solver(L, f, x, u, t, x0, m, phi=None, *args, **kwargs):

    """Solve an optimal control problem of the form:

        min J = phi[x(tf),tf] + ∫L[x(t),u(t),t]dt (1)

    s.t         dx/dt = f(x(t),u(t),t)            (2)

    Here t is a 1-dimensional independent variable, x is an nx-dimensional
    variable and u is nu-dimensional variable known as the control variable.

    Parameters
    ----------

    L : callable

        Integrand of equation (1), representing the langrangian of the system.
        The calling signature is ''L(x,u,t)'' where x, y and t are all ndarrays:
        ''t'' with shape (m,), ''x'' with shape (nx, m) and ''u'' with shape
        (nu, m). Returns a one dimensional array with length m.
    f : callable
        Right-hand side of equation (2). The calling signature is ''f(x,u,t)''
        where x, y and t are all ndarrays: ''t'' with shape (m,), ''x'' with
        shape (nx, m) and ''u'' with shape(nu, m). Returns a two dimensional
        array with shape (nx, m).
    x : array_like, shape (nx, m)
        Intial guess for the state variable values at the mesh nodes, the i-th
        column corresponds to t[i].
    u : array_like, shape (ny, m)
        Intial guess for the control variable values at the mesh nodes, the
        i-th column corresponds to t[i].
    t : array_like, shape (m,)
        Initial mesh, a strictly increasing sequence of real numbers with
        ''t[0]=a'' and ''t[-1]=b''.
    x0 : array_like, shape (nx,)
        Intial conditions for the state variables.
    m : scalar
        Parameter value for the gradient descent.
    phi: callable (Optional)
        Penalty function for the state variables at the final time. Calling
        signature ''phi(x,t)''.

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

    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(u, dtype=np.float64)
    t = np.asarray(t, dtype=np.float64)

    # Check the validity of the mesh and function inputs.

    if x.ndim != 2:
        raise ValueError("x should be two dimensional...")
    if u.ndim != 2:
        raise ValueError("u should be two dimensional...")
    if t.ndim != 1:
        raise ValueError("t should be one dimensional...")
    if x0.ndim != 1:
        raise ValueError("x0 should be one dimensional...")

    # Infer the dimensions from the input shapes.

    m  = t.shape[0]
    nx = x.shape[0]
    nu = y.shape[0]

    if x0.shape[0] != nx:
        raise ValueError("Mismatch between the number of differential variables\
                          and the number of intial values specified...")
    # Infer the t range

    ta = t[0]
    tb = t[-1]

    # Construct the hamiltonian.

    H = construct_hamiltonian(L, f)

    # Parse the Jacobian functions.

    Hx = estimate_derivative(H, 0)
    Hu = estimate_derivative(H, 1)

    # Parse the ode and boundary conditions.

    ode, bnd = parse_system(H, Hx, Hu, f, x0, ta, tb, m, phi, nx, nu)

    # Generate the inital function surface using a zero start more lambda.

    l = np.zeros_like(x)
    z = np.concatenate((x,u,l))

    solution = solve_bvp(ode, bnd, t, z, *args, **kwargs)

    return solution
