from copy import copy
import numpy as np
import pdb

def estimate_jacobian(func, j=0):

    """Estimates the jacobians by forward differencing.

    Parameters
    ----------
    func : callable
        A function for which the jacobian is being evaluated. func should return
        a nf-dimensional numpy ndarray.
    j : int
        The index in *args of the paramter for which the Jacobian is being
        determined.

    Returns
    -------
    jac : callable
        Returns a function which upon being called returns the jacobian of the
        function with respects to the argument specified. The function reaturn
        itself returns a numpy ndarray of shape (nf, nx, m).

    Notes
    -----
    This function is heavily influence by the same function in the bvp_solve
    package.
    """

    def jac(*args, f0=None):
        args = list(args)
        x = args[j]

        if x.ndim == 3:
            raise("Dimensions of the independent variable should be 3...")

        eps = np.finfo(float).eps
        nx, m = x.shape
        if f0 == None:
            f0 = func(*args)

        nf = f0.shape[0]
        xdtype = x.dtype
        dfdx = np.empty((nf, nx, m), dtype=xdtype)

        hx = eps**0.5 * (1 + np.abs(args[j]))

        for i in range(nx):
            x_new = x.copy()
            x_new[i] += hx[i]
            hi = x_new[i] - x[i]
            args_new = copy(args)
            args_new[j] = x_new
            f_new = func(*args_new)
            dfdx[:,i,:] = (f_new-f0)/hi

        return dfdx
    return jac

def estimate_jacobians_many(func, args=[0]):

    """Loops through and calculates the Jacobian for the arguments specified.
    """
    jac_stack = []
    for arg in args:
        jac_stack.append(estimate_jacobian(func, arg))

    return jac_stack
