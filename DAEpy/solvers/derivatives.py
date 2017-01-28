from copy import copy
import numpy as np

def estimate_derivative(func, j=0):

    """Estimates the jacobians by forward differencing.

    Parameters
    ----------
    func : callable
        A function for which the derivative is being evaluated. func should
        return a scalar value.
    j : int
        The index in *args of the paramter for which the derivate is being
        determined.

    Returns
    -------
    deriv : callable
        Returns a function which upon being called returns the derivative of the
        function with respects to the argument specified. The function reaturn
        itself returns a numpy ndarray of shape (nx, m).
    """

    def deriv(*args):
        args = list(args)
        x = args[j]

        eps = np.finfo(np.float64).eps
        nx, m = x.shape

        f0 = func(*args)

        xdtype = x.dtype
        dfdx = np.empty((nx, m), dtype=xdtype)
        hx = eps**0.5 * (1 + np.abs(x))

        for i in range(nx):
            x_new = x.copy()
            x_new[i] += hx[i]
            hi = x_new[i] - x[i]
            args_new = copy(args)
            args_new[j] = x_new
            f_new = func(*args_new)
            dfdx[i,:] = (f_new-f0)/hi

        return dfdx
    return deriv
