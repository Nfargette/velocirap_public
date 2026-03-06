
""" ----------------------------------------------------------------------------
Useful numerical analysis and algebra related functions
---------------------------------------------------------------------------- """

import numpy            as np
from scipy.interpolate  import splrep, splantider, splev
from scipy.stats        import multivariate_normal

def interpolate_func(x, y, der=0, k=3, prim_cond=None, *args, **kwargs): 
    """
    Routine returning an interpolation function of (x, y) 
    for a given B-spline order k. 
    A derivative order can be specified by the value der < k 
    (if der=-1, returns an antiderivative).
    Written by Pierre Houdayer.

    Parameters
    ----------
    x : array_like, shape (N, )
        x values on which to integrate.
    y : array_like, shape (N, )
        y values to integrate.
    der : INT, optional
        Order of the derivative. 
        The default is 0.
    k : INT, optional
        Degree of the B-splines used to compute the integral. 
        The default is 3.
    s : float, optional
        Smoothing parameter. 
        The default is 0.
    prim_cond : array_like, shape (2, ), optional
        Conditions to specify the constant to add to the
        primitive function if der = -1. The first value 
        is an integer i, such that F(x[i]) = second value.
        The default is None, which correspond to F(x[0]) = 0.

    Returns
    -------
    func : function(x_eval)
        Interpolation function of x_eval.

    """
    # Valid interval (removing nans)
    VALID = ~(
        np.any(np.isnan(y), axis=0) | 
        np.isnan(x) | 
        np.hstack((x[1:] <= x[:-1], False))
    )
    xV = x[VALID]
    yV = y[..., VALID]
        
    # Handle multi-dimensional inputs
    if len(yV.shape) > 1 :
        def Func(x_eval) :
            return np.array([
                interpolate_func(
                    xV, y_i, 
                    der=der, k=k, prim_cond=prim_cond, *args, **kwargs
                )(x_eval) for y_i in yV
            ])    
        return Func
    
    # Find the B-spline representation 
    tck = splrep(xV, yV, k=k, *args, **kwargs)

    #if der is a list, return the list of derivative functions
    if not type(der) is int :
        def func(x_eval) :
            if 0 not in np.asarray(x_eval).shape :
                return [splev(x_eval, tck, der=d) for d in der]
            else : 
                return np.array([]).reshape((len(der), 0))
        return func
    
    # check validity of der
    if not -2 < der < k : 
        raise ValueError(
            f"""Derivative order should be either -1 (antiderivative)
            or 0 <= der < k={k} (derivative). Current value is {der}."""
        )
    
    #interpolate and compute the "der" derivative
    # der = 0 -> interpolation
    # der = 1 -> first derivative, etc.
    if der >= 0 :
        def func(x_eval, der_local=der) :
            if 0 not in np.asarray(x_eval).shape :
                return splev(x_eval, tck, der=der_local)
            else : 
                return np.array([])
        return func
    
    # if der = -1, compute the antiderivative (integral)
    else :
        tck_antider = splantider(tck)
        cnst = 0.0
        if prim_cond is not None :
            cnst = prim_cond[1] - splev(xV[prim_cond[0]], tck_antider)
        def func(x_eval) :
            return splev(x_eval, tck_antider) + cnst
        return func


def quaternion_rotation_matrix(Q): 
    """
    Covert a quaternion into a full three-dimensional rotation matrix.
 
    Input
    :param Q: A 4 element array representing the quaternion (qr,qi,qj,qk) 
 
    Output
    :return: A 3x3 element matrix representing the full 3D rotation matrix. 
             This rotation matrix converts a point in the local reference 
             frame to a point in the global reference frame.
    """
    # Extract the values from Q
    qr= Q[0]
    qi= Q[1]
    qj= Q[2]
    qk= Q[3]
     
    # Diagonal
    r00 = 1 - 2 * (qj**2 + qk **2)
    r11 = 1 - 2 * (qi**2 + qk **2)
    r22 = 1 - 2 * (qi**2 + qj **2)

    # Off-diagonal
    r01 = 2 * (qi * qj - qk * qr)
    r02 = 2 * (qi * qk + qj * qr)
    r10 = 2 * (qi * qj + qk * qr)
    r12 = 2 * (qj * qk - qi * qr)
    r20 = 2 * (qi * qk - qj * qr)
    r21 = 2 * (qj * qk + qi * qr)
     
    # 3x3 rotation matrix
    rot_matrix = np.array(
        [[r00, r01, r02],
        [r10, r11, r12],
        [r20, r21, r22]]
    )
                            
    return rot_matrix


def integrate(x, n, metric, axis): 
    """
    Compute the integral of x over the n-dimensions defined by axis,
    with the metric given by metric, and normalized by the integral of n
    over the same dimensions.

    Parameters
    ----------
    x : array_like
        The quantity to be integrated.
    n : array_like
        The density of the quantity to be integrated.
    metric : array_like
        The metric of the space.
    axis : tuple of int
        The dimensions over which to integrate.

    Returns
    -------
    X : array_like
        The integrated quantity, normalized by the integral of n.
    """
    
    
    #axis to consider
    is_in_axis = [id in axis for id in {0, 1, 2}] 
    
    #volume elements
    dGamma = np.prod(metric[is_in_axis], axis=0)    
    
    #if x is 1D, integrate along axis
    if np.all(x == 1):    
        X = np.nansum(x * n * dGamma, axis=axis) 
        
    # else integrate along the last three axis
    else :
        D = max(np.array(x).ndim, 3)
        ax = tuple(range(D-3, D))
        X = np.nansum(x * n * dGamma, axis=ax) / (
            np.nansum(n * dGamma, axis=axis)
        )
        
    return X


def multivariate_gaussian(u, mu, sigma):    
   
    """Multivariate normal distribution.

    Parameters
    ----------
    u : array_like
        3D grid of coordinates
    mu : array_like
        mean of the distribution
    sigma : array_like
        covariance matrix of the distribution

    Returns
    -------
    pdf : array_like
        Probability density function evaluated at u."""
    
    #Multivariate normal
    Gauss0 = multivariate_normal(mu, sigma)
    
    #flatten the grid
    u_flat = np.array([u[0].flatten(), u[1].flatten(), u[2].flatten()])
    
    #define the pdf function
    Gauss = lambda x: Gauss0.pdf(x)
    
    #compute the pdf
    pdf = Gauss(u_flat.T).reshape(u.shape[1:])    

    return pdf
    