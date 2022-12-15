import numpy as np
import scipy as sp
import discretization

def B_Jacobi(N, eps):
    """
    A function that resturns the Jaboci residual iteration matrix given N and eps.

    Parameters
    ----------
    N : int
        The number of grid points 
    eps : float64
        The epsilon parameter from the differential equation

    Returns
    -------
    array of floats (dim: (N-1)*(N-1))
        The residual iteration matrix of the Jabobi method
    """
    A_Full, D, E, F, E_hat, F_hat   = discretization.A_Full(N,eps)
    B_Jac                           = E_hat + F_hat
    return B_Jac

def B_Gauss_Seidel(N, eps, form):
    """
    A function that resturns the Gauss Seidel residual iteration matrix given N and eps.

    Parameters
    ----------
    N : int
        The number of grid points 
    eps : float64
        The epsilon parameter from the differential equation
    form : string, "Forward" or "Backward"
        Whether to implement "Forward" or "Backward" Gauss Seidel. 

    Returns
    -------
    array of floats (dim: (N-1)*(N-1))
        The residual iteration matrix of the GS method
    """
    if form == "Backward":
        A_Full, D, E, F = discretization.A_Full(N,eps)
        B_GS = np.linalg.inv(np.identity(N)) #WIP
    if form == "Forward":
        A_Full, D, E, F = discretization.A_Full(N,eps)
        B_GS = np.linalg.inv(np.identity(N)) #WIP
    else:                                       #Symmetric
        B_GS = 1
    return B_GS

def Jacobi_Iteration(N, eps, tol):
    """
    A function that solves Ax=f using a Jabobi iteration method. A is computed using N and eps.
    tol determines the stopping criteria

    Parameters
    ----------
    N : int
        The number of grid points 
    eps : float64
        The epsilon parameter from the differential equation
    tol : float64
        The stopping criterion for the residual. 

    Returns
    -------

    """
    MAX_IT  = int(1e6)
    b       = discretization.f(N,eps, BC = [1,0])
    A,D,E,F,E_hat,F_hat = discretization.A_Full(N,eps)
    u       = np.zeros(N-1)
    z       = np.zeros(N-1)
    B       = B_Jacobi(N, eps)
    r       = b                     #Starting residual for u = np.zeros(N-1)
    A       = A.toarray() #make A not sparse anymore :'(
    for j in range(MAX_IT):
        for i in range(N-1):
            z[i] = (b[i] - A[i,:i].dot(u[:i]) - A[i,i+1:].dot(u[i+1:]))/A[i,i]
        u = z
        r = B.dot(r)
        if np.linalg.norm(r)/np.linalg.norm(b) <= tol:
            return u, r, j
    return u, r, MAX_IT

def Gauss_Seidel_Iteration_Forward(N, eps, tol):
    """
    A function that solves Ax=f using a Forward Gauss Seidel method. A is computed using N and eps.
    tol determines the stopping criteria.

    Parameters
    ----------
    N : int
        The number of grid points 
    eps : float64
        The epsilon parameter from the differential equation
    tol : float64
        The stopping criterion for the residual. 

    Returns
    -------

    """
    MAX_IT  = int(1e3)
    b       = discretization.f(N,eps, BC = [1,0])
    A,D,E,F = discretization.A_Full(N,eps)
    u       = np.zeros(N-1)
    B       = B_GS(N, eps, "Forward")
    r       = b                     #Starting residual for u = np.zeros(N-1)
    for j in range(MAX_IT):
        for i in range(N-1):
            u[i] = (b[i] - A[i,:i]@u[:i] - A[i,i+1:]@u[i+1:])/A[i,i]
        r = B@r
        if np.linalg.norm(r)/np.linalg.norm(b) <= tol:
            return u, r, j
    return u, r, MAX_IT