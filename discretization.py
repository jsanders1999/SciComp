import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

def A(N, eps):
    """
    A function that resturns the discretization matrix for a system with N nodes and parameter epsilon

    Parameters
    ----------
    N : int
        The number of grid points 
    eps : float64
        The epsilon parameter from the differential equation

    Returns
    -------
    sparse array of floats (format : CSC, dim: (N-2)*(N-2))
        The discretization matrix
    """

    h = 1/N
    L = 1/h**2*( 2*sp.sparse.eye(N-2, format='csc') - sp.sparse.eye(N-2, k=1, format='csc') - sp.sparse.eye(N-2, k=-1, format='csc') )
    D = 1/h*( sp.sparse.eye(N-2, format='csc') - sp.sparse.eye(N-2, k=-1, format='csc') )
    return eps*L + D


def f(N, eps, BC = [0,0]):
    """
    A function that resturns the discretization vector for a system with N nodes and parameter epsilon.

    Parameters
    ----------
    N : int
        The number of grid points 
    eps : float64
        The epsilon parameter from the differential equation

    Returns
    -------
    array of floats (dim:(N-2))
        The discretization vector
    """
    h = 1/N
    res = np.zeros(N-2)
    res[0]  +=  BC[0]*(eps/h**2+1/h)
    res[-1] +=  BC[1]*(eps/h**2)
    return res

def simpleSolve(N, eps, BC = [0,0]):
    """
    A function that solves A*u=f by inverting the sparse matrix A.

    Parameters
    ----------
    N : int
        The number of grid points 
    eps : float64
        The epsilon parameter from the differential equation
    BC : array of float64 (dim: 2)
        The boundary conditions of the differential equation
        

    Returns
    -------
    array of floats (dim: N)
        The numerical solution of the differential equation, with boundary values included
    """
    u = np.zeros(N)
    u[0] = BC[0]
    u[-1] = BC[1]
    u[1:-1] = sp.sparse.linalg.inv(A(N,eps)).dot(f(N,eps,BC))
    return u