import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

###Matrix Creations
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
    L = 1/h**2*( 2*sp.sparse.eye(N-1, format='csc') - sp.sparse.eye(N-1, k=1, format='csc') - sp.sparse.eye(N-1, k=-1, format='csc') )
    D = 1/h*( sp.sparse.eye(N-1, format='csc') - sp.sparse.eye(N-1, k=-1, format='csc') )
    return eps*L + D

def A_Full(N,eps):
    h   = 1/N
    D   = np.diag(np.ones(N-1)*((2*eps)/(h**2)) + 1/h,0)
    E   = np.diag(np.ones(N-2)*((-eps/(h**2) - 1/h)), -1)
    F   = np.diag(np.ones(N-2)* (-eps/(h**2)), 1)
    A   = D + E + F
    return A, D, E, F

def Jacobi(N,eps):
    A_Full, D, E, F = A(N,eps)
    B_Jac           = (-E) + (-F)
    return B_Jac

def Jacobi_Iteration(N,eps):
    b       = f(N,eps, BC = [1,0])
    #print("f = ", b, "\n")
    A,D,E,F = A_Full(N,eps)
    #print("A = ", A, "\n")
    #print("D = ", D, "\n")
    #print("E = ", E, "\n")
    #print("F = ", F, "\n")
    u       = np.zeros(N-1)
    #print("u = ", u, "\n")
    z       = np.zeros(N-1)
    #print("z = ", z, "\n")
    j       = 0
    while j<30:
        for i in range(N-1):
            z[i] = (b[i] - A[i,:i]@u[:i] - A[i,i+1:]@u[i+1:])/A[i,i]
            #print("z = \n", z, "\n")
            #print("b[i] = \n", b[i], "\n")
            #print("A[i, 0:(i-1)] = \n", A[i, 0:(i-1)], "\n")
            #print("u[0:(i-1)] = \n", u[0:(i-1)], "\n")
            #print("A[0,i:(N-1)] = \n", A[0,i:(N-1)], "\n")
            #print("u[(i):(N-1)] = \n", u[(i):(N-1)], "\n")
            #print("A[i,i] = \n", A[i,i], "\n")
        j = j+1
        u = z
    return A, b, z, u

def Gauss_Seidel(N,eps,Form):
    if form == "Backward":
        A_Full, D, E, F = A(N,eps)
        B_GS = np.linalg.inv(np.identity(N))
    if form == "Forward":
        A_Full, D, E, F = A(N,eps)
        B_GS = np.linalg.inv(np.identity(N))
    else:                                       #Symmetric
        B_GS = 1
    return B_GS
    

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
    res = np.zeros(N-1)
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
    u = np.zeros(N+1)
    u[0] = BC[0]
    u[-1] = BC[1]
    u[1:-1] = sp.sparse.linalg.spsolve(A(N,eps), f(N,eps,BC)) #sp.sparse.linalg.inv(A(N,eps)).dot(f(N,eps,BC))
    return u

def refSol(N,eps):
    """
    A function that gives the values at the gricpoints of the solutions to our differential equation

    Parameters
    ----------
    N : int
        The number of grid points 
    eps : float64
        The epsilon parameter from the differential equation
        

    Returns
    -------
    array of floats (dim: N)
        The numerical solution of the differential equation, with boundary values included
    """
    x   = np.linspace(0,1,N+1)
    y   = (np.exp(x/eps) - np.exp(1/eps))/(1-np.exp(1/eps))
    return x,y


#def u(x,eps):
#    return (np.exp(x/eps - np.exp(1/eps)))/(1 - np.exp(1/eps))
