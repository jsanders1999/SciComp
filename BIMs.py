import numpy as np
import scipy as sp
import discretization
from tqdm import tqdm

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
    A_Full, D, E, F, E_hat, F_hat   = discretization.A_Full(N,eps)
    if form == "Forward":
        B_GS = np.linalg.inv((np.identity(N-1)-E_hat.toarray())).dot(F_hat.toarray()) 
    elif form == "Backward":
        B_GS = np.linalg.inv((np.identity(N-1)-F_hat.toarray())).dot(E_hat.toarray()) 
    else: #Symmetric
        B_GSF = np.linalg.inv((np.identity(N-1)-E_hat.toarray())).dot(F_hat.toarray())
        B_GSB = np.linalg.inv((np.identity(N-1)-F_hat.toarray())).dot(E_hat.toarray())
        B_GS  = B_GSB.dot(B_GSF)
    return B_GS

def Jacobi_Iteration(N, eps, tol, saveResiduals = False):
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
    MAX_IT  = N**2#int(1e6)
    b       = discretization.f(N,eps, BC = [1,0])
    A,D,E,F,E_hat,F_hat = discretization.A_Full(N,eps)
    A       = A.toarray() 
    u       = np.zeros(N-1)
    z       = np.zeros(N-1)
    #B       = B_Jacobi(N, eps)
    r       = b                     #Starting residual for u = np.zeros(N-1)
    if saveResiduals:
        res_arr = np.zeros((MAX_IT+1))
        res_arr[0] = np.linalg.norm(r)/np.linalg.norm(b)
    else:
        res_arr = None
    for j in tqdm(range(MAX_IT), desc = "Jacobi Iterations for N = {}".format(N)):
        for i in range(N-1):
            z[i] = (b[i] - A[i,:i]@u[:i] - A[i,i+1:]@u[i+1:])/A[i,i]
        u = z
        r = b - A@u
        if saveResiduals:
            res_arr[j+1] = np.linalg.norm(r)/np.linalg.norm(b)
        if np.linalg.norm(r)/np.linalg.norm(b) <= tol:
            return u, r, j, res_arr
    return u, r, MAX_IT, res_arr

def Gauss_Seidel_Iteration(N, eps, tol, form, saveResiduals = False):
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
    form : string, "Forward", "Backward" or "Symmetric"
        Whether to implement "Forward", "Backward" or "Symmetric" Gauss-Seidel. 
    Returns
    -------

    """
    A,D,E,F,E_hat,F_hat = discretization.A_Full(N,eps)
    b       = discretization.f(N,eps, BC = [1,0])
    r       = b                                         #Starting residual for u = np.zeros(N-1)
    u       = np.zeros(N-1)
    MAX_IT  = N**2 #int(1e5)
    A       = A.toarray()
    if saveResiduals:
        res_arr = np.zeros((MAX_IT+1))
        res_arr[0] = np.linalg.norm(r)/np.linalg.norm(b)
    else:
        res_arr = None                   
    #print("Starting b = ", b)
    #print("Starting r = ", r)
    if form == "Forward":
        #B       = B_Gauss_Seidel(N, eps, "Forward")
        for j in tqdm(range(MAX_IT), desc = "FGS iterations for N = {}".format(N)):
            for i in range(N-1):
                u[i] = (b[i] - A[i,:i]@u[:i] - A[i,i+1:]@u[i+1:])/A[i,i]
            r = b - A@u
            #print("B = ", B)
            #print("r = ", r)
            #print("b = ", b)
            if saveResiduals:
                res_arr[j+1] = np.linalg.norm(r)/np.linalg.norm(b)
            if np.linalg.norm(r)/np.linalg.norm(b) <= tol:
                return u, r, j, res_arr
        return u, r, MAX_IT, res_arr
    
    elif form == "Backward":
        #B       = B_Gauss_Seidel(N, eps, "Backward")
        for j in tqdm(range(MAX_IT), desc = "BGS iterations for N = {}".format(N)):
            for i in reversed(range(N-1)):
                u[i] = (b[i] - A[i,i+1:]@u[i+1:] - A[i,:i]@u[:i] )/A[i,i]       #Is this correct?
            r = b - A@u
            if saveResiduals:
                res_arr[j+1] = np.linalg.norm(r)/np.linalg.norm(b)
            if np.linalg.norm(r)/np.linalg.norm(b) <= tol:
                return u, r, j, res_arr
        return u, r, MAX_IT, res_arr
    
    elif form == "Symmetric":
        #B       = B_Gauss_Seidel(N, eps, "Symmetric")
        for j in tqdm(range(MAX_IT), desc = "SymGS iterations for N = {}".format(N)):
            for i in range(N-1):
                u[i] = (b[i] - A[i,:i]@u[:i] - A[i,i+1:]@u[i+1:])/A[i,i]
            for i in reversed(range(N-1)):
                u[i] = (b[i] - A[i,i+1:]@u[i+1:] - A[i,:i]@u[:i] )/A[i,i]
            r = b - A@u
            #print("B = ", B)
            #print("r = ", r)
            #print("b = ", b)
            if saveResiduals:
                res_arr[j+1] = np.linalg.norm(r)/np.linalg.norm(b)
            if np.linalg.norm(r)/np.linalg.norm(b) <= tol:
                return u, r, j, res_arr
        return u, r, MAX_IT, res_arr
    
    else:
        raise ValueError('Method name string not in Gaus_Seidel_Iteration not valid')
            
#defined these to conveniently plug in function names in the investigate... functions

def Forward_Gauss_Seidel_Iteration(N, eps, tol, saveResiduals = False):
    return Gauss_Seidel_Iteration(N, eps, tol, form = "Forward", saveResiduals = saveResiduals)

def Backward_Gauss_Seidel_Iteration(N, eps, tol, saveResiduals = False):
    return Gauss_Seidel_Iteration(N, eps, tol, form = "Backward", saveResiduals = saveResiduals)

def Symmetric_Gauss_Seidel_Iteration(N, eps, tol, saveResiduals = False):
    return Gauss_Seidel_Iteration(N, eps, tol, form = "Symmetric", saveResiduals = saveResiduals)
    
    
    
    
    

