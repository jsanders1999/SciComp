import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
#plt.rcParams['text.usetex'] = True
import discretization
import BIMs
from tqdm import tqdm
import matplotlib.cm as cm

def testSimpleSolve():
    """
    A function to test the function simpleSolve from discretization.py
    """
    N = 100         #The number of grid points
    eps = 0.1       #The epsilon parameter from the differential equation
    BC = [1, 0]     #The boundary conditions at x=0 and x=1

    #print(A(N, eps))
    #print(g(N, eps, BC))
    #print(simpleSolve(N, eps, BC))
    x = np.linspace(0,1,N+1)
    u = discretization.simpleSolve(N, eps, BC)
    plt.plot(x,u)
    plt.show()
    return

def investigateEpsilons():
    """
    A function to make plots of the numerical solution for different values of epsilon.
    This is exercise 1.
    """
    #Define constants
    N = int(50)                    #The number of grid points
    N_exact = int(1e3)
    eps_arr = np.logspace(-4,0,9)   #An array of epsilon parameters to solve the differential equation with
    BC = [1, 0]                     #The boundary conditions at x=0 and x=1

    #make plots for different eps
    plt.figure("Titel van een window met mooie plots")
    plt.xlabel(r"$x$")
    plt.ylabel(r"$u(x)$")
    plt.title(r"Plots of $u(x)$ for different $\epsilon$")
    x = np.linspace(0,1,N+1)
    for eps in eps_arr:
        u = discretization.simpleSolve(N, eps, BC)
        p = plt.plot(x,u, marker = ".", linestyle = "None" )
        plt.plot(*discretization.refSol(N_exact, eps), color = p[-1].get_color(), label = r"$\epsilon = {:.1e}$".format(eps))
    plt.legend()
    plt.show()
    return

def investigateAccuracySimpleSolver():
    """
    A function to make plots of the accuracy of the numerical solution for different N and eps values
    This is exercise 2.
    """
    eps_arr = np.logspace(-2,0,5)
    N_it    = np.array([16,32,64,128,256,512,1024,2048,4096,2*4096, 4*4096, 8*4096, 16*4096, 32*4096, 64*4096 ])
    BC      = [1,0]
    print ("{:<5} {:<5} {:} {:<10}".format('eps','N','||u - u_ex||','h'))
    fig = plt.figure("Mooie plots voor epsilon")
    ax = fig.add_subplot(1, 1, 1)
    for eps in eps_arr:
        error_arr = np.zeros(N_it.shape)
        for i, N in enumerate(N_it):
            h               = 1/N
            numSoly         = discretization.simpleSolve(N,eps,BC)
            refSolx,refSoly = discretization.refSol(N,eps)
            error           = np.max(np.abs(numSoly - refSoly))
            print ("{:<5} {:<5} {:<25} {:<10}".format(eps,N,error,h))
            error_arr[i] = error
        
        f = lambda x, a, b : a*x**b
        cutoff_index = 5
        popt, pcov = sp.optimize.curve_fit(f, 1/N_it[cutoff_index:], error_arr[cutoff_index:])
        print(popt)
        h_arr = np.linspace(1/N_it[-1], 1/N_it[0], 100)
        p = ax.plot(1/N_it, error_arr, marker = ".", linestyle = "None", label = r"$\epsilon = {:.2f}, b = {:.2f} $".format(eps, popt[1]) )
        plt.plot(h_arr, f(h_arr, *popt), color = p[-1].get_color())
    ax.set_yscale('log')
    ax.set_xscale('log')
    plt.xlabel(r"$h$")
    plt.ylabel(r"$||u - u_ex||$")
    plt.legend()
    plt.show()
    return refSoly, numSoly


def investigateConvergence(eps, method):
    """
    A function to make plots of the accuracy of the numerical solution 
    obtained with the Jacobi method for different N and eps values
    This is exercise part of exercise 6, (so also 7, 8, 9).

    eps: float64
        The epsilon parameter in the differential equatin
    method: function of (N, eps, tol) [BIMs.Jacobi_Iteration, BIMs.Gauss_Seidel, etc]
        The iterative method used to solve the system Au=f
    """
    tol     = 1e-6 #must always be 1e-6 as stated in the exercise
    N_it    = np.array([16,32,64,128,256])#,512,1024,2048,4096,2*4096, 4*4096, 8*4096, 16*4096, 32*4096, 64*4096 ])
    fig_sol = plt.figure("Jacobi Solutions for eps = {}".format(eps))
    ax = fig_sol.add_subplot(1, 1, 1)
    ax.set_title("Jacobi Solutions for eps = {}".format(eps))
    for i, N in tqdm(enumerate(np.flip(N_it)), desc= "N_it progress"):
        u_Jac, r, iter = method(N, eps, tol)
        x = np.linspace(0,1,N+1)
        u = discretization.AddBCtoSol(u_Jac)
        ax.plot(x, u, marker = ".", markersize = 2, linestyle = "None", label = "N = {}, iter = {}".format(N, iter))
    ax.set_xlabel(r"$x$")
    ax.set_ylabel(r"$u_{Jac}$")
    ax.legend()
    plt.show()
    return
    #TODO
    #Make a y-log plot of r^k/f versus k for different values of N
    #Make a table of the reduction factor for the last five jacobi itterations for different values of N
    return refSoly, numSoly

def Inverse():
    """ A function to inspect the properties of matrix Ah
        This is part of exercise 3"""
    N = 64
    h = 1/N
    eps = 0.5
    A = discretization.A(N,eps)
    matrixInverse = sp.sparse.linalg.inv(A)
    matrixInverse = sp.sparse.csr_matrix.toarray(matrixInverse)
    A = sp.sparse.csr_matrix.toarray(A)

    fig, (ax1,ax2) = plt.subplots(nrows=2)
    pos1 = ax1.imshow(A, cmap = 'Blues')
    fig.colorbar(pos1,ax=ax1, shrink = 0.5)
    pos2 = ax2.imshow(matrixInverse, cmap = 'Reds_r')
    cbarInv = fig.colorbar(pos2,ax=ax2, shrink = 0.5)
    cbarInv.minorticks_on()
    plt.show()
    return matrixInverse

def Eigenvalues():
    """ A function to inspect the eigenvalues of matrix A,
        This is part of exercise 4"""
    N = 64
    h = 1/N
    eps = 0.0001
    Ah  = discretization.A(N,eps)
    
    print(Ah)
    eigsInfo = sp.linalg.eig(sp.sparse.csr_matrix.toarray(Ah))
    print(eigsInfo)
    for i in range(len(eigsInfo[0])):
                   eigenvalue  = eigsInfo[0][i]
                   eigenvector = eigsInfo[1][i]
                   plt.plot(eigenvector,'ko')
                   stringTitle = "Eigenvalue = " + str(eigenvalue)
                   plt.title(stringTitle)
                   plt.show()
    Ah  = sp.sparse.csr_matrix.toarray(Ah)
    return Ah, eigsInfo

if __name__=="__main__":
    #investigateEpsilons()
    #investigateAccuracySimpleSolver()
    #Inverse()
    #Ah, Test = Eigenvalues()
    #investigateConvergenceJacobi()
    investigateConvergence(eps = 0.1, method = BIMs.Jacobi_Iteration)
