import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
#plt.rcParams['text.usetex'] = True
import discretization
import BIMs
import GMRES
from tqdm import tqdm
import matplotlib.cm as cm

NARR = np.array([16,32,64,128,256]) #np.array([16,32,64,128,256, 512, 1024]) #

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


def investigateMethodSolutions(eps, method, method_string, show = True):
    """
    A function to make plots of the numerical solutions 
    obtained with a numerical method for different N and eps values
    This is exercise part of exercise 6, (so also 7, 8, 9).

    eps: float64
        The epsilon parameter in the differential equatin
    method: function of (N, eps, tol) [BIMs.Jacobi_Iteration, BIMs.Gauss_Seidel, etc]
        The iterative method used to solve the system Au=f
    """
    #initialize values
    tol     = 1e-6 #must always be 1e-6 as stated in the exercise
    N_it    = NARR#np.array([16,32,64,128,256,512])#,512,1024,2048,4096,2*4096, 4*4096, 8*4096, 16*4096, 32*4096, 64*4096 ])
    fig_sol = plt.figure("{} Solutions for eps = {}".format(method_string, eps))

    #intialize figure and axis
    ax = fig_sol.add_subplot(1, 1, 1)
    ax.set_title("{} Solutions for eps = {}".format(method_string, eps))

    #Solve using Method for each N, plot solution in the axis
    for i, N in enumerate(np.flip(N_it)):
        u_meth, r, k_max, res_arr = method(N, eps, tol)
        x = np.linspace(0,1,N+1)
        u = discretization.AddBCtoSol(u_meth)
        ax.plot(x, u, marker = ".", markersize = 2, linestyle = "None", label = "N = {}, k_max = {}".format(N, k_max))
    
    #Add labels and to the plot
    ax.set_xlabel(r"$x$")
    ax.set_ylabel(r"$u_{}$")
    ax.legend()
    if show:
        plt.show()
    return

def investigateMethodConvergence(N, eps, method, method_string, show = True):
    """
    A function to make plots of the accuracy of the numerical solution 
    obtained with a numerical method for different N and eps values
    This is exercise part of exercise 6, (so also 7, 8, 9).

    eps: float64
        The epsilon parameter in the differential equatin
    method: function of (N, eps, tol) [BIMs.Jacobi_Iteration, BIMs.Gauss_Seidel, etc]
        The iterative method used to solve the system Au=f
    """
    #initialize values
    tol     = 1e-6 #must always be 1e-6 as stated in the exercise

    #intialize figure and axis
    fig_sol = plt.figure("{} Convergence for N = {}, eps = {}".format(method_string, N, eps))
    ax = fig_sol.add_subplot(1, 1, 1)
    ax.set_title("{} Convergence for N = {}, eps = {}".format(method_string, N, eps))

    #Solve using Method for N, plot solution in the axis
    u_meth, r, k_max, res_arr = method(N, eps, tol, saveResiduals = True)

    ax.plot(res_arr[:k_max+2], marker = ".", markersize = 2, linestyle = "None", label = "N = {}, k_max = {}".format(N, k_max))
    ax.set_xlabel(r"$k$")
    ax.set_ylabel(r"$\frac{||r^k||}{||f^h||}$")
    ax.set_yscale('log')
    ax.legend()
    if show:
        plt.show()
    return

def investigateMethodReductionFactors(eps, method, method_string):
    """
    A function to make plots of the accuracy of the numerical solution 
    obtained with a numerical method for different N and eps values
    This is exercise part of exercise 6, (so also 7, 8, 9).

    eps: float64
        The epsilon parameter in the differential equatin
    method: function of (N, eps, tol) [BIMs.Jacobi_Iteration, BIMs.Gauss_Seidel, etc]
        The iterative method used to solve the system Au=f
    """
    #initialize values
    tol     = 1e-6 #must always be 1e-6 as stated in the exercise
    N_arr   = NARR #np.array([16,32,64,128,256,512])
    red_arr = np.zeros((len(N_arr), 5))
    k_arr   = np.zeros(len(N_arr))

    #Solve using Method for each N, save reduction factors
    for i, N in enumerate(N_arr):
        u_meth, r, k_max, res_arr = method(N, eps, tol, saveResiduals = True)
        k_arr[i] = int(k_max + 1)
        for j in range(5):
            ind = j+k_max-3
            red_arr[i, j] = res_arr[ind]/res_arr[ind-1]

    #print the tables
    print(["N", "k_max", "red_{kmax-4}", "red_{kmax-3}", "red_{kmax-2}", "red_{kmax-1}", "red_{kmax}"])
    for i, N in enumerate(N_arr):
        print([N, k_arr[i],  *red_arr[i, :]])   
    
    fig_sol = plt.figure("{} reduction factors for eps = {}".format(method_string, eps))
    ax = fig_sol.add_subplot(1, 1, 1)
    ax.set_title("{} reduction factors for eps = {}".format(method_string, eps))
    ax.plot(N_arr, 1-red_arr[:, -1], marker = ".", markersize = 4, linestyle = "None")
    ax.set_xlabel(r"$N$")
    ax.set_ylabel(r"$1-red^{k_{max}}$")
    ax.set_yscale('log')
    plt.show()

    return

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

def testGMRES():
    N = 512
    eps = 0.1
    tol = 1e-6
    u_GMRES, r, it = GMRES.GMRES_Iterations(N, eps, tol)
    x = np.linspace(0,1,N+1)
    u = discretization.AddBCtoSol(u_GMRES)
    print(r, it)
    plt.plot(x,u)
    plt.show()
    return


def Exercise6():
    #Jacobi
    eps = 0.1
    investigateMethodSolutions(eps = eps, method = BIMs.Jacobi_Iteration, method_string = "Jacobi"  )
    for N in NARR:
        investigateMethodConvergence(N = N, eps = eps, method = BIMs.Jacobi_Iteration, method_string = "Jacobi" )
    investigateMethodReductionFactors(eps = eps, method = BIMs.Jacobi_Iteration, method_string = "Jacobi")
    return

def Exercise7():
    #Forward GS
    eps = 0.1
    investigateMethodSolutions(eps = eps, method = BIMs.Forward_Gauss_Seidel_Iteration, method_string = "Forward Gauss Seidel")
    for N in NARR:
        investigateMethodConvergence(N = N, eps = eps, method = BIMs.Forward_Gauss_Seidel_Iteration, method_string = "Forward Gauss Seidel")
    investigateMethodReductionFactors(eps = eps, method = BIMs.Forward_Gauss_Seidel_Iteration, method_string = "Forward Gauss Seidel")
    return

def Exercise8():
    #Backward GS
    eps = 0.1
    investigateMethodSolutions(eps = eps, method = BIMs.Backward_Gauss_Seidel_Iteration, method_string = "Backward Gauss Seidel")
    for N in NARR:
        investigateMethodConvergence(N = N, eps = eps, method = BIMs.Backward_Gauss_Seidel_Iteration, method_string = "Backward Gauss Seidel")
    investigateMethodReductionFactors(eps = eps, method = BIMs.Backward_Gauss_Seidel_Iteration, method_string = "Backward Gauss Seidel")
    return

def Exercise9():
    #Symmetric GS
    eps = 0.1
    investigateMethodSolutions(eps = eps, method = BIMs.Symmetric_Gauss_Seidel_Iteration, method_string = "Symmetric Gauss Seidel")
    for N in NARR:
        investigateMethodConvergence(N = N, eps = eps,  method = BIMs.Symmetric_Gauss_Seidel_Iteration, method_string = "Symmetric Gauss Seidel")
    investigateMethodReductionFactors(eps = eps,  method = BIMs.Symmetric_Gauss_Seidel_Iteration, method_string = "Symmetric Gauss Seidel")
    return

def Exercise12():
    #GMRES
    eps = 0.1
    investigateMethodSolutions(eps = eps, method = GMRES.GMRES_Iterations, method_string = "GMRES")
    for N in NARR:
        investigateMethodConvergence(N = N, eps = eps,  method = GMRES.GMRES_Iterations, method_string = "GMRES")
    return

def Exercise13():
    #GMRES(m)
    eps = 0.1
    m = 10
    def _GMRES_mFilledIn_Iterations(N, eps, tol, saveResiduals = False):
        return GMRES.GMRES_m_Iterations(N, eps, tol, m=m, saveResiduals = saveResiduals)
    investigateMethodSolutions(eps = eps, method = _GMRES_mFilledIn_Iterations, method_string = "GMRES({})".format(m))
    for N in NARR:
        investigateMethodConvergence(N = N, eps = eps, method = _GMRES_mFilledIn_Iterations, method_string = "GMRES({})".format(m))
    return


if __name__=="__main__":
    #investigateEpsilons()
    #investigateAccuracySimpleSolver()
    #Inverse()
    #Ah, Test = Eigenvalues()
    #investigateConvergenceJacobi()
    #investigateConvergence(eps = 0.1, method = BIMs.Jacobi_Iteration)
    #testGMRES()
    #investigateMethodSolutions(eps = 0.1, method = GMRES.GMRES_method, method_string = "GMRES" )
    #investigateMethodConvergence(N = 164, eps = 0.1, method = BIMs.Jacobi_Iteration, method_string = "Jacobi" )
    #investigateMethodReductionFactors(eps = 0.1, method = BIMs.Jacobi_Iteration, method_string = "Jacobi")
    Exercise13()
