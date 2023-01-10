import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
#plt.rcParams['text.usetex'] = True
import discretization
import BIMs
import GMRES
from tqdm import tqdm
import matplotlib.cm as cm
from investigateFunctions import *


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
    N = int(128)                    #The number of grid points
    N_exact = int(1e3)
    eps_arr = np.logspace(-2,0,6)   #An array of epsilon parameters to solve the differential equation with
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
    plt.plot(np.linspace(0,1),1 - np.linspace(0,1), label = "y = 1 - x")
    plt.axis('square')
    plt.legend()
    plt.show()
    return

def investigateAccuracySimpleSolver():
    """
    A function to make plots of the accuracy of the numerical solution for different N and eps values
    This is exercise 2.
    """
    eps_arr = np.logspace(-2,0,5)
    N_it    = np.array([16,32,64,128,256,100000,200000])
    BC      = [1,0]
    print ("{:<10} & {:<10} & {:<15} & {:<10} & {:<5} \\\\ \hline".format('eps','N','$\|u - u_ex\|$','h', 'factor'))
    fig = plt.figure("Exercise 2")
    ax = fig.add_subplot(1, 1, 1)
    for eps in eps_arr:
        error_arr = np.zeros(N_it.shape)
        for i, N in enumerate(N_it):
            h               = 1/N
            numSoly         = discretization.simpleSolve(N,eps,BC)
            refSolx,refSoly = discretization.refSol(N,eps)
            error           = np.max(np.abs(numSoly - refSoly))
            print ("{:<10} & {:<10} & {:<15} & {:<10} & {:<5} \\\\ \hline".format(np.around(eps,5),N,np.around(error,7),h, np.around(error/h,2)))
            error_arr[i] = error
        
##        f = lambda x, a, b : a*x**b
##        cutoff_index = 5
##        popt, pcov = sp.optimize.curve_fit(f, 1/N_it[cutoff_index:], error_arr[cutoff_index:])
##        print(popt)
##        h_arr = np.linspace(1/N_it[-1], 1/N_it[0], 100)
##        p = ax.plot(1/N_it, error_arr, marker = ".", linestyle = "None", label = r"$\epsilon = {:.2f}, b = {:.2f} $".format(eps, popt[1]) )
##        plt.plot(h_arr, f(h_arr, *popt), color = p[-1].get_color())
##    ax.set_yscale('log')
##    ax.set_xscale('log')
##    plt.xlabel(r"$h$")
##    plt.ylabel(r"$||u - u_ex||$")
##    plt.legend()
##    plt.show()
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

def investigateMethodConvergence(N, eps, method, method_string, ax = None):
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
    if ax == None:
        fig_sol = plt.figure("{} Convergence for N = {}, eps = {}".format(method_string, N, eps))
        ax = fig_sol.add_subplot(1, 1, 1)
        ax.set_title("{} Convergence for N = {}, eps = {}".format(method_string, N, eps))

    #Solve using Method for N, plot solution in the axis
    u_meth, r, k_max, res_arr = method(N, eps, tol, saveResiduals = True)

    ax.plot(res_arr[:k_max+2], marker = ".", markersize = 2, linestyle = "None", label = "{} for N = {}, k_max = {}".format(method_string, N, k_max))
    ax.set_xlabel(r"$k$")
    ax.set_ylabel(r"$\frac{||r^k||}{||f^h||}$")
    ax.set_yscale('log')
    ax.legend()
    if ax == None:
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

def EigenvaluePlot(Matrix, show = True):
    """A function to plot the eigenvalues of a full Matrix
        and to compute the spectral radius """
    Eigenvalues = np.linalg.eigvals(Matrix)
    SpectralRadius = np.max(np.abs(Eigenvalues))
    plt.scatter(np.real(Eigenvalues),np.imag(Eigenvalues))
    plt.scatter(np.real(SpectralRadius), np.imag(SpectralRadius),
                        marker = 'o', color ='red', label = "Spectral Radius = " + str(SpectralRadius))
    plt.title("Eigenvalue plot")
    plt.legend()
    plt.xlabel("Real part")
    plt.ylabel("Imaginary part")
    if show:
        plt.show()
    return SpectralRadius

def EigenvalueEigenvectorPlot(Matrix, k, N, eps, show = True):
    """A function to plot the eigenvalues and eigenvectors of a full Matrix """
    Eigenvalues, Eigenvectors = np.linalg.eig(Matrix)
    idx = np.argsort(Eigenvalues) 
    Eigenvalues = Eigenvalues[idx]
    Eigenvectors = Eigenvectors[:,idx]
    print(Eigenvalues)
    h  = 1/N
    x1 = np.linspace(h,1-h,num = int((N-1)))
    x2 = np.linspace(0,1,num = 5000)
    for i in k:
        y = Eigenvectors[:,i]
        if y[0] < 0:
            y = -y
        plt.scatter(x1,y, s = 1, label = "k = " + str(i+1) + r", $\lambda$  = "
                    + str(np.around(Eigenvalues[i],3)) + r" and estimated $\lambda$ = " + str(np.around(4*eps/(h)**2*np.sin(np.pi*(h)*(i+1)/2)**2,3)))
    for j in k:
        plt.plot(x2, (1/(np.sqrt(N-1)))*np.sin((j+1)*np.pi*x2)*np.exp(x2/(np.sqrt(3))))
    plt.title(r"Eigenvalue and eigenvectors plot for N = {} and $\epsilon = {}$".format(N,eps))
    plt.legend(loc = 'upper left')
    if show:
        plt.show()
    return Eigenvalues

def Exercise4():
    N       = 64
    eps     = 1
    k       = np.array([1,4,9])
    Matrix  = discretization.A(N,eps).toarray()
    EigenvalueEigenvectorPlot(Matrix, k, N, eps, show = True)
    
    return
    
def Exercise5():
    eps_arr = np.array([0.01,0.25,0.50,0.75,1.00])
    N_it    = np.array([8,16,32,64,128,256])
    print ("{:<5} {:<5} {:<25}".format('eps &','N &','Spectral Radius \\\\ \hline'))
    for eps in eps_arr:
        for N in N_it:
            Matrix = BIMs.B_Jacobi(N,eps)
            Matrix = Matrix.toarray()
            SpectralRadius = EigenvaluePlot(Matrix, show = False)
            print ("{:<5} &{:<5} &{:<25}\\\\ \hline".format(eps,N,np.around(SpectralRadius,6)))
    return

def Exercise6():
    #Jacobi
    eps = 0.1
    investigateMethodSolutions(eps = eps, method = BIMs.Jacobi_Iteration, method_string = "Jacobi"  )

    fig_sol = plt.figure("{} Convergence for eps = {}".format("Jacobi", eps))
    ax = fig_sol.add_subplot(1, 1, 1)
    ax.set_title("{} Convergence eps = {}".format("Jacobi", eps))
    k_max_arr = np.zeros(NARR.shape)
    for i,N in enumerate(NARR):
        k_max_arr[i] = investigateMethodConvergence(N = N, eps = eps, method = BIMs.Jacobi_Iteration, method_string = "Jacobi", ax_in = ax )
    ax.set_xscale('log')
    plt.show()
    fig_sol = plt.figure("k_max as a function of N")
    ax = fig_sol.add_subplot(1, 1, 1)
    ax.set_title(r"$k_{max}$ as a function of $N$")
    ax.plot(NARR, k_max_arr, marker = ".", markersize = 4, linestyle = "None")
    f = lambda x, a, b : a*x**b
    popt, pcov = sp.optimize.curve_fit(f, NARR, k_max_arr)
    h_arr = np.linspace(NARR[0], NARR[-1], 100)
    plt.plot(h_arr, f(h_arr, *popt), label = r"$a = {:.2f}, b = {:.2f} $".format(popt[0], popt[1]))
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel(r'$N$')
    ax.set_ylabel(r'$k_{max}$')
    plt.legend()
    plt.show()

    investigateMethodReductionFactors(eps = eps, method = BIMs.Jacobi_Iteration, method_string = "Jacobi")

    return

def Exercise7():
    #Forward GS
    eps = 0.1
    method = BIMs.Forward_Gauss_Seidel_Iteration
    method_string = "Forward Gauss Seidel"
    investigateMethodSolutions(eps = eps, method = method , method_string = method_string)

    fig_sol = plt.figure("{} Convergence for eps = {}".format(method_string, eps))
    ax = fig_sol.add_subplot(1, 1, 1)
    ax.set_title("{} Convergence eps = {}".format(method_string, eps))
    k_max_arr = np.zeros(NARR.shape)
    for i,N in enumerate(NARR):
        k_max_arr[i] = investigateMethodConvergence(N = N, eps = eps, method = method , method_string = method_string, ax_in = ax )
    ax.set_xscale('log')
    plt.show()

    fig_sol = plt.figure("k_max as a function of N")
    ax = fig_sol.add_subplot(1, 1, 1)
    ax.set_title(r"$k_{max}$ as a function of $N$")
    ax.plot(NARR, k_max_arr, marker = ".", markersize = 4, linestyle = "None")
    f = lambda x, a, b : a*x**b
    popt, pcov = sp.optimize.curve_fit(f, NARR, k_max_arr)
    h_arr = np.linspace(NARR[0], NARR[-1], 100)
    plt.plot(h_arr, f(h_arr, *popt), label = r"$a = {:.2f}, b = {:.2f} $".format(popt[0], popt[1]))
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel(r'$N$')
    ax.set_ylabel(r'$k_{max}$')
    plt.legend()
    plt.show()

    investigateMethodReductionFactors(eps = eps,  method = method , method_string = method_string)
    return

def Exercise8():
    #Backward GS
    eps = 0.1
    method = BIMs.Backward_Gauss_Seidel_Iteration
    method_string = "Backward Gauss Seidel"
    investigateMethodSolutions(eps = eps, method = method , method_string = method_string)

    fig_sol = plt.figure("{} Convergence for eps = {}".format(method_string, eps))
    ax = fig_sol.add_subplot(1, 1, 1)
    ax.set_title("{} Convergence eps = {}".format(method_string, eps))
    k_max_arr = np.zeros(NARR.shape)
    for i,N in enumerate(NARR):
        k_max_arr[i] = investigateMethodConvergence(N = N, eps = eps, method = method , method_string = method_string, ax_in = ax )
    ax.set_xscale('log')
    plt.show()

    fig_sol = plt.figure("k_max as a function of N")
    ax = fig_sol.add_subplot(1, 1, 1)
    ax.set_title(r"$k_{max}$ as a function of $N$")
    ax.plot(NARR, k_max_arr, marker = ".", markersize = 4, linestyle = "None")
    f = lambda x, a, b : a*x**b
    popt, pcov = sp.optimize.curve_fit(f, NARR, k_max_arr)
    h_arr = np.linspace(NARR[0], NARR[-1], 100)
    plt.plot(h_arr, f(h_arr, *popt), label = r"$a = {:.2f}, b = {:.2f} $".format(popt[0], popt[1]))
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel(r'$N$')
    ax.set_ylabel(r'$k_{max}$')
    plt.legend()
    plt.show()

    investigateMethodReductionFactors(eps = eps,  method = method , method_string = method_string)
    return

def Exercise9():
    #Symmetric GS
    eps = 0.1
    method = BIMs.Symmetric_Gauss_Seidel_Iteration
    method_string = "Symmetric Gauss Seidel"
    investigateMethodSolutions(eps = eps, method = method , method_string = method_string)

    fig_sol = plt.figure("{} Convergence for eps = {}".format(method_string, eps))
    ax = fig_sol.add_subplot(1, 1, 1)
    ax.set_title("{} Convergence eps = {}".format(method_string, eps))
    k_max_arr = np.zeros(NARR.shape)
    for i,N in enumerate(NARR):
        k_max_arr[i] = investigateMethodConvergence(N = N, eps = eps, method = method , method_string = method_string, ax_in = ax )
    ax.set_xscale('log')
    plt.show()

    fig_sol = plt.figure("k_max as a function of N")
    ax = fig_sol.add_subplot(1, 1, 1)
    ax.set_title(r"$k_{max}$ as a function of $N$")
    ax.plot(NARR, k_max_arr, marker = ".", markersize = 4, linestyle = "None")
    f = lambda x, a, b : a*x**b
    popt, pcov = sp.optimize.curve_fit(f, NARR, k_max_arr)
    h_arr = np.linspace(NARR[0], NARR[-1], 100)
    plt.plot(h_arr, f(h_arr, *popt), label = r"$a = {:.2f}, b = {:.2f} $".format(popt[0], popt[1]))
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel(r'$N$')
    ax.set_ylabel(r'$k_{max}$')
    plt.legend()
    plt.show()

    investigateMethodReductionFactors(eps = eps,  method = method , method_string = method_string)

def Exercise12():
    #GMRES
    eps = 0.1
    investigateMethodSolutions(eps = eps, method = GMRES.GMRES_Iterations, method_string = "Full GMRES")

    fig_sol = plt.figure("{} Convergence for eps = {}".format("Full GMRES", eps))
    ax = fig_sol.add_subplot(1, 1, 1)
    ax.set_title("{} Convergence eps = {}".format("Full GMRES", eps))
    k_max_arr = np.zeros(NARR.shape)
    for i,N in enumerate(NARR):
        k_max_arr[i] = investigateMethodConvergence(N = N, eps = eps, method = GMRES.GMRES_Iterations , method_string = "Full GMRES", ax_in = ax )
    ax.set_xscale('log')
    plt.show()

    fig_sol = plt.figure("k_max as a function of N")
    ax = fig_sol.add_subplot(1, 1, 1)
    ax.set_title(r"$k_{max}$ as a function of $N$")
    ax.plot(NARR, k_max_arr, marker = ".", markersize = 4, linestyle = "None")
    f = lambda x, a, b : a*x**b
    popt, pcov = sp.optimize.curve_fit(f, NARR, k_max_arr)
    h_arr = np.linspace(NARR[0], NARR[-1], 100)
    plt.plot(h_arr, f(h_arr, *popt), label = r"$a = {:.2f}, b = {:.2f} $".format(popt[0], popt[1]))
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel(r'$N$')
    ax.set_ylabel(r'$k_{max}$')
    plt.legend()
    plt.show()

    return

def Exercise13():
    #GMRES(m)
    eps = 0.1
    m_arr = np.array([1,2,4,8,16,32,64,128,256])
    #This first for loop raises errors if you do not adjust m_arr and NARR for each run.
    for m in m_arr:
        def _GMRES_mFilledIn_Iterations(N, eps, tol, saveResiduals = True):
            return GMRES.GMRES_m_Iterations(N, eps, tol, m=m, saveResiduals = saveResiduals)
        investigateMethodSolutions(eps = eps, method = _GMRES_mFilledIn_Iterations, method_string = "GMRES({})".format(m))
        
    m_arr = np.array([1,2,4,8,16,32,64,128,256,512])
    for N in NARR:
        fig_sol = plt.figure("{} Convergence for N = {}, eps = {}".format("GMRES(m)", N, eps))
        ax = fig_sol.add_subplot(1, 1, 1)
        ax.set_title("{} Convergence for N = {}, eps = {}".format("GMRES(m)", N, eps))
        for m in m_arr[m_arr<N][-5:]:
            def _GMRES_mFilledIn_Iterations(N, eps, tol, saveResiduals = True):
                return GMRES.GMRES_m_Iterations(N, eps, tol, m=m, saveResiduals = saveResiduals)
            investigateMethodConvergence(N = N, eps = eps, method = _GMRES_mFilledIn_Iterations, method_string = "GMRES({})".format(m), ax_in = ax )
        plt.show()
    return

#IDEAS
#Put all BIM convergence plots in one graph for each N
#put all GMRES(M) convergence plots in one graph for each N


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
    Exercise4()
    #Exercise13()
    #investigateAccuracySimpleSolver()
    #Exercise8()
    #Exercise9()
