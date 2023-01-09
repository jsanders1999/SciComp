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

def EigenvalueEigenvectorPlot(Matrix, k, show = True):
    """A function to plot the eigenvalues and eigenvectors of a full Matrix """
    Eigenvalues, Eigenvectors = np.linalg.eig(Matrix)
    for i in k:
        x = np.linspace(0,1,num = int(Matrix.shape[0]))
        y = Eigenvectors[:,i]
        plt.scatter(x,y, label = "k = " + str(i) + " eigenvalue = " + str(Eigenvalues[i]))
    plt.legend()
    if show:
        plt.show()
    return

def Exercise4():
    k       = np.array([1,3,5])
    Matrix  = discretization.A(32,0.5).toarray()
    EigenvalueEigenvectorPlot(Matrix, k, show = True)
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
    #Exercise4()
    #Exercise13()
    #investigateAccuracySimpleSolver()
    #Exercise8()
    #Exercise9()
    #Exercise13()
