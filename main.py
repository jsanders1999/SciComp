import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
#plt.rcParams['text.usetex'] = True
import discretization
import BIMs

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
        p = ax.plot(1/N_it, error_arr,  marker = ".", linestyle = "None", label = r"$\epsilon = {:.2f}, b = {:.2f} $".format(eps, popt[1]) )
        plt.plot(h_arr, f(h_arr, *popt), color = p[-1].get_color())
        plt.plot
    ax.set_yscale('log')
    ax.set_xscale('log')
    plt.xlabel(r"$h$")
    plt.ylabel(r"$||u - u_ex||$")
    plt.legend()
    plt.show()


    return refSoly, numSoly

def Inverse():
    """ Function to inspect the properties of matrix Ah"""
    N = 8
    h = 1/N
    eps = 0.5
    matrixInverse = sp.sparse.linalg.inv(discretization.A(N,eps))
    matrixInverse = sp.sparse.csr_matrix.toarray(matrixInverse)

    print("All entries are >= 0 : ", np.all(matrixInverse>=0))
    return matrixInverse

def Eigenvalues():
    """ Function to inspect the eigenvalues of matrix A"""
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
