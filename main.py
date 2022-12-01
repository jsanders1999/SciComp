import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
#plt.rcParams['text.usetex'] = True
import discretization

def u(x,eps):
    return (np.exp(x/eps - np.exp(1/eps)))/(1 - np.exp(1/eps))

def test():
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
    N = int(1e3)                    #The number of grid points
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
        plt.plot(x,u, label = r"$\epsilon = {:2f}$".format(eps))
    plt.legend()
    plt.show()
    return

def investigateAccuracy():
    """
    A function to make plots of the accuracy of the numerical solution for different N and eps values
    This is exercise 2.
    """
    eps_arr = np.round(np.linspace(0.2,1,num=5),1)
    N_it    = np.array([16,32,64,128,256])
    BC      = [1,0]
    print ("{:<5} {:<5} {:<25} {:<10}".format('eps','N','||u - u_ex||','h'))
    for eps in eps_arr:
        for N in N_it:
            h               = 1/N
            numSoly         = discretization.simpleSolve(N,eps,BC)
            refSolx,refSoly = discretization.refSol(N,eps)
            error           = np.max(np.abs(numSoly - refSoly))
            print ("{:<5} {:<5} {:<25} {:<10}".format(eps,N,error,h))
    return refSoly, numSoly

def Inverse():
    """ Function to inspect the properties of matrix Ah"""
    N = 8
    h = 1/N
    eps = 0.5
    matrixInverse = sp.sparse.linalg.inv(discretization.A(N,eps))
    matrixInverse = sp.sparse.csr_matrix.toarray(matrixInverse)
    
    print("All entries are >= 0 : ", np.all(matrixInverse>=0))
    return MatrixInverse

def Eigenvalues():
    N = 8
    h = 1/N
    eps = 0.5
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
    return Ah,eigsInfo

if __name__=="__main__":
    print("Ricky moet adten en Julian adt mee")
    #investigateEpsilons()
    #investigateAccuracy()
    #Inverse()
    Ah, Test = Eigenvalues()
