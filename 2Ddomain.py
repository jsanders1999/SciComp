import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from tqdm import tqdm
import matplotlib.pyplot as plt

NARR = np.array([8,16,32,64,128])#,256])#np.array([16,32,64,128])

def A_2D(N):
    """
    A function that returns the discretization matrix for a 2D system with N^2 nodes and parameter epsilon.
    Uses 

    Parameters
    ----------
    N : int
        The number of grid points 

    Returns
    -------
    sparse array of floats (format : CSC, dim: (N-1)**2 by (N-1)**2)
        The discretization matrix
    """

    h = 1/N
    L = 1/h**2*( 4*sp.sparse.eye((N-1)**2, format='csc') - sp.sparse.eye((N-1)**2, k=1, format='csc') - sp.sparse.eye((N-1)**2, k=-1, format='csc') 
                - sp.sparse.eye((N-1)**2, k=-(N-1), format='csc') - sp.sparse.eye((N-1)**2, k=(N-1), format='csc'))
    for k in range(1, N-1):
        i,j = (N-1)*k, (N-1)*k
        L[i,j-1] = 0 
        L[i-1,j] = 0 
    D = 1/h*( sp.sparse.eye((N-1)**2, format='csc') - sp.sparse.eye((N-1)**2, k=-1, format='csc') )
    for k in range(1, N-1):
        i,j = (N-1)*k, (N-1)*k-1
        D[i,j] = 0 
    #print(h**2*L.toarray())
    #print(h*D.toarray())
    A = -L + D
    return A, L, D

def f(N):
    """
    A function that resturns the discretization vector for a system with (N-1)**2 nodes and f=1.

    Parameters
    ----------
    N : int
        The number of grid points 

    Returns
    -------
    array of floats (dim:(N-1)**2)
        The discretization vector
    """
    res = np.ones((N-1)**2)
    return res

def simpleSolve(N):
    """
    A function that solves A*u=f by inverting the sparse matrix A.

    Parameters
    ----------
    N : int
        The number of grid points 

    Returns
    -------
    array of floats (dim: (N-1)**2)
        The numerical solution of the differential equation, with boundary values included
    """
    u = sp.sparse.linalg.spsolve(A_2D(N), f(N)) 
    return u

def AddBCtoSol(u_int):
    N = int(np.sqrt(u_int.size))+1
    u = np.zeros((N+1, N+1))
    for i in range(0,N-1):
        u[i+1,1:N] = u_int[(N-1)*i:(N-1)*(i+1)]
    return u

def TestA():
    N = 512
    u_sol = simpleSolve(N)
    u = AddBCtoSol(u_sol)
    fig = plt.figure("Directly Solved u(x,y) for N = " + str(N))
    ax = fig.add_subplot(1, 1, 1)
    im = ax.imshow(u, aspect="equal", origin = "lower", extent = (0,1,0,1))
    ax.set_xlabel(r"$x$")
    ax.set_ylabel(r"$y$")
    ax.set_title(r"Directly Solved $u(x,y)$ for $N =$ " + str(N))
    plt.colorbar(im)
    plt.show()
    return

def GMRES_Iterations_2D(N, tol, saveResiduals = False, k = None, u0 = [None]):
    if k == None:
        k= (N-1)**2
    if u0[0] == None:
        u0 = np.zeros((N-1)**2)
    b           = f(N)
    A = A_2D(N)
    #P = np.linalg.inv(D)
    b_prec = b/A[0,0] #diagonal has same values everywhere
    A_prec = A/A[0,0] #diagonal has same values everywhere
    r0_prec     = b_prec - A_prec.dot(u0)                     #Starting residual for u = np.zeros(N-1) 
    v_arr       = np.zeros(((k+1), (N-1)**2))
    v_arr[0,:]  = r0_prec/np.linalg.norm(r0_prec) #the direction vectors will be stored as the rows of this matrix
    #A          = A.toarray() #make A not sparse anymore :'(
    H = np.zeros((k+1, k)) 
    if saveResiduals:
        res_arr = np.zeros((k+1))
        res_arr[0] = np.linalg.norm(b - A.dot(u0))/np.linalg.norm(b)
    else:
        res_arr = None
    for j in tqdm(range(k), desc = "GMRES iterations"):
        v_arr[j+1, :] = A_prec.dot(v_arr[j, :])
        for i in range(j+1):
            H[i,j] = v_arr[j+1, :].dot(v_arr[i, :]) #could be sped up with an array multiplication?
            v_arr[j+1, :] = v_arr[j+1, :] - H[i,j]*v_arr[i, :]
        H[j+1,j] = np.linalg.norm(v_arr[j+1, :])
        if saveResiduals:
            beta = np.linalg.norm(r0_prec)
            y = np.linalg.solve((H[:j+2,:j+1].T).dot(H[:j+2,:j+1]), beta*H[0,:j+1])
            u = u0 + (v_arr[:j+1,:].T).dot(y)
            r = b - A.dot(u)
            res_arr[j+1] = np.linalg.norm(r)/np.linalg.norm(b)
            if res_arr[j+1]<=tol:
                print("tolerance reached")
                res = res_arr[j+1]
                return u, res, j+1, res_arr

        if abs(H[j+1,j])<1e-20:
            print("lucky breakdown!")
            beta = np.linalg.norm(r0_prec)
            y = np.linalg.solve((H[:j+2,:j+1].T).dot(H[:j+2,:j+1]), beta*H[0,:j+1])
            u = u0 + (v_arr[:j+1,:].T).dot(y)
            res = np.linalg.norm(b - A.dot(u))/np.linalg.norm(b)
            return u, res, j+1, res_arr
        v_arr[j+1, :] = v_arr[j+1, :]/H[j+1,j] #normalize v_arr[j+1, :]
    #H_k = h[:k+2, :k+1] #when terminated after k steps 
    #y = argmin np.linalg.norm( beta * [1,0,0,0,..,0] - H_k.dot(y))

    beta = np.linalg.norm(r0_prec)
    y = np.linalg.solve((H.T).dot(H), beta*H[0,:])
    u = u0 + (v_arr[:-1,:].T).dot(y)
    res = np.linalg.norm(b - A.dot(u))/np.linalg.norm(b)
    return u, res, k, res_arr

def investigateMethodSolutions2D(method, method_string):
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

    #Solve using Method for each N, plot solution in the axis
    for i, N in enumerate(np.flip(N_it)):
        u_meth, r, k_max, res_arr = method(N, tol, saveResiduals = True)
        u = AddBCtoSol(u_meth)

        fig = plt.figure(method_string + " solution of $u(x,y)$ for N = " + str(N))
        ax = fig.add_subplot(1, 1, 1)
        im = ax.imshow(u, aspect="equal", origin = "lower", extent = (0,1,0,1))
        ax.set_xlabel(r"$x$")
        ax.set_ylabel(r"$y$")
        ax.set_title(method_string + r" solution of $u(x,y)$ for N = " + str(N))
        plt.colorbar(im)
        plt.show()
    return

def investigateMethodConvergence2D(N, method, method_string, ax_in = None):
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
    if ax_in == None:
        fig_sol = plt.figure("{} Convergence for N = {}".format(method_string, N))
        ax = fig_sol.add_subplot(1, 1, 1)
        ax.set_title("{} Convergence for N = {}".format(method_string, N))
    else:
        ax = ax_in

    #Solve using Method for N, plot solution in the axis
    u_meth, r, k_max, res_arr = method(N, tol, saveResiduals = True)

    ax.plot(res_arr[:k_max+2], marker = ".", markersize = 2, linestyle = "None", label = "{}, k_max = {}".format(method_string, k_max))
    ax.set_xlabel(r"$k$")
    ax.set_ylabel(r"$\frac{||r^k||}{||f^h||}$")
    ax.set_yscale('log')
    ax.legend()
    if ax_in == None:
        plt.show()
    return k_max

def GMRES_m_Iterations_2D(N, tol, m, saveResiduals = True):
    MAX_REC = 10000
    res_arr_full = np.zeros(MAX_REC*m+1)
    k_max_full = 0
    u, r, k_max, res_arr = GMRES_Iterations_2D(N, tol, k = m, saveResiduals = saveResiduals)
    res_arr_full[:m+1] = res_arr

    k_max_full      += k_max
    if r <= tol:
        return  u, r, k_max_full, res_arr_full
    
    for rec in range(1, MAX_REC):
        u, r, k_max, res_arr = GMRES_Iterations_2D(N, tol, k = m, saveResiduals = saveResiduals, u0 = u)
        res_arr_full[1+(rec)*(m):(rec+1)*(m)+1] = res_arr[1:] #first value was already in the full array
        k_max_full      += k_max
        if r <= tol:
            return  u, r, k_max_full, res_arr_full
    return  u, r, k_max_full, res_arr_full

def Exercise14_10():
    N       = 8
    A,L,D  = A_2D(N)
    A = A.toarray()
    D = D.toarray()
    Matrix = np.identity((N-1)**2) - 1/A[0,0]*A
    Eigenvalues = np.linalg.eigvals(Matrix)
    idx = np.argmax(np.abs(Eigenvalues))
    plt.scatter(np.real(Eigenvalues),np.imag(Eigenvalues))
    plt.scatter(np.real(Eigenvalues[idx]), np.imag(Eigenvalues[idx]),
                        marker = 'o', color ='red', label = "Spectral Radius = " + str(np.abs(Eigenvalues[idx])))
    plt.title(r"Eigenvalue plot together with an ellipse for N = {}".format(N))
    plt.legend()
    plt.xlabel("Real part")
    plt.ylabel("Imaginary part")

    #MinimumEigv = Eigenvalues[np.argmin(Eigenvalues)]
    #MaximumEigv = Eigenvalues[np.argmax(Eigenvalues)]
    MaximumRealpart = np.max(np.real(Eigenvalues))
    MinimumRealpart = np.min(np.real(Eigenvalues))
    MaximumImagpart = np.max(np.imag(Eigenvalues))
    MinimumImagpart = np.min(np.imag(Eigenvalues))

    u= np.real((MinimumRealpart + MaximumRealpart)/2)    #x-position of the center
    v= np.imag((MinimumImagpart + MaximumImagpart)/2)   #y-position of the center
    a= (u - MaximumRealpart)     #radius on the x-axis
    b= 0.125      #radius on the y-axis
    
    u = np.around(u,3)
    v = np.around(v,3)
    coef1 = np.around((1/a)**2,3)
    coef2 = np.around((1/b)**2,3)
    t = np.linspace(0, 2*np.pi, 5000)
    plt.plot( u+a*np.cos(t) , v+b*np.sin(t), label = r"{} (x-{})^2+ {} (y-{})^2".format(coef1,u,coef2,v))
    plt.grid(color='black',linestyle='--')
    plt.legend(loc = 'lower center')
    plt.show()

    return

def Exercise14_11():
    return

def Exercise14_12():
    investigateMethodSolutions2D( method = GMRES_Iterations_2D, method_string = "Full GMRES")

    fig_sol = plt.figure("{} Convergence for the 2D problem".format("Full GMRES"))
    ax = fig_sol.add_subplot(1, 1, 1)
    ax.set_title("{} Convergence for the 2D problem".format("Full GMRES"))
    k_max_arr = np.zeros(NARR.shape)
    for i,N in enumerate(NARR):
        k_max_arr[i] = investigateMethodConvergence2D(N = N, method = GMRES_Iterations_2D , method_string = "Full GMRES", ax_in = ax )
    #ax.set_xscale('log')
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

def Exercise14_13():
    #GMRES(m)
    #m_arr = np.array([32]) #np.array([1,2,4,8,16,32,64,128,256])
    #for m in m_arr:
    #    def _GMRES_mFilledIn_Iterations_2D(N, tol, saveResiduals = True):
    #        return GMRES_m_Iterations_2D(N, tol, m=m, saveResiduals = saveResiduals)
    #    investigateMethodSolutions2D(method = _GMRES_mFilledIn_Iterations_2D, method_string = "GMRES({})".format(m))
    m_arr = np.array([1,2,4,8,16,32,64,128])
    for N in NARR:
        fig_sol = plt.figure("{} Convergence for N = {}".format("GMRES(m)", N))
        ax = fig_sol.add_subplot(1, 1, 1)
        ax.set_title("{} Convergence for N = {}".format("GMRES(m)", N))
        for m in m_arr[m_arr<=(N-1)**2][-5:]:
            def _GMRES_mFilledIn_Iterations_2D(N, tol, saveResiduals = True):
                return GMRES_m_Iterations_2D(N, tol, m=m, saveResiduals = saveResiduals)
            investigateMethodConvergence2D(N = N, method = _GMRES_mFilledIn_Iterations_2D, method_string = "GMRES({})".format(m), ax_in = ax )
        plt.show()
    return
    return

    

if __name__ == "__main__":
    #TestA()
    #Exercise14_12()
    Exercise14_10()