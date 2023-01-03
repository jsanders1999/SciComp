import numpy as np
import scipy as sp
import discretization
from tqdm import tqdm

def GMRES_method(N, eps, tol):
    k           = (N-1)
    b           = discretization.f(N,eps, BC = [1,0])
    A           = discretization.A(N,eps)
    #TODO: Add Jacobi matrix M as a preconditioner
    u           = np.zeros(N-1)
    r           = b                     #Starting residual for u = np.zeros(N-1)
    v_arr       = np.zeros((k+1, N-1))
    v_arr[0,:]  = r/np.linalg.norm(r) #the direction vectors will be stored as the rows of this matrix
    #A          = A.toarray() #make A not sparse anymore :'(
    H = np.zeros((k+1, k)) 
    for j in tqdm(range(k), desc = "GMRES iterations"):

        v_arr[j+1, :] = A.dot(v_arr[j, :])
        for i in range(j+1):
            H[i,j] = v_arr[j+1, :].dot(v_arr[i, :]) #could be sped up with an array multiplication?
            v_arr[j+1, :] = v_arr[j+1, :] - H[i,j]*v_arr[i, :]
        H[j+1,j] = np.linalg.norm(v_arr[j+1, :])
        if abs(H[j+1,j])<1e-20:
            print("lucky breakdown!")
            beta = np.linalg.norm(r)
            y = np.linalg.solve((H[:j+2,:j+1].T).dot(H[:j+2,:j+1]), beta*H[0,:j+1])
            u = u + (v_arr[:j+1,:].T).dot(y)
            return u, np.linalg.norm(A.dot(u)-b) ,j+1
        v_arr[j+1, :] = v_arr[j+1, :]/H[j+1,j] #normalize v_arr[j+1, :]
    #H_k = h[:k+2, :k+1] #when terminated after k steps 
    #y = argmin np.linalg.norm( beta * [1,0,0,0,..,0] - H_k.dot(y))

    beta = np.linalg.norm(r)
    y = np.linalg.solve((H.T).dot(H), beta*H[0,:])
    u = u + (v_arr[:-1,:].T).dot(y)
    return u, np.linalg.norm(A.dot(u)-b) , k

