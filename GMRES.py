import numpy as np
import scipy as sp
import discretization
from tqdm import tqdm
import matplotlib.pyplot as plt

def GMRES_Iterations(N, eps, tol, saveResiduals = False, k = None, u0 = [None]):
    if k == None:
        k= N-1
    if u0[0] == None:
        u0 = np.zeros(N-1)
    b           = discretization.f(N,eps, BC = [1,0])
    A, D, E, F, E_hat, F_hat = discretization.A_Full(N, eps)
    #P = np.linalg.inv(D)
    b_prec = b/D[0,0] #diagonal has same values everywhere
    A_prec = A/D[0,0] #diagonal has same values everywhere
    r0_prec     = b_prec - A_prec.dot(u0)                     #Starting residual for u = np.zeros(N-1) 
    v_arr       = np.zeros((k+1, N-1))
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
        if abs(H[j+1,j])<1e-20:
            print("lucky breakdown!")
            beta = np.linalg.norm(r0_prec)
            y = np.linalg.solve((H[:j+2,:j+1].T).dot(H[:j+2,:j+1]), beta*H[0,:j+1])
            u = u0 + (v_arr[:j+1,:].T).dot(y)
            return u, np.linalg.norm(A.dot(u)-b), j+1, res_arr
        v_arr[j+1, :] = v_arr[j+1, :]/H[j+1,j] #normalize v_arr[j+1, :]
    #H_k = h[:k+2, :k+1] #when terminated after k steps 
    #y = argmin np.linalg.norm( beta * [1,0,0,0,..,0] - H_k.dot(y))

    beta = np.linalg.norm(r0_prec)
    y = np.linalg.solve((H.T).dot(H), beta*H[0,:])
    u = u0 + (v_arr[:-1,:].T).dot(y)
    res = np.linalg.norm(b - A.dot(u))/np.linalg.norm(b)
    return u, res, k, res_arr

def GMRES_m_Iterations(N, eps, tol, m, saveResiduals = False):
    MAX_REC = 1000
    res_arr_full = np.zeros(MAX_REC*m)
    k_max_full = 0
    u, r, k_max, res_arr = GMRES_Iterations(N, eps, tol, k = m-1, saveResiduals = saveResiduals)
    res_arr_full[:m]    = res_arr
    k_max_full      += k_max
    
    for rec in range(1, MAX_REC):
        u, r, k_max, res_arr = GMRES_Iterations(N, eps, tol, k = m-1, saveResiduals = saveResiduals, u0 = u)
        res_arr_full[(rec)*m:(rec+1)*m] = res_arr
        k_max_full      += k_max+1
        if r < tol:
            return  u, r, k_max_full, res_arr_full
    return  u, r, k_max_full, res_arr_full



