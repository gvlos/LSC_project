# Sequential implementation of ALS
# author: Gianvito Losapio


import numpy as np
import scipy.linalg as linalg


def isInvertible(A):
    dim = A.shape
    return dim[0] == dim[1] \
                and np.linalg.matrix_rank(A) == dim[0]


def isSymmetric(A):
    return (A == A.T).all()


def regularizedLSTrain(Xtr, Ytr, reg_par):
    
    A = Xtr @ Xtr.T + reg_par * Xtr.shape[1] * np.identity(Xtr.shape[0])
    
    if not isSymmetric(A):
        raise ValueError('A is not symmetric')
        
    if not isInvertible(A):
        raise ValueError('A is not invertible')

    V = Xtr @ Ytr.T
    
    return linalg.cho_solve(linalg.cho_factor(A), V)



def ALS_local_train(R, p, reg_par, R_val, min_delta, max_iter, progress=False):
    """
    Compute a low-rank approximation of the rating matrix R as
    a greedy solution to the Tikhonov regularization problem

    Input:
    R = rating matrix for training
    p = number of features
    reg_par = regularization parameter
    R_val = rating matrix for validation ("probe dataset")
    min_delta = target RMSE improvement
    max_iter = max number of iterations
    progress = True if you want the evolution of rmse as a result,
    False if you want just the best RMSE reached

    Output:
    M = movie feature matrix
    U = user feature matrix

    """

    # Number of users, movies
    n_u, n_m = R.shape

    # Init M, U
    M = M_init(R, n_m, p)
    U = np.zeros((p, n_u))

    # Init array for rmse evolution
    rmse_ev = []

    # Init control variables
    iter = 0
    delta_err = float('inf')
    err_new = float('inf')


    while (delta_err > min_delta) and (iter < max_iter):

        # Save a copy of the matrices before the update
        M_old = np.copy(M)
        U_old = np.copy(U)
        
        # Solve U given M
        for i in range(n_u):

            # indicies of movies rated by user i
            # (Assuming that missing values are np.nan)
            I_i = ~np.isnan(R[i,:])

            # feature vectors of the movies that user i has rated
            M_I_i = M[:, I_i]

            # ratings given by user i
            R_I_i = R[i, I_i]

            # Compute column u_i
            U[:,i] = regularizedLSTrain(M_I_i, R_I_i, reg_par)


        # Solve M given U
        for j in range(n_m):

            # indicies of users who rated movie j
            # (Assuming that missing values are np.nan)
            I_j = ~np.isnan(R[:,j])

            # feature vectors of users who rated movie j
            U_I_j = U[:, I_j]

            # ratings for the movie j
            R_I_j = R[I_j , j]

            # Compute column m_j
            M[:,j] = regularizedLSTrain(U_I_j, R_I_j, reg_par)


        # Update control variables
        err_old = err_new
        err_new = calc_rmse(R_val, U, M)
        rmse_ev.append(err_new)
        delta_err = err_old - err_new
        iter += 1


    # if the error increased instead of decreasing
    # return U, M from the previous iteration
    if delta_err < 0:
        M = M_old
        U = U_old

    if progress is False:
        return U, M, rmse_ev[iter-1]
    else:
        return U, M, rmse_ev


def M_init(R, n_m, p):

    M = np.zeros((p, n_m))   
    # Assign the average rating for the movies as the first row
    M[0, :] = np.nanmean(R, axis=0)  
    # Assign small random numbers in [0,1) for the remaining entries
    M[1:,:] = np.random.rand(p-1, n_m)

    return M



def ALS_predict(U, M):
    return U.T @ M



def calc_rmse(R_true, U, M):
    R_pred = ALS_predict(U, M)
    return np.sqrt(np.nanmean((R_true - R_pred)**2))


def holdout_cv_ALS(R, R_val, min_delta, max_iter, reg_params, ranks):
    """
    Hold-out cross validation for ALS.
    Parameters evaluated: reg_param, ranks
    
    Input:
    train_data
    validation_data
    num_iters
    reg_param = list of lambdas
    ranks = list of number of features (p)
    
    """
    # initial
    min_rmse = float('inf')
    best_p = -1
    best_regularization = 0
    best_U = None
    best_M = None
    
    # Outer loop on reg_param: fix lambda and try different number of features
    for reg_par in reg_params:
    
        for p in ranks:
            
            # 
            U, M, rmse = ALS_local_train(R, p, reg_par, R_val, min_delta, max_iter, progress=False)

            print('{} latent factors and regularization = {}: validation RMSE is {}'.format(p, reg_par, rmse))
            
            if rmse < min_rmse:
                min_rmse = rmse
                best_p = p
                best_regularization = reg_par
                best_U = U
                best_M = M
    
    print('\nThe best model has {} latent factors and regularization = {}'.format(best_p, best_regularization))
    return best_U, best_M
