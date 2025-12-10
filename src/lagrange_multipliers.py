"""
Implementation of the Lagrange multiplier method to adjust the dynamical matrix in order to obey 
the acoustic sum rule.

"""

import numpy as np

import numpy as np

def lagrangian(K, dimension=3):
    '''
    Lagrange-multiplier formalism to adjust force-constant matrix K in a way to fulfill the acoustic sum rule ASR. 
    The method is implemented according to:
    N. Mingo, D. A. Stewart, D. A. Broido and D. Srivastava, "Phonon transmission through defects in carbon nanotubes from first principles", 
    PHYSICAL REVIEW B 77, 033418 2008
    DOI: 10.1103/PhysRevB.77.033418
    '''

    def extract_displacement_modes(K, dimension):
        eigenvalues, eigenvectors = np.linalg.eigh(K)
        sorted_indices = np.argsort(eigenvalues)

        if dimension == 3:
            R = eigenvectors[:, sorted_indices[:6]].T 
        elif dimension == 2:
            R = eigenvectors[:, sorted_indices[:2]].T
        elif dimension == 1:
            R = eigenvectors[:, sorted_indices[:1]].T 

        return R 

    def calculate_B(K, R):
        n_constraints = R.shape[0]
        n_rows = K.shape[0]
        
        K_sqr = K**2
        term1_partial = np.einsum('ik, mk, nk -> nmi', K_sqr, R, R, optimize=True)
        
        B_term1 = np.zeros((n_constraints, n_constraints, n_rows, n_rows))
        for n in range(n_constraints):
            for m in range(n_constraints):
                np.fill_diagonal(B_term1[n, m, :, :], term1_partial[n, m, :])
        
        B_term1 *= 0.25
        B_term2 = 0.25 * np.einsum('mi, nj, ij -> nmij', R, R, K_sqr, optimize=True)
        
        B = B_term1 + B_term2
        return B 

    def calculate_a(K, R):
        a = -np.einsum('ik, mk -> mi', K, R, optimize=True)
        return a 

    def solve_lagrange_multipliers(B, a):

        n_constraints, n_rows = a.shape 
        size = n_constraints * n_rows
        B_full = B.transpose(0, 2, 1, 3).reshape(size, size)
        
        a_flat = a.reshape((size,)) 
        lambda_full = np.linalg.lstsq(B_full, a_flat, rcond=None)[0] 
        
        lambda_reshaped = lambda_full.reshape(n_constraints, n_rows)

        return lambda_reshaped 

    def calculate_D(K, R, lambda_):
        K_sqr = K**2
    
        term1_partial = np.einsum('mi, mj -> ij', lambda_, R, optimize=True)
        D_term1 = 0.25 * K_sqr * term1_partial

        term2_partial = np.einsum('mj, mi -> ij', lambda_, R, optimize=True)
        D_term2 = 0.25 * K_sqr * term2_partial
        
        D = D_term1 + D_term2
        
        return D 

    R = extract_displacement_modes(K, dimension)
    B = calculate_B(K, R)
    a = calculate_a(K, R)
    lambda_ = solve_lagrange_multipliers(B, a)
    D = calculate_D(K, R, lambda_)
    
    return K + D


if __name__ == "__main__":
    
    # 1D Version (original)
    print("=== 1D Version ===")
    lattice_constant = 3.0
    k_l = 1e-2
    k_r = 1e-2
    k_c = 1e-2
    k_xy = 0
    k_c_xy = 0
    N_L = 50
    N_R = 50
    N_C = 2
    delta = .5*1e-2

    K_1D = np.zeros((N_L + N_C + N_R, N_L + N_C + N_R))

    
    K_CC = np.array([[k_l + k_c , -k_c],
                    [-k_c, k_r + k_c]])
    
    K_LL = np.zeros((N_L, N_L))
    K_RR = np.zeros((N_R, N_R))

    K_LC = np.zeros((N_L, N_C))
    K_LC[-1, -1] = -k_l
    K_CL = K_LC.T

    K_RC = np.zeros((N_R, N_C))
    K_RC[0, 0] = -k_r
    K_CR = K_RC.T

    for i in range(N_L):
        for j in range(N_L-1):
            if i == j:
                K_LL[i, j] = 2*k_l
                K_LL[i+1, j] = -k_l 
                K_LL[i, j+1] = -k_l
    K_LL[-1, -1] = 2*k_l #+ delta
    K_LL[-2, -1] = -k_l + delta
    K_LL[-1, -2] = -k_l + delta
    K_LL[0, 0] = k_l

    for i in range(N_R):
        for j in range(N_R-1):
            if i == j:
                K_RR[i, j] = 2*k_r
                K_RR[i+1, j] = -k_r
                K_RR[i, j+1] = -k_r
    K_RR[0, 0] = 2*k_r #+ delta
    K_RR[-1, -1] = k_r

    K_1D[:N_L, :N_L] = K_LL
    K_1D[N_L:N_L+N_C, N_L:N_L+N_C] = K_CC
    K_1D[N_L+N_C:, N_L+N_C:] = K_RR

    K_1D[:N_L, N_L:N_L+N_C] = K_LC
    K_1D[N_L:N_L+N_C, :N_L] = K_CL

    K_1D[N_L+N_C:, N_L:N_L+N_C] = K_RC
    K_1D[N_L:N_L+N_C, N_L+N_C:] = K_CR

    #K_tilde_1D = lagrangian1(K_1D, dimension=1)
    #K_tilde_1D = lagrangian1(K_1D, dimension=1)
    K_tilde_1D = lagrangian(K_1D, dimension=1)
    print("K1D:\n", K_1D)
    print(K_tilde_1D)
    print(np.sum(K_tilde_1D))
    # 2D Version
    #print("\n=== 2D Version ===")
    
    

    delta2d = 0.
    delta2dd = 0.5

    K_2D = np.array([[ 100.,    0., -100.,    0.,    0.,    0.,    0.,    0.,    0.,
           0.,    0.,    0.],
       [   0.,    0.,    0.,    0.,    0.,    0.,    0.,    0.,    0.,
           0.,    0.,    0.],
       [-100.,    0.,  200.,    0., -100.,    0.,    0.,    0.,    0.,
           0.,    0.,    0.],
       [   0.,    0.,    0.,    0.,    0.,    0.,    0.,    0.,    0.,
           0.,    0.,    0.],
       [   0.,    0., -100.,    0.,  200 + delta2d,    0.,  -100 + delta2dd,    0.,    0.,
           0.,    0.,    0.],
       [   0.,    0.,    0.,    0.,    0.,    0.,    0.,    0.,    0.,
           0.,    0.,    0.],
       [   0.,    0.,    0.,    0.,  -100.+delta2dd,    0.,  200 + delta2d,    0., -100.,
           0.,    0.,    0.],
       [   0.,    0.,    0.,    0.,    0.,    0.,    0.,    0.,    0.,
           0.,    0.,    0.],
       [   0.,    0.,    0.,    0.,    0.,    0., -100.,    0.,  200.,
           0., -100.,    0.],
       [   0.,    0.,    0.,    0.,    0.,    0.,    0.,    0.,    0.,
           0.,    0.,    0.],
       [   0.,    0.,    0.,    0.,    0.,    0.,    0.,    0., -100.,
           0.,  100.,    0.],
       [   0.,    0.,    0.,    0.,    0.,    0.,    0.,    0.,    0.,
           0.,    0.,    0.]])
    
    #K2Tilde = lagrangian_outdated(K_2D, dimension=2)
    #print(K_2D)
    #print("K2Tilde:\n", K2Tilde)


    K3D = np.array([[ 2.65889094e-01, -1.39136905e-06, -9.27579365e-07,
        -2.65889094e-01,  1.39136905e-06,  9.27579365e-07],
       [-1.39136905e-06,  1.42791766e-04,  0.00000000e+00,
         1.39136905e-06, -1.42791766e-04, -0.00000000e+00],
       [-9.27579365e-07,  0.00000000e+00,  1.42791766e-04,
         9.27579365e-07, -0.00000000e+00, -1.42791766e-04],
       [-2.65889094e-01,  1.39136905e-06,  9.27579365e-07,
         2.65889094e-01, -1.39136905e-06, -9.27579365e-07],
       [ 1.39136905e-06, -1.42791766e-04, -0.00000000e+00,
        -1.39136905e-06,  1.42791766e-04,  0.00000000e+00],
       [ 9.27579365e-07, -0.00000000e+00, -1.42791766e-04,
        -9.27579365e-07,  0.00000000e+00,  1.42791766e-04]])

    K3D[0,1] *= 100
    K3D[1,0] *= 100
    K3Dnew = lagrangian(K3D, dimension=3)

    print("test")


    # Plot
    import matplotlib.pyplot as plt
    
    def plot_matrix_comparison():
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1D Original
        im1 = axes[0,0].imshow(K_1D, cmap='RdBu_r', aspect='equal')
        axes[0,0].set_title('1D Steifigkeitsmatrix K')
        plt.colorbar(im1, ax=axes[0,0])
        
        # 1D nach Lagrange
        im2 = axes[0,1].imshow(K_tilde_1D, cmap='RdBu_r', aspect='equal')  
        axes[0,1].set_title('1D nach Lagrange K_tilde')
        plt.colorbar(im2, ax=axes[0,1])
        
        plt.tight_layout()
        plt.show()


    