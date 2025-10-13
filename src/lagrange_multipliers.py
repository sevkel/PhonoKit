"""
Implementation of the Lagrange multiplier method to adjust the dynamical matrix in order to obey 
the acoustic sum rule.

"""

import numpy as np

def lagrangian1(K, dimension=3):
    '''
    Lagrange-multiplier formalism to adjust force-constant matrix K in a way to fulfill the acoustic sum rule ASR. 
    The method is implemented according to:
    N. Mingo, D. A. Stewart, D. A. Broido and D. Srivastava, "Phonon transmission through defects in carbon nanotubes from first principles", 
    PHYSICAL REVIEW B 77, 033418 2008
    DOI: 10.1103/PhysRevB.77.033418
    '''
    def extract_displacement_modes(K, dim=dimension):

        thresh = -1E-9
        
        eigenvalues, eigenvectors = np.linalg.eigh(K)  # eigh für symmetrische K
        sorted_indices = list()
        for i, eigval in enumerate(eigenvalues):
            if eigval > thresh:
                sorted_indices.append(i)

        if dim == 3:
            R = eigenvectors[:, sorted_indices[:6]].T
        elif dim == 2:
            R = eigenvectors[:, sorted_indices[:4]].T
        elif dim == 1:
            R = eigenvectors[:, sorted_indices[:1]].T 

        return R
    
    def calculate_B(K, R):

        n_constraints, n_rows = R.shape
        
        # Initialize B in the right form
        B = np.zeros((n_constraints, n_constraints, n_rows, n_rows))
        term1 = 0.25 * np.einsum('ik,mk,nk->mni', K**2, R, R)
        term2 = 0.25 * np.einsum('ij,mi,nj->mnij', K**2, R, R)
        
        # Construct B
        B += np.einsum('mni,ij->mnij', term1, np.eye(n_rows))  
        B += term2  

        return B
    
    def calculate_a(K, R):

        n_constraints, n = R.shape
        
        a = -np.einsum('ij,mj->mi', K, R)

        return a
    
    def solve_lagrange_multipliers(B, a):
        
        n_constraints, n_rows = a.shape  
        
        # Reshape B to (n_constraints * n_rows) x (n_constraints * n_rows) matrix
        B_full = B.transpose(0, 2, 1, 3).reshape(n_constraints * n_rows, n_constraints * n_rows)
        
        # Reshape a to a vector of shape (n_constraints * n_rows)
        a_full = a.reshape(n_constraints * n_rows)
        
        try:
            lambda_full = np.linalg.solve(B_full, a_full)
        except np.linalg.LinAlgError as e:
            print("Singular matrix: {}".format(e))
            # try a regularization
            lambda_full = np.linalg.solve(B_full + 1e-8 * np.eye(B_full.shape[0]), a_full)
            #lambda_full = np.linalg.pinv(B_full) @ a_full  # Verwende Pseudoinverse als Fallback
        
        # Back to (n_constraints, n_rows) form
        lambda_reshaped = lambda_full.reshape(n_constraints, n_rows)

        return lambda_reshaped
    
    def calculate_D(K, R, lambda_):
    
        # Berechne D effizient mit np.einsum
        D = 0.25 * np.einsum('ij,mi,mj->ij', K**2, lambda_, R)
        D += 0.25 * np.einsum('ij,mj,mi->ij', K**2, lambda_, R)

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
    k_l = 100
    k_r = 100
    k_c = 100
    k_xy = 0
    k_c_xy = 0
    N_L = 300
    N_R = 300
    N_C = 2
    delta = 0.5

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
    K_LL[-1, -1] = 2*k_l + delta
    K_LL[0, 0] = k_l

    for i in range(N_R):
        for j in range(N_R-1):
            if i == j:
                K_RR[i, j] = 2*k_r
                K_RR[i+1, j] = -k_r
                K_RR[i, j+1] = -k_r
    K_RR[0, 0] = 2*k_r + delta
    K_RR[-1, -1] = k_r

    K_1D[:N_L, :N_L] = K_LL
    K_1D[N_L:N_L+N_C, N_L:N_L+N_C] = K_CC
    K_1D[N_L+N_C:, N_L+N_C:] = K_RR

    K_1D[:N_L, N_L:N_L+N_C] = K_LC
    K_1D[N_L:N_L+N_C, :N_L] = K_CL

    K_1D[N_L+N_C:, N_L:N_L+N_C] = K_RC
    K_1D[N_L:N_L+N_C, N_L+N_C:] = K_CR

    K_tilde_1D = lagrangian1(K_1D, dimension=1)
    print("K1D:\n", K_1D)
    print(K_tilde_1D)
    # 2D Version
    #print("\n=== 2D Version ===")
    
    

    delta2d = 0

    K_2D = np.array([[ 100.,    0., -100.,    0.,    0.,    0.,    0.,    0.,    0.,
           0.,    0.,    0.],
       [   0.,    0.,    0.,    0.,    0.,    0.,    0.,    0.,    0.,
           0.,    0.,    0.],
       [-100.,    0.,  200.,    0., -100.,    0.,    0.,    0.,    0.,
           0.,    0.,    0.],
       [   0.,    0.,    0.,    0.,    0.,    0.,    0.,    0.,    0.,
           0.,    0.,    0.],
       [   0.,    0., -100.,    0.,  200 + delta2d,    0.,  -100.,    0.,    0.,
           0.,    0.,    0.],
       [   0.,    0.,    0.,    0.,    0.,    0.,    0.,    0.,    0.,
           0.,    0.,    0.],
       [   0.,    0.,    0.,    0.,  -100.,    0.,  200 + delta2d,    0., -100.,
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


   #K2Tilde = lagrangian1(K_2D, dimension=2)
    #print(K_2D)
    #print("K2Tilde:\n", K2Tilde)



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


    