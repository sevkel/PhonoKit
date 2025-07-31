import numpy as np


def ranged_force_constant(lattice_constant, k_x, k_y, k_xy, k_c, k_c_xy, interaction_range, interact_potential="reciproke_squared"):
    """
    Calculate ranged force constants for the 2D Ribbon electrode dependend on which potential is used and on how many neighbors are coupled.
    
    Retruns:
        range_force_constant (list of tuples): Ranged force constant for the 2D lattice
    """

    match interact_potential:

        case "reciproke_squared":
            all_k_x = list(enumerate((k_x * (1 / (i * 1)**2) for i in range(1, interaction_range + 1))))
            all_k_y = list(enumerate((k_y * (1 / (i * 1)**2) for i in range(1, interaction_range + 1))))
            all_k_xy =  list(enumerate((k_xy * (1 / (i * 1)**2) for i in range(1, interaction_range + 1))))
            all_k_c_x = list(enumerate((k_c * (1 / (i * 1)**2) for i in range(1, interaction_range + 1))))
            all_k_c_xy = list(enumerate((k_c_xy * (1 / (i * 1)**2) for i in range(1, interaction_range + 1))))
        
        case _:
            raise ValueError("Invalid interaction potential. Choose either 'reciproke_squared', .")
        
    return all_k_x, all_k_y, all_k_xy, all_k_c_x, all_k_c_xy

def lagrangian(K, dimension=3):
    '''
    Same but with np.einsum
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
        
        # Initialisiere B mit der richtigen Form
        B = np.zeros((n_constraints, n_constraints, n_rows, n_rows))
        
        # Berechne term1 effizient mit np.einsum
        term1 = 0.25 * np.einsum('ik,mk,nk->mni', K**2, R, R)
        
        # Berechne term2 effizient mit np.einsum
        term2 = 0.25 * np.einsum('ij,mi,nj->mnij', K**2, R, R)
        
        # Setze die Werte in B
        B += np.einsum('mni,ij->mnij', term1, np.eye(n_rows))  # Diagonalbeitrag von term1
        B += term2  # term2 direkt hinzufügen

        return B
    
    def calculate_a(K, R):

        n_constraints, n = R.shape
        
        # Berechne a effizient mit np.einsum
        a = -np.einsum('ij,mj->mi', K, R)

        return a
    
    def solve_lagrange_multipliers(B, a):
        
        n_constraints, n_rows = a.shape  # (6,36)
        
        # Reshape B zu einer (n_constraints * n_rows) x (n_constraints * n_rows) Matrix
        B_full = B.transpose(0, 2, 1, 3).reshape(n_constraints * n_rows, n_constraints * n_rows)
        
        # Reshape a zu einem Vektor der Länge (n_constraints * n_rows)
        a_full = a.reshape(n_constraints * n_rows)
        
        try:
            # Lösen des Gleichungssystems
            lambda_full = np.linalg.solve(B_full, a_full)
        except np.linalg.LinAlgError as e:
            print("Singular matrix: {}".format(e))
            # try a regularization
            lambda_full = np.linalg.solve(B_full + 1e-8 * np.eye(B_full.shape[0]), a_full)
            #lambda_full = np.linalg.pinv(B_full) @ a_full  # Verwende Pseudoinverse als Fallback
        
        # Zurück in (n_constraints, n_rows) Form bringen
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

if __name__ == '__main__':

    lattice_constant = 3.0
    k_x = 1.0
    k_y = 1.0
    k_xy = 0.33
    k_c = 1.0
    k_c_xy = 1.0
    interaction_range = 2
    N_y = 3

    ranged_force_constant(lattice_constant, k_x, k_y, k_xy, k_c, k_c_xy, interaction_range)

    all_k_x, all_k_y, all_k_xy = ranged_force_constant(lattice_constant, k_x, k_y, k_xy, k_c, k_c_xy, interaction_range)[0:3]

    K_cc = np.array([[ 200,    0., -100.,    0.],
				[   0.,    0.,    0.,    0.],
				[-100.,    0.,  200,    0.],
				[   0.,    0.,    0.,    0.]])
    
    # K_lc mit 20 atomen als elektrode, genauso K_rc
    K_lc = np.array([[ 200,    0., -100.,    0.],
                    [   0.,    0.,    0.,    0.],
                    [-100.,    0.,  200,    0.],
                    [   0.,    0.,    0.,    0.],
                    [   0.,    0.,    0.,    0.]])
    
    '''K = np.array([[100, -95],
                    [-95, 100]])'''
    
    Ktilde = lagrangian(K, dimension=2)

    print("Ktilde:")
    print(Ktilde)