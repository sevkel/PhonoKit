import numpy as np


def make_sum_zero(matrix):
    # Berechne Zeilen- und Spaltensummen
    row_sums = np.sum(matrix, axis=1)
    col_sums = np.sum(matrix, axis=0)
    total_sum = np.sum(matrix)
    
    # Dimensionen der Matrix
    n, m = matrix.shape
    
    # Erstelle die Korrekturmatrix
    correction_matrix = np.zeros((n, m))
    for i in range(n):
        for j in range(m):
            correction_matrix[i, j] = -(row_sums[i] + col_sums[j] - total_sum / (n * m))
    
    # Addiere die Korrekturmatrix zur ursprünglichen Matrix
    result_matrix = matrix + correction_matrix
    
    return result_matrix

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

if __name__ == '__main__':

    lattice_constant = 3.0
    k_x = 0.1
    k_y = 0.1
    k_xy = 0.33
    k_c = 0.1
    k_c_xy = 1.0
    interaction_range = 2
    N_y = 4
    N_y_scatter = 2


    all_k_x, all_k_y, all_k_xy = ranged_force_constant(lattice_constant, k_x, k_y, k_xy, k_c, k_c_xy, interaction_range)[0:3]


    hNN = np.zeros((2 * N_y * interaction_range, 2 * N_y * interaction_range))
    #build Hessian matrix for the hNN principal bulklayer

    for i in range(interaction_range):

        for j in range(i * 2 * N_y, i * 2 * N_y + 2 * N_y):
            
            # diagonal elements x and xy coupling
            if j % 2 == 0:
                
                # ii-coupling
                hNN[j, j] = sum(2 * all_k_x[k][1] for k in range(len(all_k_x))) 

                # ij-coupling in h01

                # xy-coupling
                if j == 0 + i * 2 * N_y or j == i * 2 * N_y + 2 * N_y - 2:
                    hNN[j, j + 1] = 2 * all_k_xy[0][1]
                    hNN[j + 1, j] = 2 * all_k_xy[0][1]

                if j != 0 + i * 2 * N_y and j != i * 2 * N_y + 2 * N_y - 2 and N_y > 2:
                    hNN[j, j + 1] = 4 * all_k_xy[0][1]
                    hNN[j + 1, j] = 4 * all_k_xy[0][1]
                

            else:
                # y coupling in the coupling range -> edge layers
                if (j == i * 2 * N_y + 1) or (j == i * 2 * N_y + 2 * N_y - 1): 
                    
                    # xy-coupling
                    atomnr = np.ceil(float(j) / 2)

                    if atomnr < N_y and interaction_range > 1:#0 <= int(j - 1 + 2 * (atomnr + N_y + 1) - 1) < hNN.shape[1]:
                        hNN[j - 1, int(j - 1 + 2 * (atomnr + N_y + 1) - 1)] = -all_k_xy[0][1]
                        hNN[j, int(j - 1 + 2 * (atomnr + N_y + 1) - 2)] = -all_k_xy[0][1]

                    elif atomnr ==  N_y and interaction_range > 1:
                        hNN[j - 1, int(2 * (atomnr + N_y - 1) - 1)] = -all_k_xy[0][1]
                        hNN[j, int(2 * (atomnr + N_y - 1) - 2)] = -all_k_xy[0][1]

                    elif atomnr > N_y and interaction_range > 1:
                        hNN[j, int(j - 1 - 2 * (atomnr - N_y + 1))] = -all_k_xy[0][1]
                        hNN[j - 1, int(j - 1 - 2 * (atomnr - N_y + 1) + 1)] = -all_k_xy[0][1]


                    hNN[j, j] = all_k_y[0][1]

                    if j == 1 + i * 2 * N_y:
                        hNN[j, j + 2] = -all_k_y[0][1]
                    else:
                        hNN[j, j - 2] = -all_k_y[0][1]

                    if interaction_range >= N_y:
                        for k in range(1, N_y - 1):
                            hNN[j, j] += all_k_y[k][1]
                            
                            if j + 2 * (k + 1) < i * 2 * N_y + 2 * N_y:
                                hNN[j, j + 2 * (k + 1)] = -all_k_y[k][1]
                            if j - 2 * (k + 1) >= 0 + i * 2 * N_y:
                                hNN[j, j - 2 * (k + 1)] = -all_k_y[k][1]

                    else:
                        for k in range(1, interaction_range):
                            hNN[j, j] += all_k_y[k][1]
                        
                            if j + 2 * (k + 1) < i * 2 * N_y + 2 * N_y:
                                hNN[j, j + 2 * (k + 1)] = -all_k_y[k][1]
                            if j - 2 * (k + 1) >= 0 + i * 2 * N_y:
                                hNN[j, j - 2 * (k + 1)] = -all_k_y[k][1]


                else:
                    
                    atomnr = np.ceil(float(j) / 2)
                    hNN[j, j] = 2 * all_k_y[0][1]
                    
                    # xy-coupling inner atom
                    if atomnr < N_y and interaction_range > 1:
                        ## first layer
                        # first atom
                        hNN[j - 1, int(2 * (atomnr + N_y - 1)) - 1] = -all_k_xy[0][1]
                        hNN[j, int(2 * (atomnr + N_y - 1)) - 2] = -all_k_xy[0][1]

                        #second atom
                        hNN[j - 1, int(2 * (atomnr + N_y + 1)) - 1] = -all_k_xy[0][1]
                        hNN[j, int(2 * (atomnr + N_y + 1)) - 2] = -all_k_xy[0][1]

                    elif atomnr > i * N_y and (i + 1) * N_y == N_y * interaction_range:
                        ## last layer
                        # first atom
                        hNN[j - 1, int(2 * (atomnr - N_y - 1)) - 1] = -all_k_xy[0][1]
                        hNN[j, int(2 * (atomnr - N_y - 1)) - 2] = -all_k_xy[0][1]

                        # second atom
                        hNN[j - 1, int(2 * (atomnr - N_y + 1)) - 1] = -all_k_xy[0][1]
                        hNN[j, int(2 * (atomnr - N_y + 1)) - 2] = -all_k_xy[0][1]
                    
                    elif atomnr > i * N_y and atomnr < (i + 1) * N_y and (i + 1) * N_y < N_y * interaction_range:
                        ## layer before
                        # first atom
                        hNN[j - 1, int(2 * (atomnr - N_y - 1)) - 1] = -all_k_xy[0][1]
                        hNN[j, int(2 * (atomnr - N_y - 1)) - 2] = -all_k_xy[0][1]
                        # second atom
                        hNN[j - 1, int(2 * (atomnr - N_y + 1)) - 1] = -all_k_xy[0][1]
                        hNN[j, int(2 * (atomnr - N_y + 1)) - 2] = -all_k_xy[0][1]

                        ## layer after
                        # first atom
                        hNN[j - 1, int(2 * (atomnr + N_y - 1)) - 1] = -all_k_xy[0][1]
                        hNN[j, int(2 * (atomnr + N_y - 1)) - 2] = -all_k_xy[0][1]
                        # second atom
                        hNN[j - 1, int(2 * (atomnr + N_y + 1)) - 1] = -all_k_xy[0][1]
                        hNN[j, int(2 * (atomnr + N_y + 1)) - 2] = -all_k_xy[0][1]

                        

                    if j + 2 < i * 2 * N_y + 2 * N_y:
                        hNN[j, j + 2] = -all_k_y[0][1]
                    if j - 2 >= 0 + i * 2 * N_y:
                        hNN[j, j - 2] = -all_k_y[0][1]


                    if interaction_range >= N_y:
                        for k in range(1, N_y - 1):
                            if atomnr - k - 1 > i * N_y and atomnr + k < i * N_y + N_y:
                                hNN[j, j] += 2 * all_k_y[k][1]
                                
                            elif (atomnr - k - 1 <= i * N_y and atomnr + k < i * N_y + N_y) or (atomnr - k - 1 > i * N_y and atomnr + k >= i * N_y + N_y):
                                hNN[j, j] += all_k_y[k][1]
                            
                            if j + 2 * (k + 1) < i * 2 * N_y + 2 * N_y:
                                hNN[j, j + 2 * (k + 1)] = -all_k_y[k][1]
                            if j - 2 * (k + 1) >= 0:
                                hNN[j, j - 2 * (k + 1)] = -all_k_y[k][1]

                    else:
                        for k in range(1, interaction_range):
                            if atomnr - k - 1 > 0 + i * 2 * N_y and atomnr + k < N_y * interaction_range:
                                hNN[j, j] += 2 * all_k_y[k][1]
                            elif (atomnr - k - 1 <= i * N_y and atomnr + k < i * N_y + N_y) or (atomnr - k - 1 > i * N_y and atomnr + k >= i * N_y + N_y):
                                hNN[j, j] += all_k_y[k][1]
                            
                            if j + 2 * (k + 1) < i * 2 * N_y + 2 * N_y:
                                hNN[j, j + 2 * (k + 1)] = -all_k_y[k][1]
                            if j - 2 * (k + 1) >= 0 + i * 2 * N_y:
                                hNN[j, j - 2 * (k + 1)] = -all_k_y[k][1]

    print(hNN)


    h00 = np.zeros((2 * N_y * interaction_range, 2 * N_y * interaction_range))
    #build Hessian matrix for the h00 principal surface layer

    for i in range(interaction_range):

        for j in range(i * 2 * N_y, i * 2 * N_y + 2 * N_y):
            
            # diagonal elements x and xy coupling
            if j % 2 == 0:

                atomnr = np.ceil(float(j + 1) / 2)
                
                # ii-coupling
                if atomnr <= N_y and interaction_range > 1:
                    ## first layer
                    h00[j, j] = sum(all_k_x[k][1] for k in range(len(all_k_x)))

                elif atomnr > i * N_y and (i + 1) * N_y == N_y * interaction_range:
                    ## last layer
                    for k in range(interaction_range):
                        if i - k > 0:
                            h00[j, j] += 2 * all_k_x[k][1]
                        else:
                            h00[j, j] += all_k_x[k][1]
                
                elif atomnr > i * N_y and atomnr < (i + 1) * N_y and (i + 1) * N_y < N_y * interaction_range:
                    for k in range(interaction_range):
                        if i - k > 0:
                            h00[j, j] += 2 * all_k_x[k][1]
                        else:
                            h00[j, j] += all_k_x[k][1]

                for k in range(interaction_range):
                    if j + 2 * (k + 1) * N_y < h00.shape[0]:                    
                        h00[j, j + 2 * (k + 1) * N_y] = -all_k_x[k][1]
                    if j - 2 * (k + 1) * N_y >= 0:
                        h00[j, j - 2 * (k + 1) * N_y] = -all_k_x[k][1]

                # xy-coupling
                if j == 0 or j == 2 * N_y - 2:
                    h00[j, j + 1] = all_k_xy[0][1]
                    h00[j + 1, j] = all_k_xy[0][1]

                elif j < 2 * N_y - 2:
                    h00[j, j + 1] = 2 * all_k_xy[0][1]
                    h00[j + 1, j] = 2 * all_k_xy[0][1]

                elif (j == i * 2 * N_y or j == i * 2 * N_y + 2 * N_y - 2) and (j != 0 and j != 2 * N_y - 2):
                    h00[j, j + 1] = 2 * all_k_xy[0][1]
                    h00[j + 1, j] = 2 * all_k_xy[0][1]

                elif j != 0 + i * 2 * N_y and j != i * 2 * N_y + 2 * N_y - 2 and N_y > 2:
                    h00[j, j + 1] = 4 * all_k_xy[0][1]
                    h00[j + 1, j] = 4 * all_k_xy[0][1]
                

            else:
                # y coupling in the coupling range -> edge layers
                if (j == i * 2 * N_y + 1) or (j == i * 2 * N_y + 2 * N_y - 1): 
                    
                    # xy-coupling
                    atomnr = np.ceil(float(j) / 2)

                    if atomnr < N_y and interaction_range > 1:
                        print(atomnr, j)
                        h00[j - 1, int(j - 1 + 2 * (atomnr + N_y + 1) - 1)] = -all_k_xy[0][1]
                        h00[j, int(j - 1 + 2 * (atomnr + N_y + 1) - 2)] = -all_k_xy[0][1]

                    elif atomnr ==  N_y and interaction_range > 1:
                        h00[j - 1, int(2 * (atomnr + N_y - 1) - 1)] = -all_k_xy[0][1]
                        h00[j, int(2 * (atomnr + N_y - 1) - 2)] = -all_k_xy[0][1]

                    elif atomnr > N_y and interaction_range > 1:
                        h00[j, int(j - 1 - 2 * (atomnr - N_y + 1))] = -all_k_xy[0][1]
                        h00[j - 1, int(j - 1 - 2 * (atomnr - N_y + 1) + 1)] = -all_k_xy[0][1]
                    


                    h00[j, j] = all_k_y[0][1]

                    if j == 1 + i * 2 * N_y:
                        h00[j, j + 2] = -all_k_y[0][1]
                    else:
                        h00[j, j - 2] = -all_k_y[0][1]

                    if interaction_range >= N_y:
                        for k in range(1, N_y - 1):
                            h00[j, j] += all_k_y[k][1]
                            
                            if j + 2 * (k + 1) < i * 2 * N_y + 2 * N_y:
                                h00[j, j + 2 * (k + 1)] = -all_k_y[k][1]
                            if j - 2 * (k + 1) >= 0 + i * 2 * N_y:
                                h00[j, j - 2 * (k + 1)] = -all_k_y[k][1]

                    else:
                        for k in range(1, interaction_range):
                            h00[j, j] += all_k_y[k][1]
                        
                            if j + 2 * (k + 1) < i * 2 * N_y + 2 * N_y:
                                h00[j, j + 2 * (k + 1)] = -all_k_y[k][1]
                            if j - 2 * (k + 1) >= 0 + i * 2 * N_y:
                                h00[j, j - 2 * (k + 1)] = -all_k_y[k][1]


                else:
                    
                    atomnr = np.ceil(float(j) / 2)
                    h00[j, j] = 2 * all_k_y[0][1]
                    
                    # xy-coupling inner atom
                    if atomnr < N_y and interaction_range > 1:
                        ## first layer
                        # first atom
                        h00[j - 1, int(2 * (atomnr + N_y - 1)) - 1] = -all_k_xy[0][1]
                        h00[j, int(2 * (atomnr + N_y - 1)) - 2] = -all_k_xy[0][1]

                        #second atom
                        h00[j - 1, int(2 * (atomnr + N_y + 1)) - 1] = -all_k_xy[0][1]
                        h00[j, int(2 * (atomnr + N_y + 1)) - 2] = -all_k_xy[0][1]

                    elif atomnr > i * N_y and (i + 1) * N_y == N_y * interaction_range:
                        ## last layer
                        # first atom
                        h00[j - 1, int(2 * (atomnr - N_y - 1)) - 1] = -all_k_xy[0][1]
                        h00[j, int(2 * (atomnr - N_y - 1)) - 2] = -all_k_xy[0][1]

                        # second atom
                        h00[j - 1, int(2 * (atomnr - N_y + 1)) - 1] = -all_k_xy[0][1]
                        h00[j, int(2 * (atomnr - N_y + 1)) - 2] = -all_k_xy[0][1]
                    
                    elif atomnr > i * N_y and atomnr < (i + 1) * N_y and (i + 1) * N_y < N_y * interaction_range:
                        ## layer before
                        # first atom
                        h00[j - 1, int(2 * (atomnr - N_y - 1)) - 1] = -all_k_xy[0][1]
                        h00[j, int(2 * (atomnr - N_y - 1)) - 2] = -all_k_xy[0][1]
                        # second atom
                        h00[j - 1, int(2 * (atomnr - N_y + 1)) - 1] = -all_k_xy[0][1]
                        h00[j, int(2 * (atomnr - N_y + 1)) - 2] = -all_k_xy[0][1]

                        ## layer after
                        # first atom
                        h00[j - 1, int(2 * (atomnr + N_y - 1)) - 1] = -all_k_xy[0][1]
                        h00[j, int(2 * (atomnr + N_y - 1)) - 2] = -all_k_xy[0][1]
                        # second atom
                        h00[j - 1, int(2 * (atomnr + N_y + 1)) - 1] = -all_k_xy[0][1]
                        h00[j, int(2 * (atomnr + N_y + 1)) - 2] = -all_k_xy[0][1]

                        

                    if j + 2 < i * 2 * N_y + 2 * N_y:
                        h00[j, j + 2] = -all_k_y[0][1]
                    if j - 2 >= 0 + i * 2 * N_y:
                        h00[j, j - 2] = -all_k_y[0][1]


                    if interaction_range >= N_y:
                        for k in range(1, N_y - 1):
                            if atomnr - k - 1 > i * N_y and atomnr + k < i * N_y + N_y:
                                h00[j, j] += 2 * all_k_y[k][1]
                                
                            elif (atomnr - k - 1 <= i * N_y and atomnr + k < i * N_y + N_y) or (atomnr - k - 1 > i * N_y and atomnr + k >= i * N_y + N_y):
                                h00[j, j] += all_k_y[k][1]
                            
                            if j + 2 * (k + 1) < i * 2 * N_y + 2 * N_y:
                                h00[j, j + 2 * (k + 1)] = -all_k_y[k][1]
                            if j - 2 * (k + 1) >= 0:
                                h00[j, j - 2 * (k + 1)] = -all_k_y[k][1]

                    else:
                        for k in range(1, interaction_range):
                            if atomnr - k - 1 > 0 + i * 2 * N_y and atomnr + k < N_y * interaction_range:
                                h00[j, j] += 2 * all_k_y[k][1]
                            elif (atomnr - k - 1 <= i * N_y and atomnr + k < i * N_y + N_y) or (atomnr - k - 1 > i * N_y and atomnr + k >= i * N_y + N_y):
                                h00[j, j] += all_k_y[k][1]
                            
                            if j + 2 * (k + 1) < i * 2 * N_y + 2 * N_y:
                                h00[j, j + 2 * (k + 1)] = -all_k_y[k][1]
                            if j - 2 * (k + 1) >= 0 + i * 2 * N_y:
                                h00[j, j - 2 * (k + 1)] = -all_k_y[k][1]

    print('\n', h00)
            

    h01 = np.zeros((2 * N_y * interaction_range, 2 * N_y * interaction_range))
    # build Hessian matrix for the h01 interaction between principal layers
    # rows are A layer atoms, columns are B layer atoms

    for k in range(N_y):
        # direct x coupling
        h01[2 * (interaction_range * N_y - k) - 2, 2 * (N_y - k) - 2] = -all_k_x[0][1]

        #direct xy coupling
        h01[2 * (interaction_range * N_y - k) - 2, 2 * k + 1] = -all_k_xy[0][1]
        h01[2 * (interaction_range * N_y - k) - 1, 2 * k] = -all_k_xy[0][1]

    for i in range(0, h01.shape[0], 2):
        
        h01[i, i] = -all_k_x[-1][1]
        atomnr = np.ceil(float(i + 1) / 2)

        for k in range(1, interaction_range):

            if 2 * k * N_y - 2 < i < 2 * (k + 1) * N_y - 2:
                h01[i, i - 2 * k * N_y] = -all_k_x[-1-k][1]

        #check in which layer the atom is
        for k in range(1, interaction_range + 1):
            if (k - 1) * N_y < atomnr <= k * N_y:

                for h in range(1, k):
                    h01[i, i - 2 * h * N_y] = -all_k_x[-1-h][1]
                    

    print('\n', h01)


    # build Hessian matrix for the h01 interaction between principal layers
    # rows are A layer atoms, columns are B layer atoms
    
    #TODO: If N_y scatter == N_y electrode -> K_LC == h01

    if N_y_scatter == N_y:
        k_lc_LL = h01


    direct_interaction = np.zeros((2 * (N_y_scatter + 2), 2 * N_y_scatter), dtype=float)

    for i in range(0, direct_interaction.shape[0], 2):
        
        if 0 < i <= direct_interaction.shape[1]:
            direct_interaction[i, i - 2] = -all_k_x[0][1]
            
            if i - 2 + 3 <= direct_interaction.shape[1] - 1:
                direct_interaction[i, i - 2 + 3] = -all_k_xy[0][1]
                direct_interaction[i + 1, i - 2 + 2] = -all_k_xy[0][1]
            if i - 2 - 2 >= 0:
                direct_interaction[i, i - 2 - 1] = -all_k_xy[0][1]
                direct_interaction[i + 1, i - 2 - 2] = -all_k_xy[0][1]

        # xy coupling
        if i == 0:
            direct_interaction[i, i + 1] = -all_k_xy[0][1]
            direct_interaction[i + 1, i] = -all_k_xy[0][1]

        elif i == direct_interaction.shape[0] - 2:
            direct_interaction[i, direct_interaction.shape[1] - 1] = -all_k_xy[0][1]
            direct_interaction[i + 1, direct_interaction.shape[1] - 2] = -all_k_xy[0][1]



    print('\n', direct_interaction)


    #TODO: get on the fly force constants between electrode and junction (or anything?)
    k_lc_LL = np.zeros((2 * N_y * interaction_range, 2 * N_y * interaction_range), dtype=float) 

    for i in range(interaction_range):
        for j in range(i * 2 * N_y, i * 2 * N_y + 2 * N_y, 2):
            if j % 2 == 0:
                atomnr = np.ceil(float(j + 1) / 2)

                if i * N_y + 1 + (N_y - N_y_scatter) / 2 <= atomnr <= i * N_y + N_y - (N_y - N_y_scatter) / 2:
                    k_lc_LL[j, j] = all_k_x[-1][1]

                if i == interaction_range - 1 and atomnr == i * N_y +  (N_y - N_y_scatter) / 2:

                    atomnr_b = int(1 + (N_y - N_y_scatter) / 2)
                    k_lc_LL[j: j + direct_interaction.shape[0], 2 * (atomnr_b - 1): 2 * (atomnr_b - 1) + direct_interaction.shape[1]] = direct_interaction
                    

    print('\n', k_lc_LL)


    #TODO: get on the fly force constants between electrode and junction (or anything?)
    k_LC = np.zeros((2 * N_y * interaction_range, 2 * N_y_scatter * interaction_range), dtype=float) 
    scatter_temp = 0

    for i in range(interaction_range):
        for j in range(i * 2 * N_y, i * 2 * N_y + 2 * N_y, 2):
            if j % 2 == 0:
                atomnr = np.ceil(float(j + 1) / 2)

                
                for l in range(i * 2 * N_y_scatter, i * 2 * N_y_scatter + 2 * N_y_scatter, 2):

                    if l % 2 == 0:

                        atomnr_b = np.ceil(float(l + 1) / 2)

                        if i * N_y + 1 + (N_y - N_y_scatter) / 2 <= atomnr <= i * N_y + N_y - (N_y - N_y_scatter) / 2 and atomnr_b > scatter_temp:

                            k_LC[j, l] = -all_k_x[-1][1]
                            scatter_temp = atomnr_b
                            break
                            
                        if i == interaction_range - 1 and atomnr == i * N_y + (N_y - N_y_scatter) / 2:

                            k_LC[j: j + direct_interaction.shape[0], 0: 0 + direct_interaction.shape[1]] = direct_interaction


    print('\n', k_LC)


