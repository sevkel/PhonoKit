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


    hNN = np.zeros((2 * N_y * interaction_range, 2 * N_y * interaction_range))
    #build Hessian matrix for the hNN principal bulklayer

    for i in range(interaction_range):

        for j in range(i * 2 * N_y, i * 2 * N_y + 2 * N_y):
            
            # diagonal elements x and xy coupling
            if j % 2 == 0:
                
                # ii-coupling
                hNN[j, j] = sum(2 * all_k_x[k][1] for k in range(len(all_k_x))) 

                # ij-coupling
                for k in range(interaction_range):
                    if j + 2 * (k + 1) * N_y < hNN.shape[0]:                    
                        hNN[j, j + 2 * (k + 1) * N_y] = -all_k_x[k][1]
                    if j - 2 * (k + 1) * N_y >= 0:
                        hNN[j, j - 2 * (k + 1) * N_y] = -all_k_x[k][1]

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

                    if atomnr < N_y:#0 <= int(j - 1 + 2 * (atomnr + N_y + 1) - 1) < hNN.shape[1]:
                        hNN[j - 1, int(j - 1 + 2 * (atomnr + N_y + 1) - 1)] = -all_k_xy[0][1]
                        hNN[j, int(j - 1 + 2 * (atomnr + N_y + 1) - 2)] = -all_k_xy[0][1]

                    elif atomnr ==  N_y:
                        hNN[j - 1, int(2 * (atomnr + N_y - 1) - 1)] = -all_k_xy[0][1]
                        hNN[j, int(2 * (atomnr + N_y - 1) - 2)] = -all_k_xy[0][1]

                    elif atomnr > N_y:
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
                                print("atomnr: ", atomnr, k)
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
                                print("atomnr: ", atomnr, k)
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

                # ij-coupling
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
                    if atomnr < N_y:#0 <= int(j - 1 + 2 * (atomnr + N_y + 1) - 1) < hNN.shape[1]:
                        h00[j - 1, int(j - 1 + 2 * (atomnr + N_y + 1) - 1)] = -all_k_xy[0][1]
                        h00[j, int(j - 1 + 2 * (atomnr + N_y + 1) - 2)] = -all_k_xy[0][1]

                    elif atomnr ==  N_y:
                        h00[j - 1, int(2 * (atomnr + N_y - 1) - 1)] = -all_k_xy[0][1]
                        h00[j, int(2 * (atomnr + N_y - 1) - 2)] = -all_k_xy[0][1]

                    elif atomnr > N_y:
                        h00[j, int(j - 1 - 2 * (atomnr - N_y + 1))] = -all_k_xy[0][1]
                        h00[j - 1, int(j - 1 - 2 * (atomnr - N_y + 1) + 1)] = -all_k_xy[0][1]
                    """if 0 <= int(j - 1 + 2 * (atomnr + N_y + 1) - 1) < h00.shape[1]:
                        h00[j - 1, int(j - 1 + 2 * (atomnr + N_y + 1) - 1)] = -all_k_xy[0][1]
                        h00[j, int(j - 1 + 2 * (atomnr + N_y + 1) - 2)] = -all_k_xy[0][1]
                    elif atomnr - N_y >= 0:
                        h00[j, int(j - 1 - 2 * (atomnr - N_y + 1))] = -all_k_xy[0][1]
                        h00[j - 1, int(j - 1 - 2 * (atomnr - N_y + 1) + 1)] = -all_k_xy[0][1]"""


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
                                print("atomnr: ", atomnr, k)
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
                                print("atomnr: ", atomnr, k)
                                h00[j, j] += all_k_y[k][1]
                            
                            if j + 2 * (k + 1) < i * 2 * N_y + 2 * N_y:
                                h00[j, j + 2 * (k + 1)] = -all_k_y[k][1]
                            if j - 2 * (k + 1) >= 0 + i * 2 * N_y:
                                h00[j, j - 2 * (k + 1)] = -all_k_y[k][1]

    print('\n', h00)
            

    h01 = np.zeros((2 * N_y * interaction_range, 2 * N_y * interaction_range))
        #build Hessian matrix for the h01 interaction between principal layers

    for i in range(interaction_range):

        for j in range(i * 2 * N_y, i * 2 * N_y + 2 * N_y):
            
            # diagonal elements x and xy coupling
            if j % 2 == 0:

                atomnr = np.ceil(float(j + 1) / 2)

                # ij-coupling
                for k in range(interaction_range):
                    if j + 2 * (k + 1) * N_y < h01.shape[0]:                    
                        h01[j, j + 2 * (k + 1) * N_y] = -all_k_x[k][1]
                    if j - 2 * (k + 1) * N_y >= 0:
                        h01[j, j - 2 * (k + 1) * N_y] = -all_k_x[k][1]

            else:
                # y coupling in the coupling range -> edge layers
                if (j == i * 2 * N_y + 1) or (j == i * 2 * N_y + 2 * N_y - 1): 
                    
                    # xy-coupling
                    atomnr = np.ceil(float(j) / 2)

                    if atomnr < N_y:#0 <= int(j - 1 + 2 * (atomnr + N_y + 1) - 1) < hNN.shape[1]:
                        h01[j - 1, int(j - 1 + 2 * (atomnr + N_y + 1) - 1)] = -all_k_xy[0][1]
                        h01[j, int(j - 1 + 2 * (atomnr + N_y + 1) - 2)] = -all_k_xy[0][1]

                    elif atomnr ==  N_y:
                        h01[j - 1, int(2 * (atomnr + N_y - 1) - 1)] = -all_k_xy[0][1]
                        h01[j, int(2 * (atomnr + N_y - 1) - 2)] = -all_k_xy[0][1]

                    elif atomnr > N_y:
                        h01[j, int(j - 1 - 2 * (atomnr - N_y + 1))] = -all_k_xy[0][1]
                        h01[j - 1, int(j - 1 - 2 * (atomnr - N_y + 1) + 1)] = -all_k_xy[0][1]

                    """if 0 <= int(j - 1 + 2 * (atomnr + N_y + 1) - 1) < h01.shape[1]:
                        h01[j - 1, int(j - 1 + 2 * (atomnr + N_y + 1) - 1)] = -all_k_xy[0][1]
                        h01[j, int(j - 1 + 2 * (atomnr + N_y + 1) - 2)] = -all_k_xy[0][1]
                    elif atomnr - N_y >= 0:
                        h01[j, int(j - 1 - 2 * (atomnr - N_y + 1))] = -all_k_xy[0][1]
                        h01[j - 1, int(j - 1 - 2 * (atomnr - N_y + 1) + 1)] = -all_k_xy[0][1]"""


                else:
                    
                    atomnr = np.ceil(float(j) / 2)
                    
                    # xy-coupling inner atom
                    if atomnr < N_y and interaction_range > 1:
                        ## first layer
                        # first atom
                        h01[j - 1, int(2 * (atomnr + N_y - 1)) - 1] = -all_k_xy[0][1]
                        h01[j, int(2 * (atomnr + N_y - 1)) - 2] = -all_k_xy[0][1]

                        #second atom
                        h01[j - 1, int(2 * (atomnr + N_y + 1)) - 1] = -all_k_xy[0][1]
                        h01[j, int(2 * (atomnr + N_y + 1)) - 2] = -all_k_xy[0][1]

                    elif atomnr > i * N_y and (i + 1) * N_y == N_y * interaction_range:
                        ## last layer
                        # first atom
                        h01[j - 1, int(2 * (atomnr - N_y - 1)) - 1] = -all_k_xy[0][1]
                        h01[j, int(2 * (atomnr - N_y - 1)) - 2] = -all_k_xy[0][1]

                        # second atom
                        h01[j - 1, int(2 * (atomnr - N_y + 1)) - 1] = -all_k_xy[0][1]
                        h01[j, int(2 * (atomnr - N_y + 1)) - 2] = -all_k_xy[0][1]
                    
                    elif atomnr > i * N_y and atomnr < (i + 1) * N_y and (i + 1) * N_y < N_y * interaction_range:
                        ## layer before
                        # first atom
                        h01[j - 1, int(2 * (atomnr - N_y - 1)) - 1] = -all_k_xy[0][1]
                        h01[j, int(2 * (atomnr - N_y - 1)) - 2] = -all_k_xy[0][1]
                        # second atom
                        h01[j - 1, int(2 * (atomnr - N_y + 1)) - 1] = -all_k_xy[0][1]
                        h01[j, int(2 * (atomnr - N_y + 1)) - 2] = -all_k_xy[0][1]

                        ## layer after
                        # first atom
                        h01[j - 1, int(2 * (atomnr + N_y - 1)) - 1] = -all_k_xy[0][1]
                        h01[j, int(2 * (atomnr + N_y - 1)) - 2] = -all_k_xy[0][1]
                        # second atom
                        h01[j - 1, int(2 * (atomnr + N_y + 1)) - 1] = -all_k_xy[0][1]
                        h01[j, int(2 * (atomnr + N_y + 1)) - 2] = -all_k_xy[0][1]

    print('\n', h01)