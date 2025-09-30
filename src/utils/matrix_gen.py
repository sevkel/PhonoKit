import numpy as np

def ranged_force_constant(
    k_el_x=0, k_c_x=0,
    k_el_y=0, k_c_y=0, 
    k_el_xy=0, k_c_xy=0,
    k_coupl_x=0, k_coupl_xy=0,
    lattice_constant=3.0, interaction_range=1, interact_potential="reciproke_squared"
) -> dict:
    """
    Calculate range dependent force-constants for the 2D lattice depending on the potential and interaction range.
    Only couplings !=0 are not taken into account.
    
    Args: 
        k_el_x (float): coupling in x within the electrode
        k_coupl_x (float): coupling in x from a electrode (L/R) to the center
        k_c_x (float): coupling in x withing the center part 

        k_el_y (float): coupling in y within the electrode
        k_c_y (float): coupling in y withing the center part 

        k_el_xy (float): coupling in xy within the electrode
        k_coupl_xy (float): coupling in xy from a electrode (L/R) to the center
        k_c_xy (float): coupling in xy within the center part 

        lattice_constant (float): lattice constant of the material
        interaction_range (int): interaction range in units of lattice constant
        interact_potential (str): interaction potential, currently only "reciproke_squared" implemented

    Returns:
        dict: containing ranged force-constants only for non-zero couplings
    """

    if interact_potential != "reciproke_squared":
        raise ValueError("Invalid interaction potential. Currently only 'reciproke_squared' is supported.")
    
    couplings = {
        "k_el_x": k_el_x,
        "k_c_x": k_c_x,
        "k_el_y": k_el_y,
        "k_c_y": k_c_y,
        "k_el_xy": k_el_xy,
        "k_c_xy": k_c_xy,
        "k_coupl_x": k_coupl_x,
        "k_coupl_xy": k_coupl_xy
    }
    
    results = {}
    for key, k_value in couplings.items():
        if k_value != 0:
            results[key] = [
                k_value * (1 / (i * lattice_constant)**2) 
                for i in range(1, interaction_range + 1)
            ]
        else:
            results[key] = [0.0] * interaction_range
    
    return results

def build_H_NN(N_y, interaction_range, k_values):
    """
    Build up an actual bulk layer of the electrode. The coupling includes options for x, y and xy coupling. 
    The coupling range is defined by the parameter interaction range.
    
    Args:
        N_y: Number of atoms in y-direction
        interaction_range: Interaction range
        k_values: Dictionary with force constants (from ranged_force_constant function)

    Returns: 
        np.array: Principal layer with force constants
    """
    
    all_k_x = k_values.get("k_el_x", [])
    all_k_y = k_values.get("k_el_y", []) 
    all_k_xy = k_values.get("k_el_xy", [])

    hNN = np.zeros((2 * N_y * interaction_range, 2 * N_y * interaction_range))

    #build Hessian matrix for the hNN principal bulklayer

    for i in range(interaction_range):

        for j in range(i * 2 * N_y, i * 2 * N_y + 2 * N_y):
            
            # diagonal elements x and xy coupling
            if j % 2 == 0:
                
                atomnr = np.ceil(float(j + 1) / 2)
                
                # ii-coupling
                hNN[j, j] = sum(2 * all_k_x[k] for k in range(len(all_k_x))) 


                for k in range(interaction_range):
                    if j + 2 * (k + 1) * N_y < hNN.shape[0]:                    
                        hNN[j, j + 2 * (k + 1) * N_y] = -all_k_x[k]
                    if j - 2 * (k + 1) * N_y >= 0:
                        hNN[j, j - 2 * (k + 1) * N_y] = -all_k_x[k]

                # ij-coupling in h01

                # xy-coupling
                if N_y > 1:
                    
                    if j == i * 2 * N_y or j == i * 2 * N_y + 2 * N_y - 2:
                        hNN[j, j] += 2 * all_k_xy[0]
                        hNN[j + 1, j + 1] += 2 * all_k_xy[0]

                    if j != 0 + i * 2 * N_y and j != i * 2 * N_y + 2 * N_y - 2 and N_y > 2:
                        hNN[j, j] += 4 * all_k_xy[0]
                        hNN[j + 1, j + 1] += 4 * all_k_xy[0]
                

            else:
                if N_y > 1:
                    # y coupling in the coupling range -> edge layers
                    if (j == i * 2 * N_y + 1) or (j == i * 2 * N_y + 2 * N_y - 1): 
                        
                        # xy-coupling
                        atomnr = np.ceil(float(j) / 2)

                        if interaction_range > 1:
                            
                            if atomnr < N_y:# and interaction_range > 1:
                                hNN[j - 1, int(j - 1 + 2 * (atomnr + N_y + 1) - 2)] = -all_k_xy[0]
                                hNN[j, int(j - 1 + 2 * (atomnr + N_y + 1) - 1)] = -all_k_xy[0]

                            elif atomnr ==  N_y and interaction_range > 1:
                                hNN[j - 1, int(2 * (atomnr + N_y - 1) - 2)] = -all_k_xy[0]
                                hNN[j, int(2 * (atomnr + N_y - 1) - 1)] = -all_k_xy[0]

                            
                            elif atomnr > N_y and interaction_range > 1 and (atomnr == (i + 1) * N_y or atomnr == i * N_y + 1) and atomnr <= N_y * interaction_range - N_y:
                                # N_y == 2 case
                                if atomnr == i * N_y + 1:
                                    hNN[j, 2 * int(i * N_y + 1 + N_y + 1) - 1] = -all_k_xy[0]
                                    hNN[j - 1, 2 * int(i * N_y + 1 + N_y + 1) - 2] = -all_k_xy[0]
                                    
                                    hNN[j, 2 * int(i * N_y + 1 - N_y + 1) - 1] = -all_k_xy[0]
                                    hNN[j - 1, 2 * int(i * N_y + 1 - N_y + 1) - 2] = -all_k_xy[0]
                                
                                elif atomnr == (i + 1) * N_y:
                                    hNN[j, 2 * int((i + 1) * N_y + N_y - 1) - 1] = -all_k_xy[0]
                                    hNN[j - 1, 2 * int((i + 1) * N_y + N_y - 1) - 2] = -all_k_xy[0]
                                    
                                    hNN[j, 2 * int((i + 1) * N_y - N_y - 1) - 1] = -all_k_xy[0]
                                    hNN[j - 1, 2 * int((i + 1) * N_y - N_y - 1) - 2] = -all_k_xy[0]
                                    
                            elif (atomnr == N_y * interaction_range - N_y + 1 or atomnr == N_y * interaction_range):
                                if atomnr == N_y * interaction_range - N_y + 1:
                                    hNN[j, 2 * int(N_y * interaction_range - N_y + 1 - N_y + 1) - 1] = -all_k_xy[0]
                                    hNN[j - 1, 2 * int(N_y * interaction_range - N_y + 1 - N_y + 1) - 2] = -all_k_xy[0]
                                elif atomnr == N_y * interaction_range:
                                    hNN[j, 2 * int(N_y * interaction_range - N_y - 1) - 1] = -all_k_xy[0]
                                    hNN[j - 1, 2 * int(N_y * interaction_range - N_y - 1) - 2] = -all_k_xy[0]


                        hNN[j, j] = all_k_y[0] + 2 * all_k_xy[0]

                        if j == 1 + i * 2 * N_y:
                            hNN[j, j + 2] = -all_k_y[0]
                        else:
                            hNN[j, j - 2] = -all_k_y[0]

                        if interaction_range >= N_y:
                            for k in range(1, N_y - 1):
                                hNN[j, j] += all_k_y[k]
                                
                                if j + 2 * (k + 1) < i * 2 * N_y + 2 * N_y:
                                    hNN[j, j + 2 * (k + 1)] = -all_k_y[k]
                                if j - 2 * (k + 1) >= i * 2 * N_y:
                                    hNN[j, j - 2 * (k + 1)] = -all_k_y[k]

                        else:
                            for k in range(1, interaction_range):
                                hNN[j, j] += all_k_y[k]
                            
                                if j + 2 * (k + 1) < i * 2 * N_y + 2 * N_y:
                                    hNN[j, j + 2 * (k + 1)] = -all_k_y[k]
                                if j - 2 * (k + 1) >= i * 2 * N_y:
                                    hNN[j, j - 2 * (k + 1)] = -all_k_y[k]


                    else:
                        
                        atomnr = np.ceil(float(j) / 2)
                        hNN[j, j] = 2 * all_k_y[0] + 4 * all_k_xy[0]
                        
                        # xy-coupling inner atom
                        if interaction_range > 1:
                            if atomnr < N_y:# and interaction_range > 1:
                                ## first layer
                                # first atom
                                hNN[j - 1, int(2 * (atomnr + N_y - 1)) - 2] = -all_k_xy[0]
                                hNN[j, int(2 * (atomnr + N_y - 1)) - 1] = -all_k_xy[0]

                                #second atom
                                hNN[j - 1, int(2 * (atomnr + N_y + 1)) - 2] = -all_k_xy[0]
                                hNN[j, int(2 * (atomnr + N_y + 1)) - 1] = -all_k_xy[0]

                            elif atomnr > i * N_y and (i + 1) * N_y == N_y * interaction_range:
                                ## last layer
                                # first atom
                                hNN[j - 1, int(2 * (atomnr - N_y - 1)) - 2] = -all_k_xy[0]
                                hNN[j, int(2 * (atomnr - N_y - 1)) - 1] = -all_k_xy[0]

                                # second atom
                                hNN[j - 1, int(2 * (atomnr - N_y + 1)) - 2] = -all_k_xy[0]
                                hNN[j, int(2 * (atomnr - N_y + 1)) - 1] = -all_k_xy[0]
                            
                            elif atomnr > i * N_y and atomnr < (i + 1) * N_y and (i + 1) * N_y < N_y * interaction_range:
                                ## layer before
                                # first atom
                                hNN[j - 1, int(2 * (atomnr - N_y - 1)) - 2] = -all_k_xy[0]
                                hNN[j, int(2 * (atomnr - N_y - 1)) - 1] = -all_k_xy[0]
                                # second atom
                                hNN[j - 1, int(2 * (atomnr - N_y + 1)) - 2] = -all_k_xy[0]
                                hNN[j, int(2 * (atomnr - N_y + 1)) - 1] = -all_k_xy[0]

                                ## layer after
                                # first atom
                                hNN[j - 1, int(2 * (atomnr + N_y - 1)) - 2] = -all_k_xy[0]
                                hNN[j, int(2 * (atomnr + N_y - 1)) - 1] = -all_k_xy[0]
                                # second atom
                                hNN[j - 1, int(2 * (atomnr + N_y + 1)) - 2] = -all_k_xy[0]
                                hNN[j, int(2 * (atomnr + N_y + 1)) - 1] = -all_k_xy[0]

                        if j + 2 < i * 2 * N_y + 2 * N_y:
                            hNN[j, j + 2] = -all_k_y[0]
                        if j - 2 >= 0 + i * 2 * N_y:
                            hNN[j, j - 2] = -all_k_y[0]

                        if interaction_range >= N_y:
                            for k in range(1, N_y - 1):
                                if atomnr - k - 1 > i * N_y and atomnr + k < i * N_y + N_y:
                                    hNN[j, j] += 2 * all_k_y[k]
                                    
                                elif (atomnr - k - 1 <= i * N_y and atomnr + k < i * N_y + N_y) or (atomnr - k - 1 > i * N_y and atomnr + k >= i * N_y + N_y):
                                    hNN[j, j] += all_k_y[k]
                                
                                if j + 2 * (k + 1) < i * 2 * N_y + 2 * N_y:
                                    hNN[j, j + 2 * (k + 1)] = -all_k_y[k]
                                if j - 2 * (k + 1) > i * 2 * N_y:
                                    hNN[j, j - 2 * (k + 1)] = -all_k_y[k]

                        else:
                            for k in range(1, interaction_range):
                                if atomnr - k - 1 > i * N_y and atomnr + k < N_y + i * N_y:
                                    hNN[j, j] += 2 * all_k_y[k]
                                elif (atomnr - k - 1 <= i * N_y and atomnr + k < i * N_y + N_y) or (atomnr - k - 1 > i * N_y and atomnr + k >= i * N_y + N_y):
                                    hNN[j, j] += all_k_y[k]
                                
                                if j + 2 * (k + 1) < i * 2 * N_y + 2 * N_y:
                                    hNN[j, j + 2 * (k + 1)] = -all_k_y[k]
                                if j - 2 * (k + 1) >= 0 + i * 2 * N_y:
                                    hNN[j, j - 2 * (k + 1)] = -all_k_y[k]

    return hNN

def build_H_00(N_y, interaction_range, k_values):
    """
    Build the hessian matrix for the first layer. The interaction range is taken into account.
    
    Returns:
        H_00 (np.ndarray): Hessian matrix of shape (2 * N_y, 2 * N_y)
    """

    all_k_x = k_values.get("k_el_x", [])
    all_k_y = k_values.get("k_el_y", []) 
    all_k_xy = k_values.get("k_el_xy", [])

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
                    h00[j, j] = sum(all_k_x[k] for k in range(len(all_k_x)))

                elif atomnr > i * N_y and (i + 1) * N_y == N_y * interaction_range:
                    ## last layer
                    for k in range(interaction_range):
                        if i - k > 0:
                            h00[j, j] += 2 * all_k_x[k]
                        else:
                            h00[j, j] += all_k_x[k]
                
                elif atomnr > i * N_y and atomnr <= (i + 1) * N_y and (i + 1) * N_y < N_y * interaction_range:
                    for k in range(interaction_range):
                        if i - k > 0:
                            h00[j, j] += 2 * all_k_x[k]
                        else:
                            h00[j, j] += all_k_x[k]
                            
                for k in range(interaction_range):
                    if j + 2 * (k + 1) * N_y < h00.shape[0]:                    
                        h00[j, j + 2 * (k + 1) * N_y] = -all_k_x[k]
                    if j - 2 * (k + 1) * N_y >= 0:
                        h00[j, j - 2 * (k + 1) * N_y] = -all_k_x[k]

                # xy-coupling # TODO: do something to take account that only for Ny > 1 possible or leave it to the user?
                if N_y > 1:
                    
                    if j == 0 or j == 2 * N_y - 2:
                        h00[j, j] += all_k_xy[0]
                        h00[j + 1, j + 1] += all_k_xy[0]

                    elif j < 2 * N_y - 2:
                        h00[j, j] += 2 * all_k_xy[0]
                        h00[j + 1, j + 1] += 2 * all_k_xy[0]

                    elif (j == i * 2 * N_y or j == i * 2 * N_y + 2 * N_y - 2) and (j != 0 and j != 2 * N_y - 2):
                        h00[j, j] += 2 * all_k_xy[0]
                        h00[j + 1, j + 1] += 2 * all_k_xy[0]

                    elif j != 0 + i * 2 * N_y and j != i * 2 * N_y + 2 * N_y - 2 and N_y > 2:
                        h00[j, j] += 4 * all_k_xy[0]
                        h00[j + 1, j + 1] += 4 * all_k_xy[0]
                

            else:
                
                if N_y > 1:
                    # y coupling in the coupling range -> edge layers/atoms
                    if (j == i * 2 * N_y + 1) or (j == i * 2 * N_y + 2 * N_y - 1): 
                        
                        # xy-coupling
                        atomnr = np.ceil(float(j) / 2)

                        if interaction_range > 1:
                            
                            if atomnr < N_y: #and interaction_range > 1:
                                h00[j - 1, int(j - 1 + 2 * (atomnr + N_y + 1) - 2)] = -all_k_xy[0]
                                h00[j, int(j - 1 + 2 * (atomnr + N_y + 1) - 1)] = -all_k_xy[0]

                            elif atomnr == N_y and interaction_range > 1:
                                h00[j - 1, int(2 * (atomnr + N_y - 1) - 2)] = -all_k_xy[0]
                                h00[j, int(2 * (atomnr + N_y - 1) - 1)] = -all_k_xy[0]

                            elif atomnr > N_y and interaction_range > 1 and (atomnr == (i + 1) * N_y or atomnr == i * N_y + 1) and atomnr <= N_y * interaction_range - N_y:
                                # N_y == 2 case
                                if atomnr == i * N_y + 1:
                                    h00[j, 2 * int(i * N_y + 1 + N_y + 1) - 1] = -all_k_xy[0]
                                    h00[j - 1, 2 * int(i * N_y + 1 + N_y + 1) - 2] = -all_k_xy[0]
                                    
                                    h00[j, 2 * int(i * N_y + 1 - N_y + 1) - 1] = -all_k_xy[0]
                                    h00[j - 1, 2 * int(i * N_y + 1 - N_y + 1) - 2] = -all_k_xy[0]
                                
                                elif atomnr == (i + 1) * N_y:
                                    h00[j, 2 * int((i + 1) * N_y + N_y - 1) - 1] = -all_k_xy[0]
                                    h00[j - 1, 2 * int((i + 1) * N_y + N_y - 1) - 2] = -all_k_xy[0]
                                    
                                    h00[j, 2 * int((i + 1) * N_y - N_y - 1) - 1] = -all_k_xy[0]
                                    h00[j - 1, 2 * int((i + 1) * N_y - N_y - 1) - 2] = -all_k_xy[0]
                                    
                            elif (atomnr == N_y * interaction_range - N_y + 1 or atomnr == N_y * interaction_range):
                                
                                if atomnr == N_y * interaction_range - N_y + 1:
                                    h00[j, 2 * int(N_y * interaction_range - N_y + 1 - N_y + 1) - 1] = -all_k_xy[0]
                                    h00[j - 1, 2 * int(N_y * interaction_range - N_y + 1 - N_y + 1) - 2] = -all_k_xy[0]
                                elif atomnr == N_y * interaction_range:
                                    h00[j, 2 * int(N_y * interaction_range - N_y - 1) - 1] = -all_k_xy[0]
                                    h00[j - 1, 2 * int(N_y * interaction_range - N_y - 1) - 2] = -all_k_xy[0]



                        #y - coupling
                        if i == 0:
                            h00[j, j] = all_k_y[0] + all_k_xy[0]
                        else:
                            h00[j, j] = all_k_y[0] + 2 * all_k_xy[0]

                        if j == 1 + i * 2 * N_y:
                            h00[j, j + 2] = -all_k_y[0]
                        else:
                            h00[j, j - 2] = -all_k_y[0]

                        if interaction_range >= N_y:
                            for k in range(1, N_y - 1):
                                h00[j, j] += all_k_y[k]
                                
                                if j + 2 * (k + 1) < i * 2 * N_y + 2 * N_y:
                                    h00[j, j + 2 * (k + 1)] = -all_k_y[k]
                                if j - 2 * (k + 1) >= i * 2 * N_y:
                                    h00[j, j - 2 * (k + 1)] = -all_k_y[k]

                        else:
                            for k in range(1, interaction_range):
                                h00[j, j] += all_k_y[k]
                            
                                if j + 2 * (k + 1) < i * 2 * N_y + 2 * N_y:
                                    h00[j, j + 2 * (k + 1)] = -all_k_y[k]
                                if j - 2 * (k + 1) >= 0 + i * 2 * N_y:
                                    h00[j, j - 2 * (k + 1)] = -all_k_y[k]


                    else:
                        
                        atomnr = np.ceil(float(j) / 2)

                        if i == 0:
                            h00[j, j] = 2 * all_k_y[0] + 2 * all_k_xy[0]
                        else:
                            h00[j, j] = 2 * all_k_y[0] + 4 * all_k_xy[0]
                        
                        # xy-coupling inner atom, inner layers
                        if interaction_range > 1:
                            
                            if atomnr < N_y: #and interaction_range > 1:
                                ## first layer
                                # first atom
                                h00[j - 1, int(2 * (atomnr + N_y - 1)) - 2] = -all_k_xy[0]
                                h00[j, int(2 * (atomnr + N_y - 1)) - 1] = -all_k_xy[0]

                                #second atom
                                h00[j - 1, int(2 * (atomnr + N_y + 1)) - 2] = -all_k_xy[0]
                                h00[j, int(2 * (atomnr + N_y + 1)) - 1] = -all_k_xy[0]

                            elif atomnr > i * N_y and (i + 1) * N_y == N_y * interaction_range:
                                ## last layer
                                # first atom
                                h00[j - 1, int(2 * (atomnr - N_y - 1)) - 2] = -all_k_xy[0]
                                h00[j, int(2 * (atomnr - N_y - 1)) - 1] = -all_k_xy[0]

                                # second atom
                                h00[j - 1, int(2 * (atomnr - N_y + 1)) - 2] = -all_k_xy[0]
                                h00[j, int(2 * (atomnr - N_y + 1)) - 1] = -all_k_xy[0]
                            
                            elif atomnr > i * N_y and atomnr < (i + 1) * N_y and (i + 1) * N_y < N_y * interaction_range:
                                ## layer before
                                # first atom
                                h00[j - 1, int(2 * (atomnr - N_y - 1)) - 2] = -all_k_xy[0]
                                h00[j, int(2 * (atomnr - N_y - 1)) - 1] = -all_k_xy[0]
                                # second atom
                                h00[j - 1, int(2 * (atomnr - N_y + 1)) - 2] = -all_k_xy[0]
                                h00[j, int(2 * (atomnr - N_y + 1)) - 1] = -all_k_xy[0]

                                ## layer after
                                # first atom
                                h00[j - 1, int(2 * (atomnr + N_y - 1)) - 2] = -all_k_xy[0]
                                h00[j, int(2 * (atomnr + N_y - 1)) - 1] = -all_k_xy[0]
                                # second atom
                                h00[j - 1, int(2 * (atomnr + N_y + 1)) - 2] = -all_k_xy[0]
                                h00[j, int(2 * (atomnr + N_y + 1)) - 1] = -all_k_xy[0]

                            

                        if j + 2 < i * 2 * N_y + 2 * N_y:
                            h00[j, j + 2] = -all_k_y[0]
                        if j - 2 >= 0 + i * 2 * N_y:
                            h00[j, j - 2] = -all_k_y[0]


                        if interaction_range >= N_y:
                            for k in range(1, N_y - 1):
                                if atomnr - k - 1 > i * N_y and atomnr + k < i * N_y + N_y:
                                    h00[j, j] += 2 * all_k_y[k]
                                    
                                elif (atomnr - k - 1 <= i * N_y and atomnr + k < i * N_y + N_y) or (atomnr - k - 1 > i * N_y and atomnr + k >= i * N_y + N_y):
                                    h00[j, j] += all_k_y[k]
                                
                                if j + 2 * (k + 1) < i * 2 * N_y + 2 * N_y:
                                    h00[j, j + 2 * (k + 1)] = -all_k_y[k]
                                if j - 2 * (k + 1) > i * 2 * N_y:
                                    h00[j, j - 2 * (k + 1)] = -all_k_y[k]

                        else:
                            for k in range(1, interaction_range):
                                if atomnr - k - 1 > i * N_y and atomnr + k < N_y + i * N_y:#N_y * interaction_range:
                                    h00[j, j] += 2 * all_k_y[k]
                                elif (atomnr - k - 1 <= i * N_y and atomnr + k < i * N_y + N_y) or (atomnr - k - 1 > i * N_y and atomnr + k >= i * N_y + N_y):
                                    h00[j, j] += all_k_y[k]
                                
                                if j + 2 * (k + 1) < i * 2 * N_y + 2 * N_y:
                                    h00[j, j + 2 * (k + 1)] = -all_k_y[k]
                                if j - 2 * (k + 1) >= 0 + i * 2 * N_y:
                                    h00[j, j - 2 * (k + 1)] = -all_k_y[k]

    return h00

def build_H_01(N_y, interaction_range, k_values):
    """
    Build the hessian matrix for the interaction between two princial layers. The interaction range is taken into account.
    """

    all_k_x = k_values.get("k_el_x", [])
    all_k_y = k_values.get("k_el_y", []) 
    all_k_xy = k_values.get("k_el_xy", [])

    h01 = np.zeros((2 * N_y * interaction_range, 2 * N_y * interaction_range))
    # build Hessian matrix for the h01 interaction between principal layers
    # rows are A layer atoms, columns are B layer atoms
    
    for i in range(h01.shape[0]):
        for j in range(h01.shape[1]):
            if i % 2 == 0 and j % 2 == 0:
                atomnr_lay1 = np.ceil(float(i + 1) / 2)
                atomnr_lay2 = np.ceil(float(j + 1) / 2)
                
                if atomnr_lay1 == atomnr_lay2:
                    h01[i, j] = -all_k_x[-1]
                
                if interaction_range > 1:
                    # for more than next nearest neighbour coupling in x-direction
                    for r in range(interaction_range - 1, -1, -1):
                        
                        if atomnr_lay2 == atomnr_lay1 - r * N_y:
                            h01[i, j] = -all_k_x[-(r + 1)]
                    
                    # xy coupling
                    if (interaction_range - 1) * N_y < atomnr_lay1 <= interaction_range * N_y and atomnr_lay2 <= N_y:
                        # edge atoms
                        if atomnr_lay1 == (interaction_range - 1) * N_y + 1 and atomnr_lay2 == 2:
                            h01[i, j] = -all_k_xy[0]
                            h01[i + 1, j + 1] = -all_k_xy[0]
                        
                        elif atomnr_lay1 == interaction_range * N_y and atomnr_lay2 == N_y - 1:
                            h01[i, j] = -all_k_xy[0]
                            h01[i + 1, j + 1] = -all_k_xy[0]
                        
                        # middle atoms
                        elif atomnr_lay2 == atomnr_lay1 - (interaction_range - 1) * N_y + 1 or atomnr_lay2 == atomnr_lay1 - (interaction_range - 1) * N_y - 1:
                            h01[i, j] = -all_k_xy[0]
                            h01[i + 1, j + 1] = -all_k_xy[0]
                    
                else:
                    # edge atoms
                    if atomnr_lay1 == 1 and atomnr_lay2 == 2:
                        h01[i, j] = -all_k_xy[0]
                        h01[i + 1, j + 1] = -all_k_xy[0]
                    elif atomnr_lay1 == N_y and atomnr_lay2 == N_y - 1:
                        h01[i, j] = -all_k_xy[0]
                        h01[i + 1, j + 1] = -all_k_xy[0]
                    elif atomnr_lay2 == atomnr_lay1 - 1 or atomnr_lay2 == atomnr_lay1 + 1:
                        h01[i, j] = -all_k_xy[0]
                        h01[i + 1, j + 1] = -all_k_xy[0]
                    
                    # make matrix now symmetric
                    h01[j, i] = h01[i, j]        
                                                

    return h01