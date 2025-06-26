__docformat__ = "google"

import numpy as np
from utils import constants
import tmoutproc as top

class Model:
    """
    Mother class for all model systems.
    """
    
    def __init__(self, k_c, interact_potential="reciproke_squared", interaction_range=1, lattice_constant=3.0, atom_type="Au"):
        self.k_c = k_c * 1#(constants.eV2hartree / constants.ang2bohr ** 2)
        self.interact_potential = interact_potential
        self.interaction_range = interaction_range
        self.lattice_constant = lattice_constant
        self.atom_type = atom_type

    def ranged_force_constant(self):
        """
        Calculate ranged force constants for the 2D lattice dependend on which potential is used and on how many neighbors are coupled.
        
        Retruns:
            range_force_constant (list of tuples): Ranged force constant for the 2D lattice
        """

        match self.interact_potential:
            
            case "reciproke_squared":
                
                all_k_x = list(enumerate((self.k_x * (1 / (i * self.lattice_constant)**2) for i in range(1, self.interaction_range + 1))))
                all_k_c = list(enumerate((self.k_c * (1 / (i * self.lattice_constant)**2) for i in range(1, self.interaction_range + 1))))
                
                try:
                    all_k_y = list(enumerate((self.k_y * (1 / (i * self.lattice_constant)**2) for i in range(1, self.interaction_range + 1))))
                    all_k_xy =  list(enumerate((self.k_xy * (1 / (i * self.lattice_constant)**2) for i in range(1, self.interaction_range + 1))))
                    all_k_c_xy = list(enumerate((self.k_c_xy * (1 / (i * self.lattice_constant)**2) for i in range(1, self.interaction_range + 1))))
                    
                except AttributeError:
                    return all_k_x, all_k_c


        return all_k_x, all_k_y, all_k_xy, all_k_c, all_k_c_xy

class Chain1D(Model):
    """
    This class creates a 1D chain with a given number of atoms (N) and a given spring constant (k). Inherits from the Model class.
    """

    def __init__(self, k_c, interact_potential, interaction_range, lattice_constant, atom_type, k_x, N): 
        super().__init__(k_c, interact_potential, interaction_range, lattice_constant, atom_type)
        self.k_x = k_x * 1#(constants.eV2hartree / constants.ang2bohr ** 2)
        self.N = N
        
        self.hessian = self.build_hessian()

    def build_hessian(self):
        """Build the hessian matrix for a 1D chain.

        Returns:
            hessian (np.ndarray): Hessian matrix of shape (N, N)
        """

        assert self.interaction_range < self.N, "Interaction range must be smaller than the number of atoms in the chain!"

        hessian = np.zeros((self.N, self.N), dtype=float)
        all_k_x, all_k_c = self.ranged_force_constant()

        for i in range(self.N):
            
            atomnr = i + 1

            # take care of interaction range
            for j in range(self.interaction_range):
                
                if i + j + 1 < self.N:
                    hessian[i, i + j + 1] = -all_k_x[j][1]
                if i - j - 1 >= 0:
                    hessian[i, i - j - 1] = -all_k_x[j][1]  
            
            hessian[i, i] = -np.sum(hessian[i, :]) #+ sum(all_k_c[k][1] for k in range(self.interaction_range) if atomnr - self.interaction_range <= 0)
            
            # left side
            if atomnr - self.interaction_range < 0:
                hessian[i, i] += sum(all_k_c[k][1] for k in range(self.interaction_range) if atomnr - (k + 1) <= 0)
            
            # middle atom within interaction range
            elif atomnr - self.interaction_range == 0 and atomnr + self.interaction_range >= self.N:
                
                # must be case sensitive for interaction range == 1
                if self.interaction_range == 1:
                    hessian[i, i] += all_k_c[0][1]
                else:
                    hessian[i, i] += 2 * all_k_c[-1][1]
            
            # right side
            elif atomnr + self.interaction_range > self.N:
                hessian[i, i] += sum(all_k_c[k][1] for k in range(self.interaction_range) if atomnr + (k + 1) > self.N)

        #assert np.sum(hessian) == 0, "Acoustic sum rule fullfilled! Check the initialization of the hessian"
        return hessian
     
    def create_fake_coord_file(self, output_file="", xyz=True):
        """
        Creates fake coord file in xyz format as default.
        Args:
            output_file (String): Outputfile. If string is empty no file will be written
            xyz (bool): Create xyz file (True) or turbomole format (False)

        Returns:
            coord_xyz
        """
        
        coord_xyz = list()
        
        
        for j in range(0, self.N):

            tmp = np.zeros(4, dtype=object)
            # atom_type
            tmp[0] = self.atom_type
            #x
            tmp[1] = j * self.lattice_constant
            #y
            tmp[2] = 0
            #z
            tmp[3] = 0

            coord_xyz.append(tmp)

        coord_xyz = np.asarray(coord_xyz)

        if xyz == True:

            if output_file != "":
                top.write_xyz_file(output_file, coord_xyz, "", suppress_sci_not=False)

            return coord_xyz
        
        else:

            coord_turbomole = top.x2t(coord_xyz)
            if output_file != "":
                top.write_coord_file(output_file, coord_turbomole)

            return coord_turbomole

class FiniteLattice2D(Model):
    """
    This class creates a 2D finite lattice with a given number of layers (N_y) and a given number of atoms in each layer (N_x). Inherits from the Model class.
    """

    def __init__(self, N_y, N_x, k_c, k_c_xy, k_x, k_y, k_xy, interact_potential="reciproke_squared", interaction_range=1, lattice_constant=3.0, atom_type="Au"):
        super().__init__(k_c, interact_potential, interaction_range, lattice_constant, atom_type)
        self.N_y = N_y
        self.N_x = N_x
        self.k_c = k_c * 1#(constants.eV2hartree / constants.ang2bohr ** 2)
        self.k_c_xy = k_c_xy * 1#(constants.eV2hartree / constants.ang2bohr ** 2)
        self.k_x = k_x * 1#(constants.eV2hartree / constants.ang2bohr ** 2) 
        self.k_y = k_y * 1#(constants.eV2hartree / constants.ang2bohr ** 2) 
        self.k_xy = k_xy * 1#(constants.eV2hartree / constants.ang2bohr ** 2)
        self.atom_type = atom_type
        self.hessian = self.build_hessian()


    def build_hessian(self):
        """
        Build the hessinan matrix for a 2D lattice including variable neighbor coupling.

        Returns:
            hessian (np.ndarray): Hessian matrix of shape (2 * N_y * N_x, 2 * N_y * N_x)
        """
        
        assert (self.interaction_range < self.N_y or self.interaction_range < self.N_x), "Interaction range must be smaller than the number of atoms in x- and y-direction!"
        assert (self.N_y > 1 and self.N_x > 1), "Number of atoms in x- and y-direction must be greater than 1 otherwise take Chain1D model!"

        def build_bulk_layers():
            """
            Building bulk submatrices until the layer where the full interaction range is reached. Returns combination of layer index from apart from the surface and its corresponding matrix.
            
            Returns:
                List of tuples: Contains combination of layer index from apart from the surface and its corresponding matrix as np.ndarray.
            """
            #TODO: check for bulk layer adjustments for coupling via jan. Tho, hNN should be fine as interaction layers cover the coupling

            bulk_hessians = list()
            hNNtemplate = np.zeros((2 * self.N_y, 2 * self.N_y), dtype=float)

            all_k_x, all_k_y, all_k_xy, all_k_c, all_k_c_xy = self.ranged_force_constant()

            
            for i in range(1, self.interaction_range + 1):

                hNN = hNNtemplate.copy()

                for j in range(hNN.shape[0]):

                    # diagonal elements x coupling
                    if j % 2 == 0:
                        
                        hNN[j, j] = sum(2 * all_k_x[k][1] for k in range(i)) 
                        
                        if i <= (self.N_x - 2) // 2:
                            hNN[j, j] += sum(all_k_x[k][1] for k in range(i, self.interaction_range) if k <= (self.N_x - 2) // 2)

                        # xy-coupling
                        if j == 0 or j == hNN.shape[0] - 2:
                            hNN[j, j + 1] = 2 * all_k_xy[0][1] 
                            hNN[j + 1, j] = 2 * all_k_xy[0][1]

                        if j != 0 and j != hNN.shape[0] - 2 and self.N_y > 2:
                            hNN[j, j + 1] = 4 * all_k_xy[0][1]
                            hNN[j + 1, j] = 4 * all_k_xy[0][1]
                        

                    else:
                        # y coupling in the coupling range
                        if j == 1 or j == hNN.shape[0] - 1:
                            hNN[j, j] = all_k_y[0][1]

                            if j == 1:
                                hNN[j, j + 2] = -all_k_y[0][1]
                            else:
                                hNN[j, j - 2] = -all_k_y[0][1]

                            if self.interaction_range >= self.N_y:
                                for k in range(1, self.N_y - 1):
                                    hNN[j, j] += all_k_y[k][1]
                                    
                                    if j + 2 * (k + 1) < hNN.shape[0]:
                                        hNN[j, j + 2 * (k + 1)] = -all_k_y[k][1]
                                    if j - 2 * (k + 1) >= 0:
                                        hNN[j, j - 2 * (k + 1)] = -all_k_y[k][1]

                            else:
                                for k in range(1, self.interaction_range):
                                    hNN[j, j] += all_k_y[k][1]
                                
                                    if j + 2 * (k + 1) < hNN.shape[0]:
                                        hNN[j, j + 2 * (k + 1)] = -all_k_y[k][1]
                                    if j - 2 * (k + 1) >= 0:
                                        hNN[j, j - 2 * (k + 1)] = -all_k_y[k][1]
                            
                        else:
                            hNN[j, j] = 2 * all_k_y[0][1]
                            
                            if j + 2 < hNN.shape[0]:
                                hNN[j, j + 2] = -all_k_y[0][1]
                            if j - 2 >= 0:
                                hNN[j, j - 2] = -all_k_y[0][1]

                            atomnr = np.ceil(float(j) / 2)

                            if self.interaction_range >= self.N_y:
                                for k in range(1, self.N_y - 1):
                                    if atomnr - k - 1 > 0 and atomnr + k < self.N_y:
                                        hNN[j, j] += 2 * all_k_y[k][1]
                                        
                                    elif (atomnr - k - 1 <= 0 and atomnr + k < self.N_y) or (atomnr - k - 1 > 0 and atomnr + k >= self.N_y):
                                        hNN[j, j] += all_k_y[k][1]
                                    
                                    if j + 2 * (k + 1) < hNN.shape[0]:
                                        hNN[j, j + 2 * (k + 1)] = -all_k_y[k][1]
                                    if j - 2 * (k + 1) >= 0:
                                        hNN[j, j - 2 * (k + 1)] = -all_k_y[k][1]

                            else:
                                for k in range(1, self.interaction_range):
                                    if atomnr - k - 1 > 0 and atomnr + k < self.N_y:
                                        hNN[j, j] += 2 * all_k_y[k][1]
                                    elif (atomnr - k - 1 <= 0 and atomnr + k < self.N_y) or (atomnr - k - 1 > 0 and atomnr + k >= self.N_y):
                                        hNN[j, j] += all_k_y[k][1]
                                    
                                    if j + 2 * (k + 1) < hNN.shape[0]:
                                        hNN[j, j + 2 * (k + 1)] = -all_k_y[k][1]
                                    if j - 2 * (k + 1) >= 0:
                                        hNN[j, j - 2 * (k + 1)] = -all_k_y[k][1]

                bulk_hessians.append((i, hNN))
                            
            return bulk_hessians
        
        def build_layer_interactions():
            """
            Builds interaction matrices for the layers in the x-direction. The interaction range is taken into account.

            Returns:
                List of tuples: Contains combination of layer index and its corresponding interaction matrix as np.ndarray.
            """
            #TODO: adjust interaction layers for coupling

            interact_layer_list = list()
            h_interact_template = np.zeros((2 * self.N_y, 2 * self.N_y), dtype=float)
            all_k_x, all_k_y, all_k_xy, all_k_c, all_k_c_xy = self.ranged_force_constant()

            for i in range(self.interaction_range):
                h_interact = h_interact_template.copy()
            
                for j in range(h_interact.shape[0]):

                    # diagonal elements x coupling
                    if j % 2 == 0:
                        h_interact[j, j] = -all_k_x[i][1]

                        if i == 0:
                            # xy-coupling
                            if j == 0:
                                h_interact[j, j + 3] = -all_k_xy[0][1]
                                h_interact[j + 3, j] = -all_k_xy[0][1]

                            elif j == h_interact.shape[0] - 2:
                                h_interact[j, j - 1] = -all_k_xy[0][1]
                                h_interact[j - 1, j] = -all_k_xy[0][1]

                            else:
                                h_interact[j, j + 3] = -all_k_xy[0][1]
                                h_interact[j + 3, j] = -all_k_xy[0][1]
                                h_interact[j, j - 1] = -all_k_xy[0][1]
                                h_interact[j - 1, j] = -all_k_xy[0][1]


                interact_layer_list.append((i, h_interact))

            return interact_layer_list
        
        def build_H_00():
            """
            Build the hessian matrix for the first layer. The interaction range is taken into account.
            
            Returns:
                H_00 (np.ndarray): Hessian matrix of shape (2 * N_y, 2 * N_y)
            """
            
            H_00 = np.zeros((2 * self.N_y, 2 * self.N_y), dtype=float)
            all_k_x, all_k_y, all_k_xy, all_k_c, all_k_c_xy = self.ranged_force_constant()

            for i in range(H_00.shape[0]):

                if i % 2 == 0:
                    # x coupling in the coupling range
                    H_00[i, i] = sum(all_k_x[k][1] for k in range(self.interaction_range) if (self.N_x > self.interaction_range and k < self.N_x - self.interaction_range)
                                    or (self.N_x <= self.interaction_range and k < self.N_x // 2))
                    
                    if i == 0 or i == H_00.shape[0] - 2:
                        # xy coupling
                        H_00[i, i + 1] = all_k_xy[0][1]
                        H_00[i + 1, i] = all_k_xy[0][1]

                        #TODO: adjust H00 for interaction

                    else:
                        H_00[i, i + 1] = 2 * all_k_xy[0][1]
                        H_00[i + 1, i] = 2 * all_k_xy[0][1]

                    
                else:
                    # y coupling in the coupling range, first and last k_y, rest 2k_y
                    if i == 1 or i == H_00.shape[0] - 1:

                        H_00[i, i] = all_k_y[0][1]
                        if i == 1:
                            H_00[i, i + 2] = -all_k_y[0][1]
                        else:
                            H_00[i, i - 2] = -all_k_y[0][1]

                        if self.interaction_range >= self.N_y:
                            for k in range(1, self.N_y - 1):
                                H_00[i, i] += all_k_y[k][1]
                                
                                if i + 2 * (k + 1) < H_00.shape[0]:
                                    H_00[i, i + 2 * (k + 1)] = -all_k_y[k][1]
                                if i - 2 * (k + 1) >= 0:
                                    H_00[i, i - 2 * (k + 1)] = -all_k_y[k][1]

                        else:
                            for k in range(1, self.interaction_range):
                                H_00[i, i] += all_k_y[k][1]
                            
                                if i + 2 * (k + 1) < H_00.shape[0]:
                                    H_00[i, i + 2 * (k + 1)] = -all_k_y[k][1]
                                if i - 2 * (k + 1) >= 0:
                                    H_00[i, i - 2 * (k + 1)] = -all_k_y[k][1]
                        
                    else:
                        H_00[i, i] = 2 * all_k_y[0][1]
                        
                        if i + 2 < H_00.shape[0]:
                            H_00[i, i + 2] = -all_k_y[0][1]
                        if i - 2 >= 0:
                            H_00[i, i - 2] = -all_k_y[0][1]

                        atomnr = np.ceil(float(i) / 2)

                        if self.interaction_range >= self.N_y:
                            for k in range(1, self.N_y - 1):
                                if atomnr - k - 1 > 0 and atomnr + k < self.N_y:
                                    H_00[i, i] += 2 * all_k_y[k][1]
                                    
                                elif (atomnr - k - 1 <= 0 and atomnr + k < self.N_y) or (atomnr - k - 1 > 0 and atomnr + k >= self.N_y):
                                    H_00[i, i] += all_k_y[k][1]
                                
                                if i + 2 * (k + 1) < H_00.shape[0]:
                                    H_00[i, i + 2 * (k + 1)] = -all_k_y[k][1]
                                if i - 2 * (k + 1) >= 0:
                                    H_00[i, i - 2 * (k + 1)] = -all_k_y[k][1]

                        else:
                            for k in range(1, self.interaction_range):
                                if atomnr - k - 1 > 0 and atomnr + k < self.N_y:
                                    H_00[i, i] += 2 * all_k_y[k][1]
                                elif (atomnr - k - 1 <= 0 and atomnr + k < self.N_y) or (atomnr - k - 1 > 0 and atomnr + k >= self.N_y):
                                    H_00[i, i] += all_k_y[k][1]
                                
                                if i + 2 * (k + 1) < H_00.shape[0]:
                                    H_00[i, i + 2 * (k + 1)] = -all_k_y[k][1]
                                if i - 2 * (k + 1) >= 0:
                                    H_00[i, i - 2 * (k + 1)] = -all_k_y[k][1]
            
            return H_00


        bulk_layers = build_bulk_layers()
        layer_interactions = build_layer_interactions()
        H_00 = build_H_00()
    
        hessian = np.zeros((2 * self.N_y * self.N_x, 2 * self.N_y * self.N_x), dtype=float)

        for i in range(self.N_x):

            #surface layers + interaction
            if (i == 0 or i == self.N_x - 1) and self.N_x > 1:
                hessian[i * H_00.shape[0]: i * H_00.shape[0] + H_00.shape[0], # rows
                        i * H_00.shape[0]: i * H_00.shape[0] + H_00.shape[0]] = H_00
                
                #interaction with the layers within the interaction range
                if i == 0:
                    for j in range(len(layer_interactions)):
                        if i + j + 1 < self.N_x:
                            hessian[i * H_00.shape[0]: i * H_00.shape[0] + H_00.shape[0], 
                                    (i + j + 1) * H_00.shape[0]: (i + j + 1) * H_00.shape[0] + H_00.shape[0]] = layer_interactions[j][1]
                else:
                    for j in range(len(layer_interactions)):
                        if i - j - 1 >= 0:
                            hessian[i * H_00.shape[0]: i * H_00.shape[0] + H_00.shape[0], 
                                    (i - j - 1) * H_00.shape[0]: (i - j - 1) * H_00.shape[0] + H_00.shape[0]] = layer_interactions[j][1]
                        


            #bulk layers + interaction
            elif i < len(bulk_layers) and i < self.N_x // 2:
                hessian[i * H_00.shape[0]: i * H_00.shape[0] + H_00.shape[0], 
                        i * H_00.shape[0]: i * H_00.shape[0] + H_00.shape[0]] = bulk_layers[i - 1][1]
                
                #interaction with the layers within the interaction range, depending on how many layers are to the left and right
                for j in range(len(layer_interactions)):
                    if i - j - 1 >= 0:
                        hessian[i * H_00.shape[0]: i * H_00.shape[0] + H_00.shape[0], 
                                (i - j - 1) * H_00.shape[0]: (i - j - 1) * H_00.shape[0] + H_00.shape[0]] = layer_interactions[j][1]
                    if i + j + 1 < self.N_x:
                        hessian[i * H_00.shape[0]: i * H_00.shape[0] + H_00.shape[0], 
                                (i + j + 1) * H_00.shape[0]: (i + j + 1) * H_00.shape[0] + H_00.shape[0]] = layer_interactions[j][1]

            elif i >= self.N_x // 2 and i >= self.N_x - len(bulk_layers):
                hessian[i * H_00.shape[0]: i * H_00.shape[0] + H_00.shape[0], 
                        i * H_00.shape[0]: i * H_00.shape[0] + H_00.shape[0]] = bulk_layers[self.N_x - i - 2][1]

                #interaction with the layers within the interaction range, depending on how many layers are to the left and right
                for j in range(len(layer_interactions)):
                    if i - j - 1 >= 0:
                        hessian[i * H_00.shape[0]: i * H_00.shape[0] + H_00.shape[0], 
                                (i - j - 1) * H_00.shape[0]: (i - j - 1) * H_00.shape[0] + H_00.shape[0]] = layer_interactions[j][1]
                    if i + j + 1 < self.N_x:
                        hessian[i * H_00.shape[0]: i * H_00.shape[0] + H_00.shape[0], 
                                (i + j + 1) * H_00.shape[0]: (i + j + 1) * H_00.shape[0] + H_00.shape[0]] = layer_interactions[j][1]

            else:
                hessian[i * H_00.shape[0]: i * H_00.shape[0] + H_00.shape[0], 
                        i * H_00.shape[0]: i * H_00.shape[0] + H_00.shape[0]] = bulk_layers[-1][1]

                #interaction with the layers within the interaction range
                for j in range(len(layer_interactions)):
                    if i - j - 1 >= 0:
                        hessian[i * H_00.shape[0]: i * H_00.shape[0] + H_00.shape[0], 
                                (i - j - 1) * H_00.shape[0]: (i - j - 1) * H_00.shape[0] + H_00.shape[0]] = layer_interactions[j][1]
                    if i + j + 1 < self.N_x:
                        hessian[i * H_00.shape[0]: i * H_00.shape[0] + H_00.shape[0], 
                                (i + j + 1) * H_00.shape[0]: (i + j + 1) * H_00.shape[0] + H_00.shape[0]] = layer_interactions[j][1]
                


        #Check if acoustic sum rule is fulfilled or not
        assert np.abs(np.sum(hessian)) < 1E-12, "Acoustic sum rule fullfilled! Check the initialization of the hessian"
        return hessian

    def create_fake_coord_file(self, output_file="", xyz=True):
        """
        Creates fake coord file in xyz format as default.
        Args:
            output_file (String): Outputfile. If the string is empty no file will be written.
            xyz (bool): Create xyz file (True) or turbomole format (False)

        Returns:
            coord_xyz
        """

        coord_xyz = list()
        
        for i in range(0, self.N_y):
            for j in range(0,self.N_x):

                tmp = np.zeros(4, dtype=object)
                
                tmp[0] = self.atom_type
                #x
                tmp[1] = j * self.lattice_constant
                #y
                tmp[2] = i * self.lattice_constant
                #z
                tmp[3] = 0

                coord_xyz.append(tmp)

        coord_xyz = np.asarray(coord_xyz)

        if xyz == True:

            if output_file != "":
                top.write_xyz_file(output_file, coord_xyz, "", suppress_sci_not=False)

            return coord_xyz
        
        else:

            coord_turbomole = top.x2t(coord_xyz)
            if output_file != "":
                top.write_coord_file(output_file, coord_turbomole)

            return coord_turbomole


if __name__ == '__main__':

    #TODO: Doesn't work for interaction_range > N_x // 2 --> Fix this
    junction2D = FiniteLattice2D(N_y=3, N_x=2, k_x=180, k_c=900, k_c_xy=0, k_y=180, k_xy=0, interaction_range=1, atom_type="Au")
    junction1D = Chain1D(interact_potential="reciproke_squared", interaction_range=1, lattice_constant=3.0, atom_type="Au", k_x=180, k_c=900, N=2)
    print('debugging')






