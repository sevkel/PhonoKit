"""
Model Systems for Phononic Transport

This module provides model systems for phonon transport calculations:
- 1D chains with various configurations
- 2D ribbons with different geometries
- Dynamical matrix construction
- Force constant management

All models support configurable:
- Coupling constants between electrodes and central region
- Interaction potentials and ranges
- Lattice parameters and atom types
- Matrix generation with sparse storage

Note: Force constants should be multiplied by (constants.eV2hartree / constants.ang2bohr ** 2) if needed

Author: Severin Keller
Date: 2025
"""

import numpy as np

# Local imports
from utils import matrix_gen as mg


# ============================================================================
# MODEL SYSTEM CLASSES
# ============================================================================

class Model:
    """
    Base Model Class for Phononic Systems
    
    Abstract base class for constructing various model systems used in phonon transport
    calculations. Provides common parameters and functionality for all derived models.

    Args:
        k_coupl_x_l (float): Left electrode-central region coupling constant
        k_c_x (float): Intra-central region coupling constant
        k_coupl_x_r (float): Right electrode-central region coupling constant
        interact_potential (str): Interaction potential type (default: "reciproke_squared")
        interaction_range (int): Interaction range in nearest neighbors (default: 1)
        lattice_constant (float): Lattice constant in appropriate units (default: 3.0)
        atom_type (str): Atom type identifier for the central region (default: "Au")

    Attributes:
        k_coupl_x_l (float): Left coupling constant
        k_c_x (float): Central coupling constant  
        k_coupl_x_r (float): Right coupling constant
        interact_potential (str): Interaction potential specification
        interaction_range (int): Range of interactions
        lattice_constant (float): System lattice constant
        atom_type (str): Central region atom type
    """
    
    def __init__(self, k_coupl_x_l, k_c_x, k_coupl_x_r, interact_potential="reciproke_squared", interaction_range=1, lattice_constant=3.0, atom_type="Au"):
        self.k_c_x = k_c_x 
        self.k_coupl_x_l = k_coupl_x_l 
        self.k_coupl_x_r = k_coupl_x_r
        self.interact_potential = interact_potential
        self.interaction_range = interaction_range
        self.lattice_constant = lattice_constant
        self.atom_type = atom_type

class Chain1D(Model):
    """
    One-Dimensional Chain Model
    
    Creates a 1D atomic chain with configurable length and coupling constants.
    Inherits coupling and interaction parameters from the base Model class.

    Args:
        Model parameters: Inherited from Model base class
        N (int): Number of atoms in the chain (chain length in x-direction)

    Attributes:
        All Model attributes plus:
        N (int): Chain length (number of connected atoms)
        hessian (np.ndarray): Dynamical matrix of the central chain region
    """

    def __init__(self, k_coupl_x_l, k_c_x, k_coupl_x_r, interact_potential, interaction_range, lattice_constant, atom_type, N): 
        super().__init__(k_coupl_x_l, k_c_x, k_coupl_x_r, interact_potential, interaction_range, lattice_constant, atom_type)
        self.N = N
        self.hessian = self._build_hessian()

    def _build_hessian(self) -> np.ndarray:
        """Build the hessian matrix for a 1D chain.

        Returns:
            hessian (np.ndarray): Hessian matrix of shape (N, N)
        """

        assert self.interaction_range < self.N, "Interaction range must be smaller than the number of atoms in the chain!"

        hessian = np.zeros((self.N, self.N), dtype=float)

        all_k_coupl_x_l = mg.ranged_force_constant(k_coupl_x=self.k_coupl_x_l)["k_coupl_x"]
        all_k_c_x = mg.ranged_force_constant(k_c_x=self.k_c_x)["k_c_x"]
        all_k_coupl_x_r = mg.ranged_force_constant(k_coupl_x=self.k_coupl_x_r)["k_coupl_x"]

        for i in range(self.N):
            
            atomnr = i + 1

            # take care of interaction range
            for j in range(self.interaction_range):
                
                if i + j + 1 < self.N:
                    hessian[i, i + j + 1] = -all_k_c_x[j]
                if i - j - 1 >= 0:
                    hessian[i, i - j - 1] = -all_k_c_x[j]
            
            hessian[i, i] = -np.sum(hessian[i, :]) 
            
            # left side
            if atomnr - self.interaction_range <= 0:
                hessian[i, i] += sum(all_k_coupl_x_l[k] for k in range(self.interaction_range) if atomnr - (k + 1) <= 0)

            # middle atom within interaction range
            elif atomnr - self.interaction_range == 0 and atomnr + self.interaction_range >= self.N:
                
                # must be case sensitive for interaction range == 1
                if self.interaction_range == 1:
                    hessian[i, i] += all_k_coupl_x_l[0]
                else:
                    hessian[i, i] += all_k_coupl_x_l[-1] + all_k_coupl_x_r[-1]
            
            # right side
            elif atomnr + self.interaction_range > self.N:
                hessian[i, i] += sum(all_k_coupl_x_r[k] for k in range(self.interaction_range) if atomnr + (k + 1) > self.N)

        return hessian
    
class FiniteLattice2D(Model):
    """
    This class creates a 2D finite lattice with a given number of atoms (N_x, N_y). Inherits from the Model class.

    Args:
        Model (object): Inherits arguments from Model class.
        N_x (int): Number of atoms in x-direction.
        N_y (int): Number of atoms in y-direction.
        N_y_el_L (int): Number of electrons in the left (L) electrode (y-direction).
        N_y_el_R (int): Number of electrons in the left (R) electrode (y-direction).
        k_c_y (float): Coupling constant within the central part in y-direction.
        k_c_xy (float): Coupling constant within the central part in xy-direction.
        k_coupl_xy_l (float): Coupling constant between left electrode and center in xy-direction.
        k_coupl_xy_r (float): Coupling constant between right electrode and center in xy-direction.


    Attributes:
        All attributes of Model motherclass.
        N_x (int): Number of atoms in x-direction.
        N_y (int): Number of atoms in y-direction.
        N_y_el_L (int): Number of electrons in the left (L) electrode (y-direction).
        N_y_el_R (int): Number of electrons in the left (R) electrode (y-direction).
        k_c_y (float): Coupling constant within the central part in y-direction.
        k_c_xy (float): Coupling constant within the central part in xy-direction.
        k_coupl_xy_l (float): Coupling constant between left electrode and center in xy-direction.
        k_coupl_xy_r (float): Coupling constant between right electrode and center in xy-direction.

    """

    def __init__(self, N_y, N_x, N_y_el_L, N_y_el_R, k_coupl_x_l, k_c_x, k_coupl_x_r, k_c_y, k_c_xy, k_coupl_xy_l, k_coupl_xy_r, 
                 interact_potential="reciproke_squared", interaction_range=1, lattice_constant=3.0, atom_type="Au"):
        super().__init__(k_coupl_x_l, k_c_x, k_coupl_x_r, interact_potential, interaction_range, lattice_constant, atom_type)
        self.N_x = N_x
        self.N_y = N_y
        self.N_y_el_L = N_y_el_L
        self.N_y_el_R = N_y_el_R
        self.atom_type = atom_type
        self.k_values_l = mg.ranged_force_constant(k_coupl_x=k_coupl_x_l, k_coupl_xy=k_coupl_xy_l)
        self.k_values_c = mg.ranged_force_constant(k_c_x=k_c_x, k_c_y=k_c_y, k_c_xy=k_c_xy)
        self.k_values_r = mg.ranged_force_constant(k_coupl_x=k_coupl_x_r, k_coupl_xy=k_coupl_xy_r)
        self.hessian = self._build_hessian()

    def _build_hessian(self) -> np.ndarray:
        """
        Build the hessinan matrix for a 2D finite lattice including variable neighbor coupling/interaction range.

        Returns:
            hessian (np.ndarray): Hessian matrix of shape (2 * N_y * N_x, 2 * N_y * N_x)
        """
        
        assert (self.interaction_range < self.N_y or self.interaction_range < self.N_x), (
            "Interaction range must be smaller than the number of atoms in x- and y-direction!"
        )
        assert (self.interaction_range <= self.N_x // 2), (
            "Interaction range must be smaller than half the number of atoms in x-direction! (In order of simplicity and physical relevance)"
        )
        
        def build_bulk_layers(left=False, right=False) -> list[tuple[int, np.ndarray]]:
            """
            Building bulk submatrices until the layer where the full interaction range is reached. Returns combination of layer index from apart from the surface and its corresponding matrix.

            Args:
                left, right (bool): Flag for which electrode the coupling is needed.
            
            Returns:
                List of tuples: Contains combination of layer index from apart from the surface and its corresponding matrix as np.ndarray.

            """

            bulk_hessians = list()

            all_k_coupl_x_l = self.k_values_l["k_coupl_x"]
            all_k_c_x = self.k_values_c["k_c_x"]
            all_k_c_y = self.k_values_c["k_c_y"]
            all_k_c_xy = self.k_values_c["k_c_xy"]
            all_k_coupl_x_r = self.k_values_r["k_coupl_x"]

            for i in range(1, self.interaction_range + 1):

                hNN = np.zeros((2 * self.N_y, 2 * self.N_y), dtype=float)

                for j in range(hNN.shape[0]):

                    # diagonal elements x coupling
                    if j % 2 == 0:
                        
                        if right != False or left != False:
                            hNN[j, j] = sum(2 * all_k_c_x[k] for k in range(i)) 
                            hNN[j, j] += sum(all_k_c_x[k] for k in range(i, self.interaction_range) if (self.N_x > self.interaction_range and k <= self.N_x - self.interaction_range))
                        
                            if left == True and right == False:
                                hNN[j, j] += sum(all_k_coupl_x_l[k] for k in range(i, self.interaction_range))
                            
                            elif right == True and left == False:
                                hNN[j, j] += sum(all_k_coupl_x_r[k] for k in range(i, self.interaction_range))
                                
                        if left == False and right == False:
                            hNN[j, j] = sum(2 * all_k_c_x[k] for k in range(self.interaction_range))

                        # xy-coupling
                        if j == 0 or j == hNN.shape[0] - 2:
                            hNN[j, j] += 2 * all_k_c_xy[0]
                            hNN[j + 1, j + 1] += 2 * all_k_c_xy[0]

                        if j != 0 and j != hNN.shape[0] - 2 and self.N_y > 2:
                            hNN[j, j] += 4 * all_k_c_xy[0]
                            hNN[j + 1, j + 1] += 4 * all_k_c_xy[0]
                        

                    else:
                        # y coupling in the coupling range
                        if j == 1 or j == hNN.shape[0] - 1:
                            hNN[j, j] = all_k_c_y[0] + 2 * all_k_c_xy[0]

                            if j == 1:
                                hNN[j, j + 2] = -all_k_c_y[0]
                            else:
                                hNN[j, j - 2] = -all_k_c_y[0]

                            if self.interaction_range >= self.N_y:
                                for k in range(1, self.N_y - 1):
                                    hNN[j, j] += all_k_c_y[k]
                                    
                                    if j + 2 * (k + 1) < hNN.shape[0]:
                                        hNN[j, j + 2 * (k + 1)] = -all_k_c_y[k]
                                    if j - 2 * (k + 1) >= 0:
                                        hNN[j, j - 2 * (k + 1)] = -all_k_c_y[k]

                            else:
                                for k in range(1, self.interaction_range):
                                    hNN[j, j] += all_k_c_y[k]
                                
                                    if j + 2 * (k + 1) < hNN.shape[0]:
                                        hNN[j, j + 2 * (k + 1)] = -all_k_c_y[k]
                                    if j - 2 * (k + 1) >= 0:
                                        hNN[j, j - 2 * (k + 1)] = -all_k_c_y[k]
                            
                        else:
                            hNN[j, j] += 2 * all_k_c_y[0]
                            
                            if j + 2 < hNN.shape[0]:
                                hNN[j, j + 2] = -all_k_c_y[0]
                            if j - 2 >= 0:
                                hNN[j, j - 2] = -all_k_c_y[0]

                            atomnr = np.ceil(float(j) / 2)

                            if self.interaction_range >= self.N_y:
                                for k in range(1, self.N_y - 1):
                                    if atomnr - k - 1 > 0 and atomnr + k < self.N_y:
                                        hNN[j, j] += 2 * all_k_c_y[k]
                                        
                                    elif (atomnr - k - 1 <= 0 and atomnr + k < self.N_y) or (atomnr - k - 1 > 0 and atomnr + k >= self.N_y):
                                        hNN[j, j] += all_k_c_y[k]
                                    
                                    if j + 2 * (k + 1) < hNN.shape[0]:
                                        hNN[j, j + 2 * (k + 1)] = -all_k_c_y[k]
                                    if j - 2 * (k + 1) >= 0:
                                        hNN[j, j - 2 * (k + 1)] = -all_k_c_y[k]

                            else:
                                for k in range(1, self.interaction_range):
                                    if atomnr - k - 1 > 0 and atomnr + k < self.N_y:
                                        hNN[j, j] += 2 * all_k_c_y[k]
                                    elif (atomnr - k - 1 <= 0 and atomnr + k < self.N_y) or (atomnr - k - 1 > 0 and atomnr + k >= self.N_y):
                                        hNN[j, j] += all_k_c_y[k]
                                    
                                    if j + 2 * (k + 1) < hNN.shape[0]:
                                        hNN[j, j + 2 * (k + 1)] = -all_k_c_y[k]
                                    if j - 2 * (k + 1) >= 0:
                                        hNN[j, j - 2 * (k + 1)] = -all_k_c_y[k]

                bulk_hessians.append((i, hNN))
                
                if left == False and right == False:
                    break
                            
            return bulk_hessians
        
        def build_layer_interactions() -> list[tuple[int, np.ndarray]]:
            """
            Builds interaction matrices for the layers in the x-direction. The interaction range is taken into account.

            Returns:
                List of tuples: Contains combination of layer index and its corresponding interaction matrix as np.ndarray.

            """

            interact_layer_list = list()
            all_k_c_x = self.k_values_c["k_c_x"]
            all_k_c_xy = self.k_values_c["k_c_xy"]

            for i in range(self.interaction_range):
                h_interact = np.zeros((2 * self.N_y, 2 * self.N_y), dtype=float)
            
                for j in range(h_interact.shape[0]):

                    # diagonal elements x coupling
                    if j % 2 == 0:
                        h_interact[j, j] = -all_k_c_x[i]

                        if i == 0:
                            # xy-coupling
                            if j == 0:
                                h_interact[j, j + 2] += -all_k_c_xy[0]
                                h_interact[j + 2, j] += -all_k_c_xy[0]

                            elif j == h_interact.shape[0] - 2:
                                h_interact[j + 1, j - 1] += -all_k_c_xy[0]
                                h_interact[j - 1, j + 1] += -all_k_c_xy[0]

                            else:
                                h_interact[j, j + 2] += -all_k_c_xy[0]
                                h_interact[j + 2, j] += -all_k_c_xy[0]
                                h_interact[j + 1, j - 1] += -all_k_c_xy[0]
                                h_interact[j - 1, j + 1] += -all_k_c_xy[0]


                interact_layer_list.append((i, h_interact))

            return interact_layer_list
        
        def build_H_00(left=False, right=False) -> np.ndarray:
            """
            Build the hessian matrix for the first layer. The interaction range is taken into account.
            
            Returns:
                H_00 (np.ndarray): Hessian matrix of shape (2 * N_y, 2 * N_y)
            """
            
            H_00 = np.zeros((2 * self.N_y, 2 * self.N_y), dtype=float)
            
            all_k_coupl_x_l = self.k_values_l["k_coupl_x"]
            all_k_coupl_x_r = self.k_values_r["k_coupl_x"]
            all_k_coupl_xy_l = self.k_values_l["k_coupl_xy"]
            all_k_coupl_xy_r = self.k_values_r["k_coupl_xy"]
            all_k_c_x = self.k_values_c["k_c_x"]
            all_k_c_y = self.k_values_c["k_c_y"]
            all_k_c_xy = self.k_values_c["k_c_xy"]
            
            for i in range(H_00.shape[0]):

                if i % 2 == 0:
                    # x coupling in the coupling range
                    H_00[i, i] = sum(all_k_c_x[k] for k in range(self.interaction_range) if (self.N_x > self.interaction_range and k <= self.N_x - self.interaction_range))
                    
                    if left == True and right == False:
                        H_00[i, i] += sum(all_k_coupl_x_l)

                    elif right == True and left == False:
                        H_00[i, i] += sum(all_k_coupl_x_r)


                    if i == 0 or i == H_00.shape[0] - 2:
                        # xy coupling
                        H_00[i, i] += all_k_c_xy[0]
                        H_00[i + 1, i + 1] += all_k_c_xy[0]

                        if left == True and right == False:
                            if self.N_y == self.N_y_el_L:
                                H_00[i, i] += all_k_coupl_xy_l[0]
                                H_00[i + 1, i + 1] += all_k_coupl_xy_l[0]
                            else:
                                H_00[i, i] += 2 * all_k_coupl_xy_l[0]
                                H_00[i + 1, i + 1] += 2 * all_k_coupl_xy_l[0]
                        
                        elif right == True and left == False:
                            if self.N_y == self.N_y_el_R:
                                H_00[i, i] += all_k_coupl_xy_r[0]
                                H_00[i + 1, i + 1] += all_k_coupl_xy_r[0]
                            else:
                                H_00[i, i] += 2 * all_k_coupl_xy_r[0]
                                H_00[i + 1, i + 1] += 2 * all_k_coupl_xy_r[0]

                    else:
                        H_00[i, i] += 2 * all_k_c_xy[0]
                        H_00[i + 1, i + 1] += 2 * all_k_c_xy[0]

                        if left == True and right == False:
                            H_00[i, i] += 2 * all_k_coupl_xy_l[0]
                            H_00[i + 1, i + 1] += 2 * all_k_coupl_xy_l[0]

                        elif right == True and left == False:
                            H_00[i, i] += 2 * all_k_coupl_xy_r[0]
                            H_00[i + 1, i + 1] += 2 * all_k_coupl_xy_r[0]

                    
                else:
                    # y coupling in the coupling range, first and last k_y, rest 2 * k_c_y
                    if i == 1 or i == H_00.shape[0] - 1:

                        H_00[i, i] += all_k_c_y[0]
                        if i == 1:
                            H_00[i, i + 2] = -all_k_c_y[0]
                        else:
                            H_00[i, i - 2] = -all_k_c_y[0]

                        if self.interaction_range >= self.N_y:
                            for k in range(1, self.N_y - 1):
                                H_00[i, i] += all_k_c_y[k]
                                
                                if i + 2 * (k + 1) < H_00.shape[0]:
                                    H_00[i, i + 2 * (k + 1)] = -all_k_c_y[k]
                                if i - 2 * (k + 1) >= 0:
                                    H_00[i, i - 2 * (k + 1)] = -all_k_c_y[k]

                        else:
                            for k in range(1, self.interaction_range):
                                H_00[i, i] += all_k_c_y[k]
                            
                                if i + 2 * (k + 1) < H_00.shape[0]:
                                    H_00[i, i + 2 * (k + 1)] = -all_k_c_y[k]
                                if i - 2 * (k + 1) >= 0:
                                    H_00[i, i - 2 * (k + 1)] = -all_k_c_y[k]
                        
                    else:
                        H_00[i, i] += 2 * all_k_c_y[0]
                        
                        if i + 2 < H_00.shape[0]:
                            H_00[i, i + 2] = -all_k_c_y[0]
                        if i - 2 >= 0:
                            H_00[i, i - 2] = -all_k_c_y[0]

                        atomnr = np.ceil(float(i) / 2)

                        if self.interaction_range >= self.N_y:
                            for k in range(1, self.N_y - 1):
                                if atomnr - k - 1 > 0 and atomnr + k < self.N_y:
                                    H_00[i, i] += 2 * all_k_c_y[k]
                                    
                                elif (atomnr - k - 1 <= 0 and atomnr + k < self.N_y) or (atomnr - k - 1 > 0 and atomnr + k >= self.N_y):
                                    H_00[i, i] += all_k_c_y[k]
                                
                                if i + 2 * (k + 1) < H_00.shape[0]:
                                    H_00[i, i + 2 * (k + 1)] = -all_k_c_y[k]
                                if i - 2 * (k + 1) >= 0:
                                    H_00[i, i - 2 * (k + 1)] = -all_k_c_y[k]

                        else:
                            for k in range(1, self.interaction_range):
                                if atomnr - k - 1 > 0 and atomnr + k < self.N_y:
                                    H_00[i, i] += 2 * all_k_c_y[k]
                                elif (atomnr - k - 1 <= 0 and atomnr + k < self.N_y) or (atomnr - k - 1 > 0 and atomnr + k >= self.N_y):
                                    H_00[i, i] += all_k_c_y[k]
                                
                                if i + 2 * (k + 1) < H_00.shape[0]:
                                    H_00[i, i + 2 * (k + 1)] = -all_k_c_y[k]
                                if i - 2 * (k + 1) >= 0:
                                    H_00[i, i - 2 * (k + 1)] = -all_k_c_y[k]
            
            return H_00

        # special case: 1D chain in 2D
        if self.N_y == 1:

            hessian = np.zeros((2 * self.N_y * self.N_x, 2 * self.N_y * self.N_x), dtype=float)
            all_k_coupl_x_l = self.k_values_l["k_coupl_x"]
            all_k_coupl_x_r = self.k_values_r["k_coupl_x"]
            all_k_coupl_xy_l = self.k_values_l["k_coupl_xy"]
            all_k_coupl_xy_r = self.k_values_r["k_coupl_xy"]
            all_k_c_x = self.k_values_c["k_c_x"]
        
            ### -------------- START 1D chain as 2D lattice -------------- ###
            for i in range(self.N_x):
                
                atomnr = i + 1
                    
                # take care of interaction range
                for j in range(self.interaction_range):
                    
                    if i + j + 1 < self.N_x:
                        hessian[2 * i, 2 * (i + j + 1)] = -all_k_c_x[j]
                    if i - j - 1 >= 0:
                        hessian[2 * i, 2 * (i - j - 1)] = -all_k_c_x[j]

                hessian[2 * i, 2 * i] = -np.sum(hessian[2 * i, :])
                
                if atomnr == 1 or atomnr == self.N_x:
                    #xy coupling to the electrodes
                    if atomnr == 1 and self.N_y_el_L > 1:   
                        hessian[2 * i, 2 * i] += 2 * all_k_coupl_xy_l[0]
                        hessian[2 * i + 1, 2 * i + 1] += 2 * all_k_coupl_xy_l[0]
                        
                    elif atomnr == self.N_x and self.N_y_el_R > 1:
                        hessian[2 * i, 2 * i] += 2 * all_k_coupl_xy_r[0]
                        hessian[2 * i + 1, 2 * i + 1] += 2 * all_k_coupl_xy_r[0]

                # left side
                if atomnr - self.interaction_range <= 0:
                    hessian[2 * i, 2 * i] += sum(all_k_coupl_x_l[k] for k in range(self.interaction_range) if atomnr - (k + 1) <= 0)
                
                # middle atom within interaction range
                elif atomnr - self.interaction_range == 0 and atomnr + self.interaction_range >= self.N_x:
                    
                    # must be case sensitive for interaction range == 1
                    if self.interaction_range == 1:
                        hessian[2 * i, 2 * i] += all_k_coupl_x_l[0]
                    else:
                        hessian[2 * i, 2 * i] += all_k_coupl_x_l[-1] + sum(all_k_coupl_x_r[-k] for k in range(self.N_x - atomnr, self.interaction_range))
                
                # right side
                elif atomnr + self.interaction_range > self.N_x:
                    hessian[2 * i, 2 * i] += sum(all_k_coupl_x_r[k] for k in range(self.interaction_range) if atomnr + (k + 1) > self.N_x)

            return hessian
        
         ### -------------- END 1D Chain case -------------- ###

        layer_interactions = build_layer_interactions()
        H_00_l = build_H_00(left=True, right=False)
        H_00_r = build_H_00(left=False, right=True)
        bulk_layers = build_bulk_layers()
        bulk_layers_l = build_bulk_layers(left=True, right=False)
        bulk_layers_r = build_bulk_layers(left=False, right=True)
    
        hessian = np.zeros((2 * self.N_y * self.N_x, 2 * self.N_y * self.N_x), dtype=float)

        for i in range(self.N_x):

            #surface layers + interaction
            if (i == 0 or i == self.N_x - 1) and self.N_x > 1:
                
                #interaction with the layers within the interaction range
                if i == 0:
    
                    hessian[i * H_00_l.shape[0]: i * H_00_l.shape[0] + H_00_l.shape[0], # rows
                        i * H_00_l.shape[0]: i * H_00_l.shape[0] + H_00_l.shape[0]] = H_00_l
                    
                    for j in range(len(layer_interactions)):
                        if i + j + 1 < self.N_x:
                            hessian[i * H_00_l.shape[0]: i * H_00_l.shape[0] + H_00_l.shape[0], 
                                    (i + j + 1) * H_00_l.shape[0]: (i + j + 1) * H_00_l.shape[0] + H_00_l.shape[0]] = layer_interactions[j][1]
                else:

                    hessian[i * H_00_r.shape[0]: i * H_00_r.shape[0] + H_00_r.shape[0], # rows
                        i * H_00_r.shape[0]: i * H_00_r.shape[0] + H_00_r.shape[0]] = H_00_r

                    for j in range(len(layer_interactions)):
                        if i - j - 1 >= 0:
                            hessian[i * H_00_r.shape[0]: i * H_00_r.shape[0] + H_00_r.shape[0], 
                                    (i - j - 1) * H_00_r.shape[0]: (i - j - 1) * H_00_r.shape[0] + H_00_r.shape[0]] = layer_interactions[j][1]
                        
            #bulk layers + interaction
            elif i <= len(bulk_layers_l) and i < self.N_x // 2:
                hessian[i * H_00_l.shape[0]: i * H_00_l.shape[0] + H_00_l.shape[0], 
                        i * H_00_l.shape[0]: i * H_00_l.shape[0] + H_00_l.shape[0]] = bulk_layers_l[i - 1][1]
                
                #interaction with the layers within the interaction range, depending on how many layers are to the left and right
                for j in range(len(layer_interactions)):
                    if i - j - 1 >= 0:
                        hessian[i * H_00_l.shape[0]: i * H_00_l.shape[0] + H_00_l.shape[0], 
                                (i - j - 1) * H_00_l.shape[0]: (i - j - 1) * H_00_l.shape[0] + H_00_l.shape[0]] = layer_interactions[j][1]
                    if i + j + 1 < self.N_x:
                        hessian[i * H_00_l.shape[0]: i * H_00_l.shape[0] + H_00_l.shape[0], 
                                (i + j + 1) * H_00_l.shape[0]: (i + j + 1) * H_00_l.shape[0] + H_00_l.shape[0]] = layer_interactions[j][1]

            elif i >= self.N_x - len(bulk_layers_r) - 1:#and i >= self.N_x // 2:
                hessian[i * H_00_l.shape[0]: i * H_00_l.shape[0] + H_00_l.shape[0], 
                        i * H_00_l.shape[0]: i * H_00_l.shape[0] + H_00_l.shape[0]] = bulk_layers_r[self.N_x - i - 2][1]

                #interaction with the layers within the interaction range, depending on how many layers are to the left and right
                for j in range(len(layer_interactions)):
                    if i - j - 1 >= 0:
                        hessian[i * H_00_l.shape[0]: i * H_00_l.shape[0] + H_00_l.shape[0], 
                                (i - j - 1) * H_00_l.shape[0]: (i - j - 1) * H_00_l.shape[0] + H_00_l.shape[0]] = layer_interactions[j][1]
                    if i + j + 1 < self.N_x:
                        hessian[i * H_00_l.shape[0]: i * H_00_l.shape[0] + H_00_l.shape[0], 
                                (i + j + 1) * H_00_l.shape[0]: (i + j + 1) * H_00_l.shape[0] + H_00_l.shape[0]] = layer_interactions[j][1]

            else:
                hessian[i * H_00_l.shape[0]: i * H_00_l.shape[0] + H_00_l.shape[0], 
                        i * H_00_l.shape[0]: i * H_00_l.shape[0] + H_00_l.shape[0]] = bulk_layers[-1][1]

                #interaction with the layers within the interaction range
                for j in range(len(layer_interactions)):
                    if i - j - 1 >= 0:
                        hessian[i * H_00_l.shape[0]: i * H_00_l.shape[0] + H_00_l.shape[0], 
                                (i - j - 1) * H_00_l.shape[0]: (i - j - 1) * H_00_l.shape[0] + H_00_l.shape[0]] = layer_interactions[j][1]
                    if i + j + 1 < self.N_x:
                        hessian[i * H_00_l.shape[0]: i * H_00_l.shape[0] + H_00_l.shape[0], 
                                (i + j + 1) * H_00_l.shape[0]: (i + j + 1) * H_00_l.shape[0] + H_00_l.shape[0]] = layer_interactions[j][1]
                
        return hessian


if __name__ == '__main__':

    #TODO: Doesn't work for interaction_range > N_x // 2 --> Fix this
    junction2D = FiniteLattice2D(N_y=1, N_x=2, N_y_el_L=3, N_y_el_R=1, k_l_x=900, k_c_x=900, k_r_x=900, k_c_y=900, k_c_xy=180, k_l_xy=180, k_r_xy=180, interaction_range=1)
    junction1D = Chain1D(interact_potential="reciproke_squared", interaction_range=2, lattice_constant=3.0, atom_type="Au", k_c=900, k_l=900, k_r=900, N=4)
    print('debugging')






