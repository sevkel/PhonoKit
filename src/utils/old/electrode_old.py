__docformat__ = "google"

import codecs
import configparser
from functools import partial
import numpy as np
import scipy
import matplotlib
import tmoutproc as top
matplotlib.use('agg')
from matplotlib import pyplot as plt
from scipy.integrate import quad
from scipy.integrate import simps
import ray
from utils import constants

from utils import constants

class Electrode:
    """
    Motherclass for the definition of different electrode models. The class contains the greens function g0 and g, which are calculated.
    """

    def __init__(self, w, interaction_range=1, interact_potential="reciproke_squared", atom_type="Au", lattice_constant=3.0):
        self.w = w
        self.cfg = configparser.ConfigParser()
        #self.cfg.read_file(codecs.open(config_path, "r", "utf8"))
        #self.data_path = str(self.cfg.get('Data Input', 'data_path'))
        self.interaction_range = interaction_range
        self.interact_potential = interact_potential
        self.atom_type = atom_type
        self.lattice_constant = lattice_constant

    def plot_g0(self):
        fig, ax1 = plt.subplots()
        ax1.plot(self.w * constants.unit2SI * constants.h_bar * constants.J2meV, np.imag(self.g0) * self.k_c, color="red", label="Im(g0)")
        ax1.plot(self.w * constants.unit2SI * constants.h_bar * constants.J2meV, np.real(self.g0) * self.k_c, color="green", label="Re(g0)")
        plt.xlabel("Energy (meV)")
        plt.ylabel(r"g $(1/k_c)$")
        plt.grid()
        plt.legend()
        plt.title(r"$g_0$")
        plt.show()
        plt.savefig(self.data_path + "/g0_electrode.pdf", bbox_inches='tight')

    def plot_g(self):
        fig, ax1 = plt.subplots()
        ax1.plot(self.w * constants.unit2SI * constants.h_bar * constants.J2meV, np.imag(self.g) * self.k_c, color="red", label="Im(g0)")
        ax1.plot(self.w * constants.unit2SI * constants.h_bar * constants.J2meV, np.real(self.g) * self.k_c, color="green", label="Re(g0)")
        ax1.set_ylim(-1.5, 1.5)
        plt.xlabel("Energy (meV)")
        plt.ylabel(r"g $(1/k_c)$")
        plt.grid()
        plt.legend()
        plt.title(r"$g_{\mathrm{surf}}}$")
        plt.savefig(self.data_path + "/g_electrode.pdf", bbox_inches='tight')
        plt.show()

class DebeyeModel(Electrode):
    """
    Set up the electrode description via the greens function g0 and g according to the Debeye model. Inherits from the Electrode class.
    """


    def __init__(self, w, k_c, w_D):
        super().__init__(w)
        self.k_c = k_c * (constants.eV2hartree / constants.ang2bohr ** 2)
        self.w_D = w_D

        self.g0 = self.calculate_g0(w, w_D)
        self.g = self.calculate_g(self.g0)

        assert isinstance(self.k_c, (int, float)), "In the Debeye model k_c must be a single numeric value (int or float), not an array."


    def calculate_g0(self, w, w_D):
        """Calculates surface greens function according to Markussen, T. (2013). Phonon interference effects in molecular junctions. The Journal of chemical physics, 139(24), 244101 (https://doi.org/10.1063/1.4849178).

        Args:
            w (np.ndarray): Frequencies where g0 is calculated
            w_D (float): Debeye frequency
            k_c (float): Coupling constant to the center part
            interaction_range (int): Interaction range -> 1 = nearest neighbor, 2 = next nearest neighbor, etc.

        Returns:
            g0 (np.ndarray): Surface greens function g0
        """

        def im_g(w):

            if (w <= w_D):
                Im_g = -np.pi * 3.0 * w / (2 * w_D ** 3)
            else:
                Im_g = 0

            return Im_g

        Im_g = map(im_g, w)
        Im_g = np.asarray(list(Im_g))
        Re_g = -np.asarray(np.imag(scipy.signal.hilbert(Im_g)))
        g0 = np.asarray((Re_g + 1.j * Im_g), complex)

        return g0

    def calculate_g(self, g_0):
        """Calculates coupled surface greens function

        Args:
            g_0 (np.ndarray): Uncoupled surface greens function

        Returns:
            g (np.ndarray)) Surface greens function coupled by dyson equation
        """
        #TODO: Vorzeichen check
        g = g_0 / (1 - self.k_c * g_0)

        return g

class Chain1D(Electrode):
    """
    Class for the definition of a one-dimensional chain. Inherits from the Electrode class.
    """

    def __init__(self, w, interaction_range, interact_potential, atom_type, lattice_constant, k_x, k_c):
        super().__init__(w, interaction_range, interact_potential, atom_type, lattice_constant)
        self.k_x = k_x * (constants.eV2hartree / constants.ang2bohr ** 2)
        self.k_c = k_c * (constants.eV2hartree / constants.ang2bohr ** 2)
        
        self.g0 = self.calculate_g0()
        self.g = self.calculate_g(self.g0)

    def ranged_force_constant(self):
        """
        Calculate ranged force constants for the 1D chain dependend on which potential is used and on how many neighbors are coupled.
        
        Retruns:
            range_force_constant (list of tuples): Ranged force constant for the 2D lattice
        """

        match self.interact_potential:

            case "reciproke_squared":
                all_k_x = list(enumerate((self.k_x * (1 / (i * self.lattice_constant)**2) for i in range(1, self.interaction_range + 1))))
                all_k_c = list(enumerate((self.k_c * (1 / (i * self.lattice_constant)**2) for i in range(1, self.interaction_range + 1))))
            
            case _:
                raise ValueError("Invalid interaction potential. Choose either 'reciproke_squared', .")
            
        return all_k_x, all_k_c

    def calculate_g0(self):
        """Calculates surface greens of one-dimensional chain (nearest neighbor coupling) with coupling parameter k

        Args:
            w (array_like): Frequency where g0 is calculated
            k_x (float): Coupling constant within the chosen interaction range

        Returns:
            g0 (array_like): Surface greens function g0
        """

        all_k_x = self.ranged_force_constant()[0]
        k_x = sum(k_x for _, k_x in all_k_x)


        g_0 = 1 / (2 * k_x * self.w) * (self.w - np.sqrt(self.w**2 - 4 * k_x, dtype=complex)) #Jan draft

        return g_0
    
    def calculate_g(self, g0):
        """
        Calculates surface greens of one-dimensional chain with coupling parameter k_x and k_c.
        """

        all_k_c = self.ranged_force_constant()[1]
        k_c = sum(k_c for _, k_c in all_k_c)

        #TODO: Vorzeichen check
        g = g0 / (1 + k_c * g0)

        return g
    
class Ribbon2D(Electrode):
    """
    Class for the definition of a two-dimensional ribbon. Inherits from the Electrode class.
    """

    def __init__(self, w, interaction_range, interact_potential, atom_type, lattice_constant, N_y, N_y_scatter, M_L, M_C, k_x, k_y, k_xy, k_c, k_c_xy): 
        super().__init__(w, interaction_range, interact_potential, atom_type, lattice_constant)
        self.N_y = N_y
        self.N_y_scatter = N_y_scatter
        self.k_x = k_x * (constants.eV2hartree / constants.ang2bohr ** 2)
        self.k_y = k_y * (constants.eV2hartree / constants.ang2bohr ** 2)
        self.k_xy = k_xy * (constants.eV2hartree / constants.ang2bohr ** 2)
        self.k_c = k_c * (constants.eV2hartree / constants.ang2bohr ** 2)
        self.k_c_xy = k_c_xy * (constants.eV2hartree / constants.ang2bohr ** 2)
        self.M_L = M_L
        self.M_C = M_C
        self.eps = 1E-50
        self.g0 = self.calculate_g0()
        self.g = self.calculate_g(self.g0)

        assert self.N_y - self.N_y_scatter >= 0, "The number of atoms in the scattering region must be smaller than the number of atoms in the electrode. Please check your input parameters."
        assert (self.N_y - self.N_y_scatter) % 2 == 0, "The configuration must be symmetric in y-direction. Please check your input parameters."

    def ranged_force_constant(self):
        """
        Calculate ranged force constants for the 2D Ribbon electrode dependend on which potential is used and on how many neighbors are coupled.
        
        Retruns:
            range_force_constant (list of tuples): Ranged force constant for the 2D lattice
        """

        match self.interact_potential:

            case "reciproke_squared":
                all_k_x = list(enumerate((self.k_x * (1 / (i * self.lattice_constant)**2) for i in range(1, self.interaction_range + 1))))
                all_k_y = list(enumerate((self.k_y * (1 / (i * self.lattice_constant)**2) for i in range(1, self.interaction_range + 1))))
                all_k_xy =  list(enumerate((self.k_xy * (1 / (i * self.lattice_constant)**2) for i in range(1, self.interaction_range + 1))))
                all_k_c_x = list(enumerate((self.k_c * (1 / (i * self.lattice_constant)**2) for i in range(1, self.interaction_range + 1))))
                all_k_c_xy = list(enumerate((self.k_c_xy * (1 / (i * self.lattice_constant)**2) for i in range(1, self.interaction_range + 1))))
            
            case _:
                raise ValueError("Invalid interaction potential. Choose either 'reciproke_squared', .")
            
        return all_k_x, all_k_y, all_k_xy, all_k_c_x, all_k_c_xy

    def calculate_g0(self):
        """Calculates surface greens 2d half infinite square lattice with finite width N_y. The uncoupled surface greens function g0 is calculated according to:
        "Highly convergent schemes for the calculation of bulk and surface Green functions", M P Lopez Sancho etal 1985 J.Phys.F:Met.Phys. 15 851
        

        Args:
            w (array_like): Frequency where g0 is calculated

        Returns:
            g0	(array_like) Surface greens function g0
        """

        def build_H_NN():
            """
            Build up an actual bulk layer of the electrode. The coupling includes options for x, y and xy coupling. The coupling range is defined by the parameter interaction_range.
            """

            all_k_x, all_k_y, all_k_xy = self.ranged_force_constant()[0], self.ranged_force_constant()[1], self.ranged_force_constant()[2]

            # Build submatrices to set up the bulk layer
            
            H_NN = np.zeros((2 * self.N_y, 2 * self.N_y), dtype=float)
           
            for i in range(0, H_NN.shape[0]):

                if i % 2 == 0:
                    # x coupling in the coupling range
                    H_NN[i, i] = sum(2 * k_x for _, k_x in all_k_x)

                    # xy-coupling
                    if i == 0 or i == H_NN.shape[0] - 2:
                        H_NN[i, i + 1] = 2 * all_k_xy[0][1]
                        H_NN[i + 1, i] = 2 * all_k_xy[0][1]

                    if i != 0 and i != H_NN.shape[0] - 2 and self.N_y > 2:
                        H_NN[i, i + 1] = 4 * all_k_xy[0][1]
                        H_NN[i + 1, i] = 4 * all_k_xy[0][1]

                else:
                        # y coupling in the coupling range, first and last k_y, rest 2k_y
                        if i == 1 or i == H_NN.shape[0] - 1:
                            H_NN[i, i] = all_k_y[0][1]

                            if i == 1:
                                H_NN[i, i + 2] = -all_k_y[0][1]
                            else:
                                H_NN[i, i - 2] = -all_k_y[0][1]

                            if self.interaction_range >= self.N_y:
                                for k in range(1, self.N_y - 1):
                                    H_NN[i, i] += all_k_y[k][1]
                                    
                                    if i + 2 * (k + 1) < H_NN.shape[0]:
                                        H_NN[i, i + 2 * (k + 1)] = -all_k_y[k][1]
                                    if i - 2 * (k + 1) >= 0:
                                        H_NN[i, i - 2 * (k + 1)] = -all_k_y[k][1]

                            else:
                                for k in range(1, self.interaction_range):
                                    H_NN[i, i] += all_k_y[k][1]
                                
                                    if i + 2 * (k + 1) < H_NN.shape[0]:
                                        H_NN[i, i + 2 * (k + 1)] = -all_k_y[k][1]
                                    if i - 2 * (k + 1) >= 0:
                                        H_NN[i, i - 2 * (k + 1)] = -all_k_y[k][1]
                            
                        else:
                            H_NN[i, i] = 2 * all_k_y[0][1]
                            
                            if i + 2 < H_NN.shape[0]:
                                H_NN[i, i + 2] = -all_k_y[0][1]
                            if i - 2 >= 0:
                                H_NN[i, i - 2] = -all_k_y[0][1]

                            atomnr = np.ceil(float(i) / 2)

                            if self.interaction_range >= self.N_y:
                                for k in range(1, self.N_y - 1):
                                    if atomnr - k - 1 > 0 and atomnr + k < self.N_y:
                                        H_NN[i, i] += 2 * all_k_y[k][1]
                                        
                                    elif (atomnr - k - 1 <= 0 and atomnr + k < self.N_y) or (atomnr - k - 1 > 0 and atomnr + k >= self.N_y):
                                        H_NN[i, i] += all_k_y[k][1]
                                    
                                    if i + 2 * (k + 1) < H_NN.shape[0]:
                                        H_NN[i, i + 2 * (k + 1)] = -all_k_y[k][1]
                                    if i - 2 * (k + 1) >= 0:
                                        H_NN[i, i - 2 * (k + 1)] = -all_k_y[k][1]

                            else:
                                for k in range(1, self.interaction_range):
                                    if atomnr - k - 1 > 0 and atomnr + k < self.N_y:
                                        H_NN[i, i] += 2 * all_k_y[k][1]
                                    elif (atomnr - k - 1 <= 0 and atomnr + k < self.N_y) or (atomnr - k - 1 > 0 and atomnr + k >= self.N_y):
                                        H_NN[i, i] += all_k_y[k][1]
                                    
                                    if i + 2 * (k + 1) < H_NN.shape[0]:
                                        H_NN[i, i + 2 * (k + 1)] = -all_k_y[k][1]
                                    if i - 2 * (k + 1) >= 0:
                                        H_NN[i, i - 2 * (k + 1)] = -all_k_y[k][1]
            
            return H_NN
        
        def build_H_00():
            """
            Build the hessian matrix for the first layer. The interaction range is taken into account.
            
            Returns:
                H_00 (np.ndarray): Hessian matrix of shape (2 * N_y, 2 * N_y)
            """
            
            H_00 = np.zeros((2 * self.N_y, 2 * self.N_y), dtype=float)
            all_k_x, all_k_y, all_k_xy = self.ranged_force_constant()[0], self.ranged_force_constant()[1], self.ranged_force_constant()[2]

            for i in range(H_00.shape[0]):

                if(i % 2 == 0):
                    # x coupling in the coupling range
                    H_00[i, i] = sum(all_k_x[k][1] for k in range(self.interaction_range))
                    
                    if i == 0 or i == H_00.shape[0] - 2:
                        # xy coupling
                        H_00[i, i + 1] = all_k_xy[0][1]
                        H_00[i + 1, i] = all_k_xy[0][1]

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

        def build_H_01():
            """
            Build the hessian matrix for the interaction between two princial layers. The interaction range is taken into account.
            """
            
            #take care of higher interaction range than nearest neighbor
            H_01 = np.zeros((2 * self.N_y, 2 * self.N_y), dtype=float)
            all_k_x, all_k_xy = self.ranged_force_constant()[0], self.ranged_force_constant()[2]

            for i in range(H_01.shape[0]):
                #x components
                if i % 2 == 0:

                    H_01[i, i] = -all_k_x[0][1]
                    
                    # xy-coupling
                    if i == 0:
                        H_01[i, i + 3] = -all_k_xy[0][1]
                        H_01[i + 3, i] = -all_k_xy[0][1]

                    elif i == H_01.shape[0] - 2:
                        H_01[i, i - 1] = -all_k_xy[0][1]
                        H_01[i - 1, i] = -all_k_xy[0][1]

                    else:
                        H_01[i, i + 3] = -all_k_xy[0][1]
                        H_01[i + 3, i] = -all_k_xy[0][1]
                        H_01[i, i - 1] = -all_k_xy[0][1]
                        H_01[i - 1, i] = -all_k_xy[0][1]
                
            return H_01


        def build_H_NN_new():
            """
            Build up an actual bulk layer of the electrode. The coupling includes options for x, y and xy coupling. The coupling range is defined by the parameter interaction_range.
            """

            N_y = self.N_y
            interaction_range = self.interaction_range

            all_k_x, all_k_y, all_k_xy = self.ranged_force_constant()[0:3]

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
                                hNN[j, int(2 * (atomnr + N_y + 1)) - 2] = -all_k_xy[0][1]

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

            return hNN

        def build_H_00_new():
            """
            Build the hessian matrix for the first layer. The interaction range is taken into account.
            
            Returns:
                H_00 (np.ndarray): Hessian matrix of shape (2 * N_y, 2 * N_y)
            """

            N_y = self.N_y
            interaction_range = self.interaction_range

            all_k_x, all_k_y, all_k_xy = self.ranged_force_constant()[0:3]

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

            return h00
        
        def build_H_01_new():
            """
            Build the hessian matrix for the interaction between two princial layers. The interaction range is taken into account.
            """

            N_y = self.N_y
            interaction_range = self.interaction_range

            all_k_x, all_k_y, all_k_xy = self.ranged_force_constant()[0:3]
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
            
            return h01
                    

        #H_NN = build_H_NN()
        #H_00 = build_H_00()
        #H_01 = build_H_01()

        H_NN = build_H_NN_new()
        H_00 = build_H_00_new()
        H_01 = build_H_01_new()

        H_01_dagger = np.transpose(np.conj(H_01))

        #TODO: Check for Bulk layers, H_01 is only the interaction from 0 to 1, even tho it contains more then NN interaction in H_00 and H_NN -> ASR can't be fulfilled
        #assert np.sum(H_00 + H_01) == 0 and np.sum(H_NN + 2 * H_01) == 0, "Check the sum rule of the layers!"
        
        # Start decimation algorithm

        def calc_g0_w(w):
            w = np.identity(H_NN.shape[0]) * (w**2 + (1.j * 1E-24))
            g = np.linalg.inv(w - H_NN) 
            alpha_i = np.dot(np.dot(H_01, g), H_01)
            beta_i = np.dot(np.dot(H_01_dagger, g), H_01_dagger)
            epsilon_is = H_00 + np.dot(np.dot(H_01, g), H_01_dagger)
            epsilon_i = H_NN + np.dot(np.dot(H_01, g), H_01_dagger) + np.dot(np.dot(H_01_dagger, g), H_01)
            delta = np.abs(2 * np.trace(alpha_i)) 
            deltas = list()
            deltas.append(delta)
            counter = 0
            terminated = False
            while delta > self.eps:
                counter += 1

                if counter > 10000:
                    terminated = True
                    break

                g = np.linalg.inv(w - epsilon_i)
                epsilon_i = epsilon_i + np.dot(np.dot(alpha_i, g), beta_i) + np.dot(np.dot(beta_i, g), alpha_i)
                epsilon_is = epsilon_is + np.dot(np.dot(alpha_i, g), beta_i)
                alpha_i = np.dot(np.dot(alpha_i, g), alpha_i)
                beta_i = np.dot(np.dot(beta_i, g), beta_i)
                delta = np.abs(2 * np.trace(alpha_i))
                deltas.append(delta)

            if delta >= self.eps or terminated:
                print("Warning! Decimation algorithm did not converge. Delta: ", delta)

            g_0 = np.linalg.inv(w - epsilon_is)
        
            return g_0

        g_0 = map(calc_g0_w, self.w)
        g_0 = np.array([item for item in g_0])
        
        return  g_0, H_01
        
    def calculate_g(self, g_0, H_01):
        """Calculates surface greens of 2d half infinite square lattice. Taking into the interaction range.

        Args:
            g_0 (array_like): Uncoupled surface greens function

        Returns:
            g (array_like): Surface greens function coupled by dyson equation
        """

    
        # Build coupling/interaction matrix between electrode and scattering region


        """for i in range(K_lc_1c.shape[0]):

            if i % 2 == 0 and i >= first_interact_row and i < first_interact_row + self.N_y_scatter * 2:
                
                K_lc_1c[i, i - first_interact_row] = -all_k_c_x[0][1]

                if i == first_interact_row:
                    K_lc_1c[i, 3] = -all_k_c_xy[0][1]
                    K_lc_1c[i + 3, 0] = -all_k_c_xy[0][1]

                else:

                    if i - first_interact_row + 3 < K_lc_1c.shape[1]:
                        K_lc_1c[i, i - first_interact_row + 3] = -all_k_c_xy[0][1]   

                    K_lc_1c[i + 3, i - first_interact_row] = -all_k_c_xy[0][1]
                    K_lc_1c[i, i - first_interact_row - 1] = -all_k_c_xy[0][1]
                    K_lc_1c[i - 1, i - first_interact_row] = -all_k_c_xy[0][1]
            
            elif i == first_interact_row - 2 and self.N_y - self.N_y_scatter >= 2:
                K_lc_1c[i, 1] = -all_k_c_xy[0][1]
                K_lc_1c[i + 1, 0] = -all_k_c_xy[0][1]

            elif i == first_interact_row + self.N_y_scatter * 2 and self.N_y - self.N_y_scatter >= 2:
                K_lc_1c[i, K_lc_1c.shape[1] - 1] = -all_k_c_xy[0][1]
                K_lc_1c[i + 1, K_lc_1c.shape[1] - 2] = -all_k_c_xy[0][1]
        
        K_LC = np.zeros((self.N_y * 2, self.N_y * 2), dtype=float)
        K_LC[0:K_lc_1c.shape[0], 0:K_lc_1c.shape[1]] = K_lc_1c"""

        N_y = self.N_y
        N_y_scatter = self.N_y_scatter
        interaction_range = self.interaction_range

        if N_y_scatter == N_y:
            k_lc_LL = H_01

        direct_interaction = np.zeros((2 * (N_y_scatter + 2), 2 * N_y_scatter), dtype=float)
        all_k_c_x, all_k_c_xy = self.ranged_force_constant()[3], self.ranged_force_constant()[4]

        for i in range(0, direct_interaction.shape[0], 2):
            
            if 0 < i <= direct_interaction.shape[1]:
                direct_interaction[i, i - 2] = -all_k_c_x[0][1]
                
                if i - 2 + 3 <= direct_interaction.shape[1] - 1:
                    direct_interaction[i, i - 2 + 3] = -all_k_c_xy[0][1]
                    direct_interaction[i + 1, i - 2 + 2] = -all_k_c_xy[0][1]
                if i - 2 - 2 >= 0:
                    direct_interaction[i, i - 2 - 1] = -all_k_c_xy[0][1]
                    direct_interaction[i + 1, i - 2 - 2] = -all_k_c_xy[0][1]

            # xy coupling
            if i == 0:
                direct_interaction[i, i + 1] = -all_k_c_xy[0][1]
                direct_interaction[i + 1, i] = -all_k_c_xy[0][1]

            elif i == direct_interaction.shape[0] - 2:
                direct_interaction[i, direct_interaction.shape[1] - 1] = -all_k_c_xy[0][1]
                direct_interaction[i + 1, direct_interaction.shape[1] - 2] = -all_k_c_xy[0][1]

        #TODO: get on the fly force constants between electrode and junction (or anything?)
        k_lc_LL = np.zeros((2 * N_y * interaction_range, 2 * N_y * interaction_range), dtype=float) 

        for i in range(interaction_range):
            for j in range(i * 2 * N_y, i * 2 * N_y + 2 * N_y, 2):
                if j % 2 == 0:
                    atomnr = np.ceil(float(j + 1) / 2)

                    if i * N_y + 1 + (N_y - N_y_scatter) / 2 <= atomnr <= i * N_y + N_y - (N_y - N_y_scatter) / 2:
                        k_lc_LL[j, j] = all_k_c_x[-1][1]

                    if i == interaction_range - 1 and atomnr == i * N_y +  (N_y - N_y_scatter) / 2:

                        atomnr_b = int(1 + (N_y - N_y_scatter) / 2)
                        k_lc_LL[j: j + direct_interaction.shape[0], atomnr_b: atomnr_b + direct_interaction.shape[1]] = direct_interaction


                    
        g = map(lambda x: np.dot(x, np.linalg.inv(np.identity(x.shape[0]) + np.dot(k_lc_LL, x))), g_0)
        g = np.array([item for item in g])
        
        return g, k_lc_LL


if __name__ == '__main__':

    N = 750
    E_D = 50
    # convert to J
    E_D = E_D * constants.meV2J
    # convert to 1/s
    w_D = E_D / constants.h_bar
    # convert to har*s/(bohr**2*u)
    w_D = w_D / constants.unit2SI
    w = np.linspace(w_D * 1E-12, w_D * 1.1, N)
    k_c = 0.1 

    electrode_debeye = DebeyeModel(w, k_c, w_D)
    electrode_chain1d = Chain1D(w, interaction_range=1, interact_potential='reciproke_squared', atom_type="Au", lattice_constant=3.0, k_x=0.1, k_c=0.1)
    electrode_2dribbon = Ribbon2D(w, interaction_range=1, interact_potential='reciproke_squared', atom_type="Au", lattice_constant=3.0, N_y=9, N_y_scatter=5, M_L=1, M_C=1, k_x=0.1, k_y=0.1, k_xy=0.33, k_c=0.1, k_c_xy=0.33)

    print("debug")
