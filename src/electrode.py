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
from scipy.integrate import simpson
from utils import constants


class Electrode:
    """
    Motherclass for the definition of different electrode models. The class contains the greens function g0 and g, which are calculated. By default, calculate left electrode.
    """

    def __init__(self, w, interaction_range=1, interact_potential="reciproke_squared", atom_type="Au", lattice_constant=3.0, left=True, right=False):
        self.w = w
        self.interaction_range = interaction_range
        self.interact_potential = interact_potential
        self.atom_type = atom_type
        self.lattice_constant = lattice_constant
        self.left = left
        self.right = right

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
        self.k_c = k_c * 1#(constants.eV2hartree / constants.ang2bohr ** 2)
        self.w_D = w_D

        self.g0 = self.calculate_g0(w, w_D)
        self.g = self.calculate_g(self.g0)[0]
        self.dos, self.dos_real = self.calculate_g(self.g0)[1], self.calculate_g(self.g0)[2]
        self.dos_cpld, self.dos_real_cpld = self.calculate_g(self.g0)[3], self.calculate_g(self.g0)[4]

        assert isinstance(self.k_c, (int, float)), "In the Debeye model k_c must be a single numeric value (int or float), not an array."
        
    def ranged_force_constant(self):
        """
        Calculate ranged force constants for the 1D chain dependend on which potential is used and on how many neighbors are coupled.
        
        Retruns:
            range_force_constant (list of tuples): Ranged force constant for the 2D lattice
        """

        match self.interact_potential:

            case "reciproke_squared":
                all_k_c = list(enumerate((self.k_c * (1 / (i * self.lattice_constant)**2) for i in range(1, self.interaction_range + 1))))
            
            case _:
                raise ValueError("Invalid interaction potential. Choose either 'reciproke_squared', .")
            
        return all_k_c

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
        
        g = g_0 / (1 + self.k_c * g_0)
        dos = (-1 / np.pi) * np.imag(g_0)
        dos_real = np.real(g_0)
        
        dos_cpld = (-1 / np.pi) * np.imag(g)
        dos_real_cpld = np.real(g)


        return g, dos, dos_real, dos_cpld, dos_real_cpld

class Chain1D(Electrode):
    """
    Class for the definition of a one-dimensional chain. Inherits from the Electrode class.
    """

    def __init__(self, w, interaction_range, interact_potential, atom_type, lattice_constant, k_x, k_c):
        super().__init__(w, interaction_range, interact_potential, atom_type, lattice_constant)
        self.k_x = k_x * 1#(constants.eV2hartree / constants.ang2bohr ** 2)
        self.k_c = k_c * 1#(constants.eV2hartree / constants.ang2bohr ** 2)
        
        self.g0 = self.calculate_g0()
        self.g = self.calculate_g(self.g0)
        self.dos, self.dos_real = self.calculate_g(self.g0)[1], self.calculate_g(self.g0)[2]
        self.dos_cpld, self.dos_real_cpld = self.calculate_g(self.g0)[3], self.calculate_g(self.g0)[4]
        

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
        
        #TODO: is this really correct by summing k_x and k_c in that way?
        all_k_x = self.ranged_force_constant()[0]
        k_x = sum(k_x for _, k_x in all_k_x)


        g_0 = 1 / (2 * k_x * self.w) * (self.w - np.sqrt(self.w**2 - 4 * k_x, dtype=complex)) #Jan PhD Thesis p.29

        return g_0
    
    def calculate_g(self, g0):
        """
        Calculates surface greens of one-dimensional chain with coupling parameter k_x and k_c.
        """

        all_k_c = self.ranged_force_constant()[1]

        # because interaction takes the whole range
        k_c = sum(k_c for _, k_c in all_k_c)
        
        g = g0 / (1 + k_c * g0)
        
        dos = (-1 / np.pi) * np.imag(g0)
        dos_real = np.real(g0)
        
        dos_cpld = (-1 / np.pi) * np.imag(g)
        dos_real_cpld = np.real(g)

        return g, dos, dos_real, dos_cpld, dos_real_cpld
    
class Ribbon2D(Electrode):
    """
    Class for the definition of a two-dimensional ribbon. Inherits from the Electrode class.
    """

    def __init__(self, w, interaction_range, interact_potential, atom_type, lattice_constant, left, right, N_y, N_y_scatter, M_L, M_C, k_x, k_y, k_xy, k_c, k_c_xy): 
        super().__init__(w, interaction_range, interact_potential, atom_type, lattice_constant, left, right)
        self.N_y = N_y
        self.N_y_scatter = N_y_scatter
        self.k_x = k_x * 1#(constants.eV2hartree / constants.ang2bohr ** 2)
        self.k_y = k_y * 1#(constants.eV2hartree / constants.ang2bohr ** 2)
        self.k_xy = k_xy * 1#(constants.eV2hartree / constants.ang2bohr ** 2)
        self.k_c = k_c * 1#(constants.eV2hartree / constants.ang2bohr ** 2)
        self.k_c_xy = k_c_xy * 1#(constants.eV2hartree / constants.ang2bohr ** 2)
        self.M_L = M_L
        self.M_C = M_C
        self.eps = 1E-50
        self.g0, self.H_01 = self.calculate_g0() #H_01 only needed if N_y == N_y_scatter
        self.g = self.calculate_g(self.g0, self.H_01)[0]
        self.dos, self.dos_real = self.calculate_g(self.g0, self.H_01)[3], self.calculate_g(self.g0, self.H_01)[4]
        self.dos_cpld, self.dos_real_cpld = self.calculate_g(self.g0, self.H_01)[5], self.calculate_g(self.g0, self.H_01)[6]
        self.k_lc_LL = self.calculate_g(self.g0, self.H_01)[1]
        self.direct_interaction = self.calculate_g(self.g0, self.H_01)[2]

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
                        
                        atomnr = np.ceil(float(j + 1) / 2)
                        
                        # ii-coupling
                        hNN[j, j] = sum(2 * all_k_x[k][1] for k in range(len(all_k_x))) 
            

                        for k in range(interaction_range):
                            if j + 2 * (k + 1) * N_y < hNN.shape[0]:                    
                                hNN[j, j + 2 * (k + 1) * N_y] = -all_k_x[k][1]
                            if j - 2 * (k + 1) * N_y >= 0:
                                hNN[j, j - 2 * (k + 1) * N_y] = -all_k_x[k][1]

                        # ij-coupling in h01

                        # xy-coupling
                        if N_y > 1:
                            
                            if j == i * 2 * N_y or j == i * 2 * N_y + 2 * N_y - 2:
                                hNN[j, j + 1] = 2 * all_k_xy[0][1]
                                hNN[j + 1, j] = 2 * all_k_xy[0][1]

                            if j != 0 + i * 2 * N_y and j != i * 2 * N_y + 2 * N_y - 2 and N_y > 2:
                                hNN[j, j + 1] = 4 * all_k_xy[0][1]
                                hNN[j + 1, j] = 4 * all_k_xy[0][1]
                        

                    else:
                        if N_y > 1:
                            # y coupling in the coupling range -> edge layers
                            if (j == i * 2 * N_y + 1) or (j == i * 2 * N_y + 2 * N_y - 1): 
                                
                                # xy-coupling
                                atomnr = np.ceil(float(j) / 2)

                                if interaction_range > 1:
                                    
                                    if atomnr < N_y:# and interaction_range > 1:
                                        hNN[j - 1, int(j - 1 + 2 * (atomnr + N_y + 1) - 1)] = -all_k_xy[0][1]
                                        hNN[j, int(j - 1 + 2 * (atomnr + N_y + 1) - 2)] = -all_k_xy[0][1]

                                    elif atomnr ==  N_y and interaction_range > 1:
                                        hNN[j - 1, int(2 * (atomnr + N_y - 1) - 1)] = -all_k_xy[0][1]
                                        hNN[j, int(2 * (atomnr + N_y - 1) - 2)] = -all_k_xy[0][1]

                                    
                                    elif atomnr > N_y and interaction_range > 1 and (atomnr == (i + 1) * N_y or atomnr == i * N_y + 1) and atomnr <= N_y * interaction_range - N_y:
                                        # N_y == 2 case
                                        if atomnr == i * N_y + 1:
                                            hNN[j, 2 * int(i * N_y + 1 + N_y + 1) - 2] = -all_k_xy[0][1]
                                            hNN[j - 1, 2 * int(i * N_y + 1 + N_y + 1) - 1] = -all_k_xy[0][1]
                                            
                                            hNN[j, 2 * int(i * N_y + 1 - N_y + 1) - 2] = -all_k_xy[0][1]
                                            hNN[j - 1, 2 * int(i * N_y + 1 - N_y + 1) - 1] = -all_k_xy[0][1]
                                        
                                        elif atomnr == (i + 1) * N_y:
                                            hNN[j, 2 * int((i + 1) * N_y + N_y - 1) - 2] = -all_k_xy[0][1]
                                            hNN[j - 1, 2 * int((i + 1) * N_y + N_y - 1) - 1] = -all_k_xy[0][1]
                                            
                                            hNN[j, 2 * int((i + 1) * N_y - N_y - 1) - 2] = -all_k_xy[0][1]
                                            hNN[j - 1, 2 * int((i + 1) * N_y - N_y - 1) - 1] = -all_k_xy[0][1]
                                            
                                    elif (atomnr == N_y * interaction_range - N_y + 1 or atomnr == N_y * interaction_range):
                                        if atomnr == N_y * interaction_range - N_y + 1:
                                            hNN[j, 2 * int(N_y * interaction_range - N_y + 1 - N_y + 1) - 2] = -all_k_xy[0][1]
                                            hNN[j - 1, 2 * int(N_y * interaction_range - N_y + 1 - N_y + 1) - 1] = -all_k_xy[0][1]
                                        elif atomnr == N_y * interaction_range:
                                            hNN[j, 2 * int(N_y * interaction_range - N_y - 1) - 2] = -all_k_xy[0][1]
                                            hNN[j - 1, 2 * int(N_y * interaction_range - N_y - 1) - 1] = -all_k_xy[0][1]


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
                                        if j - 2 * (k + 1) >= i * 2 * N_y:
                                            hNN[j, j - 2 * (k + 1)] = -all_k_y[k][1]

                                else:
                                    for k in range(1, interaction_range):
                                        hNN[j, j] += all_k_y[k][1]
                                    
                                        if j + 2 * (k + 1) < i * 2 * N_y + 2 * N_y:
                                            hNN[j, j + 2 * (k + 1)] = -all_k_y[k][1]
                                        if j - 2 * (k + 1) >= i * 2 * N_y:
                                            hNN[j, j - 2 * (k + 1)] = -all_k_y[k][1]


                            else:
                                
                                atomnr = np.ceil(float(j) / 2)
                                hNN[j, j] = 2 * all_k_y[0][1]
                                
                                # xy-coupling inner atom
                                if interaction_range > 1:
                                    if atomnr < N_y:# and interaction_range > 1:
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
                                        if j - 2 * (k + 1) > i * 2 * N_y:
                                            hNN[j, j - 2 * (k + 1)] = -all_k_y[k][1]

                                else:
                                    for k in range(1, interaction_range):
                                        if atomnr - k - 1 > i * N_y and atomnr + k < N_y + i * N_y:#N_y * interaction_range:
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
                        
                        elif atomnr > i * N_y and atomnr <= (i + 1) * N_y and (i + 1) * N_y < N_y * interaction_range:
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

                        # xy-coupling # TODO: do something to take account that only for Ny > 1 possible or leave it to the user?
                        if N_y > 1:
                            
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
                        
                        if N_y > 1:
                            # y coupling in the coupling range -> edge layers/atoms
                            if (j == i * 2 * N_y + 1) or (j == i * 2 * N_y + 2 * N_y - 1): 
                                
                                # xy-coupling
                                atomnr = np.ceil(float(j) / 2)

                                if interaction_range > 1:
                                    
                                    if atomnr < N_y: #and interaction_range > 1:
                                        h00[j - 1, int(j - 1 + 2 * (atomnr + N_y + 1) - 1)] = -all_k_xy[0][1]
                                        h00[j, int(j - 1 + 2 * (atomnr + N_y + 1) - 2)] = -all_k_xy[0][1]

                                    elif atomnr == N_y and interaction_range > 1:
                                        h00[j - 1, int(2 * (atomnr + N_y - 1) - 1)] = -all_k_xy[0][1]
                                        h00[j, int(2 * (atomnr + N_y - 1) - 2)] = -all_k_xy[0][1]

                                    elif atomnr > N_y and interaction_range > 1 and (atomnr == (i + 1) * N_y or atomnr == i * N_y + 1) and atomnr <= N_y * interaction_range - N_y:
                                        # N_y == 2 case
                                        if atomnr == i * N_y + 1:
                                            h00[j, 2 * int(i * N_y + 1 + N_y + 1) - 2] = -all_k_xy[0][1]
                                            h00[j - 1, 2 * int(i * N_y + 1 + N_y + 1) - 1] = -all_k_xy[0][1]
                                            
                                            h00[j, 2 * int(i * N_y + 1 - N_y + 1) - 2] = -all_k_xy[0][1]
                                            h00[j - 1, 2 * int(i * N_y + 1 - N_y + 1) - 1] = -all_k_xy[0][1]
                                        
                                        elif atomnr == (i + 1) * N_y:
                                            h00[j, 2 * int((i + 1) * N_y + N_y - 1) - 2] = -all_k_xy[0][1]
                                            h00[j - 1, 2 * int((i + 1) * N_y + N_y - 1) - 1] = -all_k_xy[0][1]
                                            
                                            h00[j, 2 * int((i + 1) * N_y - N_y - 1) - 2] = -all_k_xy[0][1]
                                            h00[j - 1, 2 * int((i + 1) * N_y - N_y - 1) - 1] = -all_k_xy[0][1]
                                            
                                    elif (atomnr == N_y * interaction_range - N_y + 1 or atomnr == N_y * interaction_range):
                                        
                                        if atomnr == N_y * interaction_range - N_y + 1:
                                            h00[j, 2 * int(N_y * interaction_range - N_y + 1 - N_y + 1) - 2] = -all_k_xy[0][1]
                                            h00[j - 1, 2 * int(N_y * interaction_range - N_y + 1 - N_y + 1) - 1] = -all_k_xy[0][1]
                                        elif atomnr == N_y * interaction_range:
                                            h00[j, 2 * int(N_y * interaction_range - N_y - 1) - 2] = -all_k_xy[0][1]
                                            h00[j - 1, 2 * int(N_y * interaction_range - N_y - 1) - 1] = -all_k_xy[0][1]



                                #y - coupling
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
                                        if j - 2 * (k + 1) >= i * 2 * N_y:
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
                                
                                # xy-coupling inner atom, inner layers
                                if interaction_range > 1:
                                    
                                    if atomnr < N_y: #and interaction_range > 1:
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
                                        if j - 2 * (k + 1) > i * 2 * N_y:
                                            h00[j, j - 2 * (k + 1)] = -all_k_y[k][1]

                                else:
                                    for k in range(1, interaction_range):
                                        if atomnr - k - 1 > i * N_y and atomnr + k < N_y + i * N_y:#N_y * interaction_range:
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
            
            for i in range(h01.shape[0]):
                for j in range(h01.shape[1]):
                    if i % 2 == 0 and j % 2 == 0:
                        atomnr_lay1 = np.ceil(float(i + 1) / 2)
                        atomnr_lay2 = np.ceil(float(j + 1) / 2)
                        
                        if atomnr_lay1 == atomnr_lay2:
                            h01[i, j] = -all_k_x[-1][1]
                        
                        if interaction_range > 1:
                            # for more than next nearest neighbour coupling in x-direction
                            for r in range(interaction_range - 1, -1, -1):
                                
                                if atomnr_lay2 == atomnr_lay1 - r * N_y:
                                    h01[i, j] = -all_k_x[-(r + 1)][1]
                            
                            # xy coupling
                            if (interaction_range - 1) * N_y < atomnr_lay1 <= interaction_range * N_y and atomnr_lay2 <= N_y:
                                # edge atoms
                                if atomnr_lay1 == (interaction_range - 1) * N_y + 1 and atomnr_lay2 == 2:
                                    h01[i, j + 1] = -all_k_xy[0][1]
                                    h01[i + 1, j] = -all_k_xy[0][1]
                                
                                elif atomnr_lay1 == interaction_range * N_y and atomnr_lay2 == N_y - 1:
                                    h01[i, j + 1] = -all_k_xy[0][1]
                                    h01[i + 1, j] = -all_k_xy[0][1]
                                
                                # middle atoms
                                elif atomnr_lay2 == atomnr_lay1 - (interaction_range - 1) * N_y + 1 or atomnr_lay2 == atomnr_lay1 - (interaction_range - 1) * N_y - 1:
                                    h01[i, j + 1] = -all_k_xy[0][1]
                                    h01[i + 1, j] = -all_k_xy[0][1]
                         
                        else:
                            # edge atoms
                            if atomnr_lay1 == 1 and atomnr_lay2 == 2:
                                h01[i, j + 1] = -all_k_xy[0][1]
                                h01[i + 1, j] = -all_k_xy[0][1]
                            elif atomnr_lay1 == N_y and atomnr_lay2 == N_y - 1:
                                h01[i, j + 1] = -all_k_xy[0][1]
                                h01[i + 1, j] = -all_k_xy[0][1]
                            elif atomnr_lay2 == atomnr_lay1 - 1 or atomnr_lay2 == atomnr_lay1 + 1:
                                h01[i, j + 1] = -all_k_xy[0][1]
                                h01[i + 1, j] = -all_k_xy[0][1]
                            
                            # make matrix now symmetric
                            h01[j, i] = h01[i, j]        
                                                        

            return h01
                    

        H_NN = build_H_NN_new()
        H_00 = build_H_00_new()
        H_01 = build_H_01_new()
        
        assert (0 <= np.abs(np.sum(H_00 + H_01)) < 1E-10), "Sum rule violated! H_00 + H_01 is not zero! Check the force constants and the interaction range."
        assert (0 <= np.abs(np.sum(H_NN + 2 * H_01)) < 1E-10), "Sum rule violated! H_NN + 2 * H_01 is not zero! Check the force constants and the interaction range."
        H_01_dagger = np.transpose(np.conj(H_01))
        
        # Start decimation algorithm

        def calc_g0_w(w):
            w = np.identity(H_NN.shape[0]) * (w**2 + (1.j * 1E-10))  # add small imaginary part to avoid singularities
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

            try:
                g_0 = np.linalg.inv(w - epsilon_is)
            except np.linalg.LinAlgError:
                g_0 = np.linalg.pinv(w - epsilon_is)
        
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

        N_y = self.N_y
        N_y_scatter = self.N_y_scatter
        interaction_range = self.interaction_range

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

        k_lc_LL = np.zeros((2 * N_y * interaction_range, 2 * N_y * interaction_range), dtype=float) 
        
        interaction_layers_dict = dict()
        
        for i in range(interaction_range):
            
            interaction_layer = np.zeros((2 * N_y, 2 * N_y), dtype=float)
            
            #if i == 0: # direct NN layer
            for j in range(interaction_layer.shape[0]):
                
                if j % 2 == 0:
                    atomnr = np.ceil(float(j + 1) / 2)
                    
                    if atomnr == ((N_y - N_y_scatter) // 2) or atomnr == ((N_y - N_y_scatter) // 2) + N_y_scatter + 1:
                        
                        if i == 0:
                            interaction_layer[j, j + 1] = -all_k_c_xy[0][1]
                            interaction_layer[j + 1, j] = -all_k_c_xy[0][1]
                    
                    elif ((N_y - N_y_scatter) // 2) < atomnr < ((N_y - N_y_scatter) // 2) + N_y_scatter + 1:
                        
                        interaction_layer[j, j] = -sum(all_k_c_x[k][1] for k in range(i, interaction_range))
                        
            
                        if i == 0:

                            if (atomnr == ((N_y - N_y_scatter) // 2) + 1 or atomnr == ((N_y - N_y_scatter) // 2) + N_y_scatter) and N_y_scatter > 1:
                                interaction_layer[j, j + 1] = -all_k_c_xy[0][1]
                                interaction_layer[j + 1, j] = -all_k_c_xy[0][1]
                            elif N_y_scatter > 1:
                                interaction_layer[j, j + 1] = -2 * all_k_c_xy[0][1]
                                interaction_layer[j + 1, j] = -2 * all_k_c_xy[0][1]
            
            interaction_layers_dict[i] = interaction_layer       
                        
                    
        for l in range(interaction_range):
            k_lc_LL[l * interaction_layers_dict[l].shape[0]: l * interaction_layers_dict[l].shape[0] + interaction_layers_dict[l].shape[0], \
                l * interaction_layers_dict[l].shape[0]: l * interaction_layers_dict[l].shape[0] + interaction_layers_dict[l].shape[0]] = interaction_layers_dict[interaction_range - 1 - l]
            
            
        g = map(lambda x: np.dot(np.linalg.inv(np.identity(x.shape[0]) + np.dot(x, k_lc_LL)), x), g_0)
        g = np.array([item for item in g])
        
        dos = (-1 / np.pi) * np.imag(np.trace(g_0, axis1=1, axis2=2))
        dos_real = np.real(np.trace(g_0, axis1=1, axis2=2))
        
        dos_cpld = (-1 / np.pi) * np.imag(np.trace(g, axis1=1, axis2=2))
        dos_real_cpld = np.real(np.trace(g, axis1=1, axis2=2))
        
        return g, k_lc_LL, direct_interaction, dos, dos_real, dos_cpld, dos_real_cpld

class InfiniteFourier2D(Electrode):
    """
    Calculates the coupled surface greens function for  a 2D infinite square lattice electrode.

    Args:
        Electrode (_type_): _description_
    """

    def __init__(self, w, interaction_range, interact_potential, atom_type, lattice_constant, N_q,  k_x, k_y, k_xy, k_c, k_c_xy, N_y_scatter):
        super().__init__(w, interaction_range, interact_potential, atom_type, lattice_constant)
        self.q = np.linspace(-np.pi, np.pi, N_q)
        self.k_x = k_x
        self.k_y = k_y
        self.k_xy = k_xy
        self.k_c = k_c
        self.k_c_xy = k_c_xy
        self.N_y_scatter = N_y_scatter
        self.g0 = self.calculate_g0()
        self.g = self.calculate_g()[0]
        self.k_lc_LL, self.dos, self.dos_real = self.calculate_g()[1], self.calculate_g()[2], self.calculate_g()[3]

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
        """Calculates the uncoupled surface greens 2d infinite square lattice. 

        Returns:
            g0	(array_like) Surface greens function g0
        """

        def calc_g0_w(w, k_y, k_x, a):  
            """
            Calculates the surface greens function g0_q in reciprocal space for a 2D infinite square lattice electrode.

            Args:
                q (float): Wave vector in reciprocal space
                w (float): Frequency in reciprocal space
                k_y (float): Force constant in y direction
                k_x (float): Force constant in x direction
            
            Returns:
                g0_q (float): Surface greens function g0_q in reciprocal space
            """

            def g0_q(q, w):

                w = w + (1j * 1E-24)

                y = k_y * np.sin(q * a / 2)**2
                g0_q = 2 * (w**2 - 4 * y + np.sqrt((w**2 - 4 * y) * (w**2 - 4 * k_x - 4 * y)))**(-1) 

                return g0_q
            
            g0_q_vals = np.array([g0_q(q_val, w) for q_val in self.q])
            
            g0 = (1 / (2 * np.pi)) * simpson(g0_q_vals, self.q)

            return g0
    

        # Build coupling/interaction matrix between electrode and scattering region
        all_k_x, all_k_y, all_k_xy = self.ranged_force_constant()[0:3]

        k_x = self.k_x
        k_y = self.k_y
        k_xy = self.k_xy
        a = self.lattice_constant

        g0 = map(lambda w: calc_g0_w(w, k_y, k_x, a), self.w)
        g0 = np.array([item for item in g0])

        return g0

    def calculate_g(self):
        """
        Calculates the surface greens function in real space to represent an infinite 2D square lattice electrode. 

        Returns:
            g0 (array_like): Surface greens function g0 in real space
        """

        N_y_scatter = self.N_y_scatter
        direct_interaction = np.zeros((2 * (N_y_scatter + 2), 2 * N_y_scatter), dtype=float)
        all_k_c_x, all_k_c_xy = self.ranged_force_constant()[3], self.ranged_force_constant()[4]
        g0_template = np.identity(direct_interaction.shape[0], dtype=float)

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

        k_lc_LL = g0_template.copy()

        k_lc_LL[:direct_interaction.shape[0], :direct_interaction.shape[1]] = direct_interaction

        g = map(lambda x: np.dot(x * g0_template, np.linalg.inv(np.identity(g0_template.shape[0]) + np.dot(k_lc_LL, x * g0_template))), self.g0)
        g = np.array([item for item in g])
        
        dos = (-1 / np.pi) * np.imag(self.g0)
        dos_real = np.real(self.g0)

        return g, k_lc_LL, dos, dos_real



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
    electrode_chain1d = Chain1D(w, interaction_range=2, interact_potential='reciproke_squared', atom_type="Au", lattice_constant=3.0, k_x=0.1, k_c=0.1)
    electrode_2dribbon = Ribbon2D(w, interaction_range=3, interact_potential='reciproke_squared', atom_type="Au", lattice_constant=3.0, N_y=1, N_y_scatter=1, M_L=1, M_C=1, k_x=900, k_y=900, k_xy=180, k_c=900, k_c_xy=180)
    electrode_infinite = InfiniteFourier2D(w, interaction_range=1, interact_potential='reciproke_squared', atom_type="Au", lattice_constant=3.0, N_q=100, N_y_scatter=1, k_x=180, k_y=180, k_xy=0, k_c=180, k_c_xy=0)

    print("debug")
