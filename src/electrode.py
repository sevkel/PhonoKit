__docformat__ = "google"

import sys
import numpy as np
import scipy
from joblib import Parallel, delayed
from scipy.integrate import simpson
import scipy.sparse as sps
from utils import constants, matrix_gen as mg


###

# Multiply every force constant with (constants.eV2hartree / constants.ang2bohr ** 2) if needed.

###


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

class DebeyeModel(Electrode): 
    #TODO: How to deal with a xy-coupling? Does it even make sense as there is no information about the electrode geometry in the Debye model?
    """
    Set up the electrode description via the greens function g0 and g according to the Debeye model. Inherits from the Electrode class.
    """

    def __init__(self, w, k_coupl_x, k_coupl_xy, w_D):
        super().__init__(w)
        self.k_coupl_x = k_coupl_x
        self.k_coupl_xy = k_coupl_xy
        self.w_D = w_D

        self.g0 = self.calculate_g0(w, w_D)
        self.g = self.calculate_g(self.g0)[0]
        self.dos, self.dos_real = self.calculate_g(self.g0)[1], self.calculate_g(self.g0)[2]
        self.dos_cpld, self.dos_real_cpld = self.calculate_g(self.g0)[3], self.calculate_g(self.g0)[4]

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
        
        g = g_0 / (1 + self.k_coupl_x * g_0)
        dos = (-1 / np.pi) * np.imag(g_0)
        dos_real = np.real(g_0)
        
        dos_cpld = (-1 / np.pi) * np.imag(g)
        dos_real_cpld = np.real(g)


        return g, dos, dos_real, dos_cpld, dos_real_cpld

class Chain1D(Electrode):
    """
    Class for the definition of a one-dimensional chain. Inherits from the Electrode class.
    """

    def __init__(self, w, interaction_range, interact_potential, atom_type, lattice_constant, k_el_x, k_coupl_x):
        super().__init__(w, interaction_range, interact_potential, atom_type, lattice_constant)
        self.k_x = k_el_x
        self.k_coupl_x = k_coupl_x
        
        self.g0 = self.calculate_g0()
        self.g = self.calculate_g(self.g0)
        self.dos, self.dos_real = self.calculate_g(self.g0)[1], self.calculate_g(self.g0)[2]
        self.dos_cpld, self.dos_real_cpld = self.calculate_g(self.g0)[3], self.calculate_g(self.g0)[4]

    def calculate_g0(self):
        """Calculates surface greens of one-dimensional chain (nearest neighbor coupling) with coupling parameter k

        Args:
            w (array_like): Frequency where g0 is calculated
            k_x (float): Coupling constant within the chosen interaction range

        Returns:
            g0 (array_like): Surface greens function g0
        """
        
        all_k_x = mg.ranged_force_constant(k_coupl_x=self.k_coupl_x)["k_coupl_x"]
        k_x = sum(k_x for _ in all_k_x)

        #Jan PhD Thesis p.29
        g_0 = 1 / (2 * k_x * self.w) * (self.w - np.sqrt(self.w**2 - 4 * k_x, dtype=np.complex64)) 

        return g_0
    
    def calculate_g(self, g0):
        """
        Calculates surface greens of one-dimensional chain with coupling parameter k_x and k_c.
        """

        all_k_coupl = mg.ranged_force_constant(k_coupl_x=self.k_coupl_x)["k_coupl_x"]

        # because interaction takes the whole range
        k_coupl = sum(k_c for _ in all_k_coupl)
        
        g = g0 / (1 + k_coupl * g0)
        
        dos = (-1 / np.pi) * np.imag(g0)
        dos_real = np.real(g0)
        
        dos_cpld = (-1 / np.pi) * np.imag(g)
        dos_real_cpld = np.real(g)

        return g, dos, dos_real, dos_cpld, dos_real_cpld
    
class Ribbon2D(Electrode):
    """
    Class for the definition of a two-dimensional ribbon. Inherits from the Electrode class.
    """

    def __init__(self, w, interaction_range, interact_potential, atom_type, lattice_constant, left, right, N_y, N_y_scatter, M_L, M_C, k_el_x, k_el_y, k_el_xy, k_coupl_x, k_coupl_xy): 
        super().__init__(w, interaction_range, interact_potential, atom_type, lattice_constant, left, right)
        self.N_y = N_y
        self.N_y_scatter = N_y_scatter
        self.k_el_x = k_el_x 
        self.k_el_y = k_el_y 
        self.k_el_xy = k_el_xy 
        self.k_coupl_x = k_coupl_x 
        self.k_coupl_xy = k_coupl_xy 
        self.M_L = M_L
        self.M_C = M_C
        self.eps = 1E-50
        self.g0, self.H_01 = self.calculate_g0() #H_01 only needed if N_y == N_y_scatter
        self.g, self.k_lc_LL, self.direct_interaction = self.calculate_g(self.g0, self.H_01)[0:3]
        self.dos, self.dos_real = self.calculate_g(self.g0, self.H_01)[3:5]
        self.dos_cpld, self.dos_real_cpld = self.calculate_g(self.g0, self.H_01)[5:]

        assert self.N_y - self.N_y_scatter >= 0, "The number of atoms in the scattering region must be smaller than the number of atoms in the electrode. Please check your input parameters."
        assert (self.N_y - self.N_y_scatter) % 2 == 0, "The configuration must be symmetric in y-direction. Please check your input parameters."

    def calculate_g0(self):
        """Calculates surface greens 2d half infinite square lattice with finite width N_y. The uncoupled surface greens function g0 is calculated according to:
        "Highly convergent schemes for the calculation of bulk and surface Green functions", M P Lopez Sancho etal 1985 J.Phys.F:Met.Phys. 15 851
        

        Args:
            w (array_like): Frequency where g0 is calculated

        Returns:
            g0	(array_like) Surface greens function g0
        """
        k_values = mg.ranged_force_constant(k_el_x=self.k_el_x, k_el_y=self.k_el_y, k_el_xy=self.k_el_xy)

        H_NN = mg.build_H_NN(self.N_y, self.interaction_range, k_values=k_values)
        H_00 = mg.build_H_00(self.N_y, self.interaction_range, k_values=k_values)
        H_01 = mg.build_H_01(self.N_y, self.interaction_range, k_values=k_values)

        assert (0 <= np.abs(np.sum(H_00 + H_01)) < 1E-10), "Sum rule violated! H_00 + H_01 is not zero! Check the force constants and the interaction range."
        assert (0 <= np.abs(np.sum(H_NN + 2 * H_01)) < 1E-10), "Sum rule violated! H_NN + 2 * H_01 is not zero! Check the force constants and the interaction range."
        H_01_dagger = np.transpose(np.conj(H_01))
        
        # Start decimation algorithm

        def calc_g0_w(w):
            w_temp = w
            w = np.identity(H_NN.shape[0]) * (w + (1.j * 1E-7))**2  # add small imaginary part to avoid singularities
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

                try:
                    g = np.linalg.inv(w - epsilon_i)
                except np.linalg.LinAlgError:
                    print(f"Matrix regularization was applied during the decimation algorithm for w = {w_temp}.")
                    g = np.nan_to_num((w - epsilon_i), nan=0.0, posinf=0.0, neginf=0.0)
                    g = np.linalg.inv(g + 1e-8 * np.eye(g.shape[0]))
                    
                epsilon_i = epsilon_i + np.dot(np.dot(alpha_i, g), beta_i) + np.dot(np.dot(beta_i, g), alpha_i)
                epsilon_is = epsilon_is + np.dot(np.dot(alpha_i, g), beta_i)
                alpha_i = np.dot(np.dot(alpha_i, g), alpha_i)
                beta_i = np.dot(np.dot(beta_i, g), beta_i)
                delta = np.abs(2 * np.trace(alpha_i))
                deltas.append(delta)

            if delta >= self.eps or terminated:
                print("Warning! Decimation algorithm did not converge. Delta: ", delta)
                sys.exit()

            try:
                g_0 = np.linalg.inv(w - epsilon_is)
            except np.linalg.LinAlgError:
                g_0 = np.linalg.pinv(w - epsilon_is)
            

            if np.isnan(g_0).any():
                print(f"Warning! Surface Greens function contains NaN values for w = {w_temp}. Check the force constants and the interaction range.")
                sys.exit()
        
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
        all_k_coupl_x =  mg.ranged_force_constant(k_coupl_x=self.k_coupl_x)["k_coupl_x"]
        all_k_coupl_xy =  mg.ranged_force_constant(k_coupl_xy=self.k_coupl_xy)["k_coupl_xy"]

        for i in range(0, direct_interaction.shape[0], 2):
            
            if 0 < i <= direct_interaction.shape[1]:
                direct_interaction[i, i - 2] = -all_k_coupl_x[0]
                
                if i - 2 + 3 <= direct_interaction.shape[1] - 1:
                    direct_interaction[i, i - 2 + 3] = -all_k_coupl_xy[0]
                    direct_interaction[i + 1, i - 2 + 2] = -all_k_coupl_xy[0]
                if i - 2 - 2 >= 0:
                    direct_interaction[i, i - 2 - 1] = -all_k_coupl_xy[0]
                    direct_interaction[i + 1, i - 2 - 2] = -all_k_coupl_xy[0]

            # xy coupling
            if i == 0:
                direct_interaction[i, i + 1] = -all_k_coupl_xy[0]
                direct_interaction[i + 1, i] = -all_k_coupl_xy[0]

            elif i == direct_interaction.shape[0] - 2:
                direct_interaction[i, direct_interaction.shape[1] - 1] = -all_k_coupl_xy[0]
                direct_interaction[i + 1, direct_interaction.shape[1] - 2] = -all_k_coupl_xy[0]

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
                            interaction_layer[j, j] += -all_k_coupl_xy[0]
                            interaction_layer[j + 1, j + 1] += -all_k_coupl_xy[0]
                    
                    elif ((N_y - N_y_scatter) // 2) < atomnr < ((N_y - N_y_scatter) // 2) + N_y_scatter + 1:
                        
                        interaction_layer[j, j] += -sum(all_k_coupl_x)
                        
            
                        if i == 0:

                            if (atomnr == ((N_y - N_y_scatter) // 2) + 1 or atomnr == ((N_y - N_y_scatter) // 2) + N_y_scatter) and N_y_scatter > 1:
                                interaction_layer[j, j] += -all_k_coupl_xy[0]
                                interaction_layer[j + 1, j + 1] += -all_k_coupl_xy[0]
                            elif N_y_scatter > 1:
                                interaction_layer[j, j] += -2 * all_k_coupl_xy[0]
                                interaction_layer[j + 1, j + 1] += -2 * all_k_coupl_xy[0]
            
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

class AnalyticalFourier(Electrode):

    ### Only for next nearest neighbour coupling at the moment!

    """
    Calculates the coupled surface greens function for  a 2D infinite square lattice electrode.

    Args:
        Electrode (_type_): _description_
    """
    def __init__(self, w, interaction_range, interact_potential, atom_type, lattice_constant, N_q,  k_el_x, k_el_y, k_el_xy, k_coupl_x, k_coupl_xy, N_y_scatter):
        super().__init__(w, interaction_range, interact_potential, atom_type, lattice_constant)
        self.q = np.linspace(-np.pi, np.pi, N_q)
        self.k_el_x = k_el_x
        self.k_el_y = k_el_y
        self.k_xy = k_el_xy
        self.k_coupl_x = k_coupl_x
        self.k_coupl_xy = k_coupl_xy
        self.N_y_scatter = N_y_scatter
        self.g0 = self.calculate_g0()
        self.g, self.k_lc_LL, self.dos, self.dos_real = self.calculate_g()

    def calculate_g0(self):
        """Calculates the uncoupled surface greens 2d infinite square lattice. 

        Returns:
            g0	(array_like) Surface greens function g0
        """

        def calc_g0_w(w, k_el_y, k_el_x, a):  
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

                y = k_el_y * np.sin(q / 2)**2
                g0_q = 2 * (w**2 - 4 * y + np.sqrt((w**2 - 4 * y) * (w**2 - 4 * k_el_x - 4 * y)))**(-1) 

                return g0_q
            
            g0_q_vals = np.array([g0_q(q_val, w) for q_val in self.q])
            
            g0 = (1 / (2 * np.pi)) * simpson(g0_q_vals, self.q)

            return g0
    

        # Build coupling/interaction matrix between electrode and scattering region
        all_k_x, all_k_y, all_k_xy = self.ranged_force_constant()[0:3]

        g0 = map(lambda w: calc_g0_w(w, self.k_el_y, self.k_el_x), self.w)
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
        all_k_coupl_x =  mg.ranged_force_constant(k_coupl_x=self.k_coupl_x)["k_coupl_x"]
        all_k_coupl_xy =  mg.ranged_force_constant(k_coupl_xy=self.k_coupl_xy)["k_coupl_xy"]
        g0_template = np.identity(direct_interaction.shape[0], dtype=float)


        # setting up the coupling
        for i in range(0, direct_interaction.shape[0], 2):
            
            if 0 < i <= direct_interaction.shape[1]:
                direct_interaction[i, i - 2] = -all_k_coupl_x[0]
                
                if i - 2 + 3 <= direct_interaction.shape[1] - 1:
                    direct_interaction[i, i - 2 + 3] = -all_k_coupl_xy[0]
                    direct_interaction[i + 1, i - 2 + 2] = -all_k_coupl_xy[0]
                if i - 2 - 2 >= 0:
                    direct_interaction[i, i - 2 - 1] = -all_k_coupl_xy[0]
                    direct_interaction[i + 1, i - 2 - 2] = -all_k_coupl_xy[0]

            if i == 0:
                direct_interaction[i, i + 1] = -all_k_coupl_xy[0]
                direct_interaction[i + 1, i] = -all_k_coupl_xy[0]

            elif i == direct_interaction.shape[0] - 2:
                direct_interaction[i, direct_interaction.shape[1] - 1] = -all_k_coupl_xy[0]
                direct_interaction[i + 1, direct_interaction.shape[1] - 2] = -all_k_coupl_xy[0]

        k_lc_LL = g0_template.copy()

        k_lc_LL[:direct_interaction.shape[0], :direct_interaction.shape[1]] = direct_interaction

        g = map(lambda x: np.dot(x * g0_template, np.linalg.inv(np.identity(g0_template.shape[0]) + np.dot(k_lc_LL, x * g0_template))), self.g0)
        g = np.array([item for item in g])
        
        dos = (-1 / np.pi) * np.imag(self.g0)
        dos_real = np.real(self.g0)

        return g, k_lc_LL, dos, dos_real

class DecimationFourier(Electrode):
    """
    Calculates the coupled surface greens function for  a 2D infinite square lattice electrode using the Sancho-Rubi method combined with Fourier transformation to take periodicity into account.

    Args:
        Electrode (_type_): _descriptison_
    """

    def __init__(self, w, interaction_range, interact_potential, atom_type, lattice_constant, left, right, N_y, N_y_scatter, M_L, M_C, k_el_x, k_el_y, k_el_xy, k_coupl_x, k_coupl_xy, N_q): 
        super().__init__(w, interaction_range, interact_potential, atom_type, lattice_constant, left, right)
        self.N_y = N_y
        self.N_y_scatter = N_y_scatter
        self.k_el_x = k_el_x 
        self.k_el_y = k_el_y 
        self.k_el_xy = k_el_xy 
        self.k_coupl_x = k_coupl_x
        self.k_coupl_xy = k_coupl_xy 
        self.M_L = M_L
        self.M_C = M_C
        self.eps = 1E-50
        self.N_q = N_q
        self.g0, self.H_01 = self.calculate_g0() #H_01 only needed if N_y == N_y_scatter
        self.g = self.calculate_g(self.g0, self.H_01)[0]
        self.dos, self.dos_real = self.calculate_g(self.g0, self.H_01)[3:5]
        self.dos_cpld, self.dos_real_cpld = self.calculate_g(self.g0, self.H_01)[5:]
        self.k_lc_LL, self.direct_interaction = self.calculate_g(self.g0, self.H_01)[1:3]

        assert self.N_y - self.N_y_scatter >= 0, "The number of atoms in the scattering region must be smaller than the number of atoms in the electrode. Please check your input parameters."
        assert (self.N_y - self.N_y_scatter) % 2 == 0, "The configuration must be symmetric in y-direction. Please check your input parameters."

    def calculate_g0(self):
        """Calculates surface greens 2d half infinite square lattice with finite width N_y. The uncoupled surface greens function g0 is calculated according to:
        "Highly convergent schemes for the calculation of bulk and surface Green functions", M P Lopez Sancho etal 1985 J.Phys.F:Met.Phys. 15 851
    
        Args:
            w (array_like): Frequency where g0 is calculated

        Returns:
            g0	(array_like) Surface greens function g0
        """

        def decimation(w, H_00, H_01, H_NN):
            """
            Decimation algorithm to calculate the surface greens function of a semi-infinite lattice.
            """
            
            w_temp = w
            H_01_dagger = np.conjugate(np.transpose(H_01))
            w = np.identity(H_NN.shape[0]) * (w**2 + (1.j * 1E-10)) 
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

                try:
                    g = np.linalg.inv(w - epsilon_i)
                except np.linalg.LinAlgError:
                    print(f"Matrix regularization was applied during the decimation algorithm for w = {w_temp}.")
                    g = np.nan_to_num((w - epsilon_i), nan=0.0, posinf=0.0, neginf=0.0)
                    g = np.linalg.inv(g + 1e-8 * np.eye(g.shape[0]))
                    
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
        
        k_values = mg.ranged_force_constant(k_el_x=self.k_el_x, k_el_y=self.k_el_y, k_el_xy=self.k_el_xy)
        
        H_NN = mg.build_H_NN(self.N_y, self.interaction_range, k_values)
        H_00 = mg.build_H_00(self.N_y, self.interaction_range, k_values)
        H_01 = mg.build_H_01(self.N_y, self.interaction_range, k_values)

        assert (0 <= np.abs(np.sum(H_00 + H_01)) < 1E-10), "Sum rule violated! H_00 + H_01 is not zero! Check the force constants and the interaction range."
        assert (0 <= np.abs(np.sum(H_NN + 2 * H_01)) < 1E-10), "Sum rule violated! H_NN + 2 * H_01 is not zero! Check the force constants and the interaction range."

        q_y_vals = np.linspace(-np.pi, np.pi, self.N_q)

        # Initialize array to store g_w matrices for all frequencies
        g_0_wk_list= []

        all_k_el_y = k_values["k_el_y"]
        all_k_el_xy = k_values["k_el_xy"]
    
        ### TRY STORE IN ZERO PADDED MATRIX
        for w in self.w:    ### --> MAP
            # Initialize arrays instead of dictionary
            g_0_k_list = []
            
            for q_y in q_y_vals:

                H_NN_k = H_NN.copy().astype(np.complex64)
                H_00_k = H_00.copy().astype(np.complex64)
                H_01_k = H_01.copy().astype(np.complex64)

                # all three matrices have the same shape, there for the loop is over the shape of just anyone of them --> next nearest neighbours for now!
                for i in range(H_00.shape[0]):
                    
                    if i % 2 == 0:
                        atomnr_i = np.ceil(float(i + 1) / 2)
                    
                        if (atomnr_i == 1 or atomnr_i == self.N_y) and i < H_00.shape[0] - 1:
            
                            if atomnr_i == 1:
                                # H_00
                                H_00_k[i, i] += all_k_el_xy[0]
                                H_00_k[i + 1, i + 1] += all_k_el_y[0] + all_k_el_xy[0]
                                
                                #H_NN
                                H_NN_k[i, i] += 2 * all_k_el_xy[0]
                                H_NN_k[i + 1, i + 1] += all_k_el_y[0] + 2 * all_k_el_xy[0]

                            else:
                                # H_00
                                H_00_k[i, i] += all_k_el_xy[0]
                                H_00_k[i + 1, i + 1] += all_k_el_y[0] + all_k_el_xy[0]
                                
                                #H_NN
                                H_NN_k[i, i] += 2 * all_k_el_xy[0]
                                H_NN_k[i + 1, i + 1] += all_k_el_y[0] + 2 * all_k_el_xy[0]


                    for j in range(H_00.shape[1]):

                        if j % 2 == 0:
                            atomnr_j = np.ceil(float(j + 1) / 2)

                            if (atomnr_i == 1 and i == 0) and atomnr_j == self.N_y and j < H_00.shape[0] - 1:         
                                #H_00 y HIER ALLE MINUSZEICHEN UMGEDREHT!!! (siehe Definition in Czycholl)
                                H_00_k[i + 1, j + 1] += -all_k_el_y[0] * np.exp(1j * q_y)         
                                H_00_k[j + 1, i + 1] += -all_k_el_y[0] * np.exp(-1j * q_y)

                                #H_NN
                                H_NN_k[i + 1, j + 1] += -all_k_el_y[0] * np.exp(1j * q_y)         
                                H_NN_k[j + 1, i + 1] += -all_k_el_y[0] * np.exp(-1j * q_y)

                                #H_01
                                H_01_k[i + 1, j + 1] += -all_k_el_xy[0] * np.exp(1j * q_y)     
                                H_01_k[i, j] += -all_k_el_xy[0] * np.exp(1j * q_y)    

                                H_01_k[j + 1, i + 1] += -all_k_el_xy[0] * np.exp(-1j * q_y)
                                H_01_k[j, i] += -all_k_el_xy[0] * np.exp(-1j * q_y)

                                break

                g_k = decimation(w, H_00_k, H_01_k, H_NN_k)

                g_0_k_list.append(g_k)

            # Convert list to numpy array
            g_0_k_array = np.array(g_0_k_list)

            g_0_wk_list.append(g_0_k_array)
            
        # Convert to numpy array with shape (N_w, matrix_dim, matrix_dim)
        g_0_wk_array = np.array(g_0_wk_list)

        return g_0_wk_array, H_01_k

    @staticmethod
    def _compute_g_single(g_0_wq, k_lc_LL):
        """Worker function für die Parallelisierung der g-Berechnung.
        
        Args:
            g_0_wq: Ein einzelnes g_0 Matrix element für spezifisches (w, q)
            k_lc_LL: Kopplungsmatrix (shared read-only)
            
        Returns:
            g: Berechnete g-Matrix für dieses (w, q) Element
        """
        x = g_0_wq
        g = np.dot(np.linalg.inv(np.identity(x.shape[0]) + np.dot(x, k_lc_LL)), x)
        return g

    def calculate_g(self, g_0):
    
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
        all_k_coupl_x = mg.ranged_force_constant(k_coupl_x=self.k_coupl_x)
        all_k_coupl_xy = mg.ranged_force_constant(k_coupl_xy=self.k_coupl_xy)

        for i in range(0, direct_interaction.shape[0], 2):
            
            if 0 < i <= direct_interaction.shape[1]:
                direct_interaction[i, i - 2] = -all_k_coupl_x[0]
                
                if i - 2 + 3 <= direct_interaction.shape[1] - 1:
                    direct_interaction[i, i - 2 + 3] = -all_k_coupl_xy[0]
                    direct_interaction[i + 1, i - 2 + 2] = -all_k_coupl_xy[0]
                if i - 2 - 2 >= 0:
                    direct_interaction[i, i - 2 - 1] = -all_k_coupl_xy[0]
                    direct_interaction[i + 1, i - 2 - 2] = -all_k_coupl_xy[0]

            # xy coupling
            if i == 0:
                direct_interaction[i, i + 1] = -all_k_coupl_xy[0]
                direct_interaction[i + 1, i] = -all_k_coupl_xy[0]

            elif i == direct_interaction.shape[0] - 2:
                direct_interaction[i, direct_interaction.shape[1] - 1] = -all_k_coupl_xy[0]
                direct_interaction[i + 1, direct_interaction.shape[1] - 2] = -all_k_coupl_xy[0]

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
                            interaction_layer[j, j] += -all_k_coupl_xy[0]
                            interaction_layer[j + 1, j + 1] += -all_k_coupl_xy[0]
                    
                    elif ((N_y - N_y_scatter) // 2) < atomnr < ((N_y - N_y_scatter) // 2) + N_y_scatter + 1:
                        
                        interaction_layer[j, j] += -sum(all_k_coupl_x)
                        
            
                        if i == 0:

                            if (atomnr == ((N_y - N_y_scatter) // 2) + 1 or atomnr == ((N_y - N_y_scatter) // 2) + N_y_scatter) and N_y_scatter > 1:
                                interaction_layer[j, j] += -all_k_coupl_xy[0]
                                interaction_layer[j + 1, j + 1] += -all_k_coupl_xy[0]
                            elif N_y_scatter > 1:
                                interaction_layer[j, j] += -2 * all_k_coupl_xy[0]
                                interaction_layer[j + 1, j + 1] += -2 * all_k_coupl_xy[0]
            
            interaction_layers_dict[i] = interaction_layer       
                        
                    
        for l in range(interaction_range):
            k_lc_LL[l * interaction_layers_dict[l].shape[0]: l * interaction_layers_dict[l].shape[0] + interaction_layers_dict[l].shape[0], \
                l * interaction_layers_dict[l].shape[0]: l * interaction_layers_dict[l].shape[0] + interaction_layers_dict[l].shape[0]] = interaction_layers_dict[interaction_range - 1 - l]
            

        q_y_vals = np.linspace(-np.pi, np.pi, self.N_q)
        
        # Optimierte Parallelisierung: Direkt über alle (w,q) Kombinationen
        # Verwende 'loky' backend für CPU-intensive Matrixoperationen (wegen Python GIL)
        # Erstelle alle (w_idx, q_idx) Kombinationen
        w_q_indices = [(w_idx, q_idx) for w_idx in range(g_0.shape[0]) for q_idx in range(self.N_q)]
        
        # Parallelisierte Berechnung aller g-Matrizen
        def compute_single_g(w_idx, q_idx):
            return self._compute_g_single(g_0[w_idx, q_idx], k_lc_LL)
        
        # Berechne alle g-Werte parallel
        g_results = Parallel(n_jobs=-1, backend='loky')(
            delayed(compute_single_g)(w_idx, q_idx) for w_idx, q_idx in w_q_indices
        )
        
        # Reshape results zurück in (w, N_q, g.shape[0], g.shape[1]) Form
        # g_results ist eine flache Liste mit len(w) * N_q Elementen
        if len(g_results) > 0:
            matrix_shape = g_results[0].shape  # (g.shape[0], g.shape[1])
            g_wq = np.array(g_results).reshape(g_0.shape[0], self.N_q, matrix_shape[0], matrix_shape[1])
        else:
            # Fallback falls keine Ergebnisse
            g_wq = np.array([]).reshape(0, self.N_q, 0, 0)
                

        dos_q = np.array([[1]])
        dos_real_q = np.array([[1]])
        
        dos_cpld_q = np.array([[1]])
        dos_real_cpld_q = np.array([[1]])

        return g_wq, k_lc_LL, direct_interaction, dos_q, dos_real_q, dos_cpld_q, dos_real_cpld_q

if __name__ == '__main__':

    N = 1000
    Nq = 2
    E_D = 80
    # convert to J
    E_D = E_D * constants.meV2J
    # convert to 1/s
    w_D = E_D / constants.h_bar
    # convert to har*s/(bohr**2*u)
    w_D = w_D / constants.unit2SI
    w = np.linspace(w_D * 1E-12, w_D * 1.1, N)
    k_c = 0.1 

    #electrode_debeye = DebeyeModel(w, k_c, w_D)
    #electrode_chain1d = Chain1D(w, interaction_range=2, interact_potential='reciproke_squared', atom_type="Au", lattice_constant=3.0, k_x=0.1, k_c=0.1)
    #electrode_2dribbon = Ribbon2D(w, interaction_range=3, interact_potential='reciproke_squared', atom_type="Au", lattice_constant=3.0, left=True, right=False, N_y=1, N_y_scatter=1, M_L=1, M_C=1, k_x=900, k_y=900, k_xy=180, k_c=900, k_c_xy=180)
    #electrode_2dSancho = DecimationFourier(w, interaction_range=1, interact_potential='reciproke_squared', atom_type="Au", lattice_constant=3.0, left=True, right=False, N_y=3, N_y_scatter=1, M_L=1, M_C=1, k_x=900, k_y=900, k_xy=0, k_c=900, k_c_xy=0, N_q=Nq)
    #electrode_infinite = AnalyticalFourier(w, interaction_range=1, interact_potential='reciproke_squared', atom_type="Au", lattice_constant=3.0, N_q=100, N_y_scatter=1, k_x=180, k_y=180, k_xy=0, k_c=180, k_c_xy=0)

    import time

    t1 = time.time()

    elec_deci = DecimationFourier(w, interaction_range=1, interact_potential='reciproke_squared', atom_type="Au", lattice_constant=3.0, left=True, right=False, N_y=3, N_y_scatter=1, M_L=1, M_C=1, k_x=900, k_y=900, k_xy=90, k_c=900, k_c_xy=90, N_q=Nq)

    t2 = time.time()

    print(t2-t1)

    print("debug")