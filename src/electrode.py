"""
Construct electrode models for the phononic transport calculation. Choose between

Debye model,
1D chain (analyitcal),
2D finite Ribbon,
Fourier transformed single molecule unit cell (for single junction),
Fourier transformed multi molecule unit cell (for periodic junction).

Multiply every force constant with (constants.eV2hartree / constants.ang2bohr ** 2) if needed

"""


import sys
import os
import numpy as np
import scipy
from joblib import Parallel, delayed
from scipy.integrate import simpson, quad
from utils import constants, matrix_gen as mg
from ase.dft.kpoints import monkhorst_pack


def decimation(w, H_00, H_01, H_NN, eps=1E-50, q_y=None, k_values=None, N_y=None) -> np.ndarray:
    """
    Decimation algorithm to calculate the surface greens function of a semi-infinite lattice.
    Based on: "Highly efficient schemes for the calculation of bulk and surface Green functions", 
    M P Lopez Sancho etal 1985 J.Phys.F:Met.Phys. 15 851
    DOI: 10.1088/0305-4608/15/4/009
    
    Args:
        w (float): Frequency value
        H_00 (np.ndarray): Hessian matrix of the surface principal layer
        H_01 (np.ndarray): Hessian matrix of the interaction between to principal layer
        H_NN (np.ndarray): Hessian matrix of a bulk principal layer
        eps (float): Convergence criterion (default: 1E-50)
        
    Returns:
        g0 (np.ndarray): Surface Green's function 

    """

    if all((q_y is not None, k_values is not None, N_y is not None)):

        H_NN = H_NN.astype(np.complex64)
        H_00 = H_00.astype(np.complex64)
        H_01 = H_01.astype(np.complex64)

        all_k_el_y = k_values["k_el_y"]
        all_k_el_xy = k_values["k_el_xy"]

        for i in range(H_00.shape[0]):
            
            if i % 2 == 0:
                atomnr_i = np.ceil(float(i + 1) / 2)
            
                if (atomnr_i == 1 or atomnr_i == N_y) and i < H_00.shape[0] - 1:
    
                    if atomnr_i == 1:
                        H_00[i, i] += all_k_el_xy[0]
                        H_00[i + 1, i + 1] += all_k_el_y[0] + all_k_el_xy[0]
                        H_NN[i, i] += 2 * all_k_el_xy[0]
                        H_NN[i + 1, i + 1] += all_k_el_y[0] + 2 * all_k_el_xy[0]

                    else:
                        H_00[i, i] += all_k_el_xy[0]
                        H_00[i + 1, i + 1] += all_k_el_y[0] + all_k_el_xy[0]
                        H_NN[i, i] += 2 * all_k_el_xy[0]
                        H_NN[i + 1, i + 1] += all_k_el_y[0] + 2 * all_k_el_xy[0]

            for j in range(H_00.shape[1]):

                if j % 2 == 0:
                    atomnr_j = np.ceil(float(j + 1) / 2)

                    if (atomnr_i == 1 and i == 0) and atomnr_j == N_y and j < H_00.shape[0] - 1:         
                        H_00[i + 1, j + 1] += -all_k_el_y[0] * np.exp(1j * q_y)         
                        H_00[j + 1, i + 1] += -all_k_el_y[0] * np.exp(-1j * q_y)
                        H_NN[i + 1, j + 1] += -all_k_el_y[0] * np.exp(1j * q_y)         
                        H_NN[j + 1, i + 1] += -all_k_el_y[0] * np.exp(-1j * q_y)
                        H_01[i + 1, j + 1] += -all_k_el_xy[0] * np.exp(1j * q_y)     
                        H_01[i, j] += -all_k_el_xy[0] * np.exp(1j * q_y)  
                        H_01[j + 1, i + 1] += -all_k_el_xy[0] * np.exp(-1j * q_y)
                        H_01[j, i] += -all_k_el_xy[0] * np.exp(-1j * q_y)

                        break

    w_temp = w                    
    w = np.identity(H_NN.shape[0]) * (w + (1.j * 1E-7))**2  # add small imaginary part to avoid singularities
    H_01_dagger = np.transpose(np.conj(H_01))
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
    
    while delta > eps:
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

    if delta >= eps or terminated:
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

class Electrode:
    """
    Motherclass for setting up different electrode models. By default, the left electrode is calculated.

    Args:
        w (np.ndarray): Frequency/energy points to calculate.
        interaction_range (int, float): Interaction range.
        interaction_potential (str): Interaction potential.
        atom_type (str): Atom type within the electrode.
        lattice_constant (int, float): Lattice constant.
        left, right (bool): Flag if its the left or the right electrode.

    Attributes:
        w (np.ndarray): Frequency/energy points to calculate.
        interaction_range (int, float): Interaction range.
        interaction_potential (str): Interaction potential.
        atom_type (str): Atom type within the electrode.
        lattice_constant (int, float): Lattice constant.
        left, right (bool): Flag if its the left or the right electrode.

    """

    def __init__(self, w, interaction_range=1, interact_potential="reciproke_squared", 
                 atom_type="Au", lattice_constant=3.0, left=True, right=False):
        self.w = w
        self.interaction_range = interaction_range
        self.interact_potential = interact_potential
        self.atom_type = atom_type
        self.lattice_constant = lattice_constant
        self.left = left
        self.right = right

class DebyeModel(Electrode): 
    """
    Set up the electrode via the Green's function description g0 and g according to the Debye model. Inherits from the Electrode class.
    Troels Markussen, Phonon interference effects in molecular junctions, J. Chem. Phys. 139, 244101 (2013).
    DOI: 10.1063/1.4849178

    Args:
        Electrode (object): Inherits arguments from Electrode class.
        k_coupl_x (float): Electrode-center coupling constant in x-direction.
        k_coupl_xy (float): Electrode-center coupling constant in xy-direction.
        w_D (float): Debye frequency

    Attributes:
        All attributes of Electrode motherclass.
        k_coupl_x (float): Electrode-center coupling constant in x-direction.
        k_coupl_xy (float): Electrode-center coupling constant in xy-direction.
        w_D (float): Debye frequency.
        g0 (np.ndarray): Uncoupled surface Green's function for each frequency.
        g (np.ndarray): Coupled surface Greens's function for each frequency
        dos (np.ndarray): Density of States (DOS) for each frequency.
        dos_real (np.ndarray): Real-part Re(DOS) for each frequency.
        dos_cpld (np.ndarray): DOS of the coupled system for each frequency.
        dos_real_cpld (np.ndarray): Real-part Re(DOS_cpld) of the coupled system for each frequency.

    """

    def __init__(self, w, k_coupl_x, k_coupl_xy, w_D):
        super().__init__(w)
        self.k_coupl_x = k_coupl_x
        self.k_coupl_xy = k_coupl_xy
        self.w_D = w_D
        self.g0 = self._calculate_g0(w, w_D)
        self.g, self.dos, self.dos_real, self.dos_cpld, self.dos_real_cpld = self._calculate_g()

    def _calculate_g0(self, w, w_D) -> np.ndarray:
        """Calculates the uncoupled surface Green's function.

        Args:
            w (np.ndarray): Frequencies where g0 is calculated
            w_D (float): Debye frequency
            k_c (float): Coupling constant to the center part
            interaction_range (int): Interaction range -> 1 = nearest neighbor, 2 = next nearest neighbor, etc.

        Returns:
            g0 (np.ndarray): Uncoupled surface Green's function.

        """

        def im_g(w):
            """Worker function for calculation the imaginary Green's function part."""

            if (w <= w_D):
                Im_g = -np.pi * 3.0 * w / (2 * w_D ** 3)
            else:
                Im_g = 0

            return Im_g

        Im_g = map(im_g, w)
        Im_g = np.asarray(list(Im_g))
        Re_g = -np.asarray(np.imag(scipy.signal.hilbert(Im_g)))
        g0 = np.asarray((Re_g + 1.j * Im_g), np.complex64)

        return g0

    def _calculate_g(self) -> tuple[np.ndarray]:
        """Calculates coupled surface Green's function.

        Returns:
            g (np.ndarray)) Coupled surface Green's function coupled by Dyson equation.
            dos (np.ndarray): Density of States (DOS) for each frequency.
            dos_real (np.ndarray): Real-part Re(DOS) for each frequency.
            dos_cpld (np.ndarray): DOS of the coupled system for each frequency.
            dos_real_cpld (np.ndarray): Real-part Re(DOS_cpld) of the coupled system for each frequency.

        """
        
        g = self.g0 / (1 + self.k_coupl_x * self.g0)
        dos = (-1 / np.pi) * np.imag(self.g0)
        dos_real = np.real(self.g0)
        
        dos_cpld = (-1 / np.pi) * np.imag(g)
        dos_real_cpld = np.real(g)

        return g, dos, dos_real, dos_cpld, dos_real_cpld

class Chain1D(Electrode):
    """
    Set up the electrode via the Green's function description g0 and g according to the analytical description of a 1D-chain. 
    Inherits from the Electrode class.
    Jan Klöckner, Dissertation: Heat transport through atomic and molecular contacts.

    Args:
        Electrode (object): Inherits arguments from Electrode class.
        k_coupl_x (float): Electrode-center coupling constant in x-direction.
        k_el_x (float): Coupling constant within the electrode in x-direction.

    Attributes:
        All attributes of Electrode motherclass.
        k_coupl_x (float): Electrode-center coupling constant in x-direction.
        k_coupl_xy (float): Electrode-center coupling constant in xy-direction.
        g0 (np.ndarray): Uncoupled surface Green's function for each frequency.
        g (np.ndarray): Coupled surface Greens's function for each frequency
        dos (np.ndarray): Density of States (DOS) for each frequency.
        dos_real (np.ndarray): Real-part Re(DOS) for each frequency.
        dos_cpld (np.ndarray): DOS of the coupled system for each frequency.
        dos_real_cpld (np.ndarray): Real-part Re(DOS_cpld) of the coupled system for each frequency.

    """

    def __init__(self, w, interaction_range, interact_potential, atom_type, 
                 lattice_constant, k_el_x, k_coupl_x):
        super().__init__(w, interaction_range, interact_potential, atom_type, lattice_constant)
        self.k_values = mg.ranged_force_constant(k_el_x=k_el_x, k_coupl_x=k_coupl_x, interaction_range=interaction_range)
        self.g0 = self._calculate_g0()
        self.g, self.dos, self.dos_real, self.dos_cpld, self.dos_real_cpld = self._calculate_g()

    def _calculate_g0(self) -> np.ndarray:
        """
        Calculates uncoupled surface Green's function of one-dimensional chain.

        Returns:
            g0 (np.ndarray): Uncoupled surface Green's function g0.

        """
        
        all_k_el_x = self.k_values["k_el_x"]
        k_x = sum(k_x for _ in all_k_el_x)

        #Jan PhD Thesis p.29
        g0 = 1 / (2 * k_x * self.w) * (self.w - np.sqrt(self.w**2 - 4 * k_x, dtype=np.complex64)) 

        return g0
    
    def _calculate_g(self) -> tuple[np.ndarray]:
        """
        Calculates coupled surface Green's function of one-dimensional chain via Dyson equation.

        Returns:
            g (np.ndarray)) Coupled surface Green's function coupled by Dyson equation.
            dos (np.ndarray): Density of States (DOS) for each frequency.
            dos_real (np.ndarray): Real-part Re(DOS) for each frequency.
            dos_cpld (np.ndarray): DOS of the coupled system for each frequency.
            dos_real_cpld (np.ndarray): Real-part Re(DOS_cpld) of the coupled system for each frequency.

        """

        all_k_coupl = self.k_values["k_coupl_x"]
        k_coupl = sum(k_c for k_c in all_k_coupl)
        
        g = self.g0 / (1 + k_coupl * self.g0)
        
        dos = (-1 / np.pi) * np.imag(self.g0)
        dos_real = np.real(self.g0)
        
        dos_cpld = (-1 / np.pi) * np.imag(g)
        dos_real_cpld = np.real(g)

        return g, dos, dos_real, dos_cpld, dos_real_cpld
    
class AnalyticalFourier(Electrode):

    ### Only for next nearest neighbour coupling at the moment!

    """
    Set up the electrode via the Green's function description g0 and g according to the analytical expression of a y-periodic electrode.
    Inherits from the Electrode class.
    Jan Klöckner, Dissertation: Heat transport through atomic and molecular contacts.

    Args:
        Electrode (object): Inherits arguments from Electrode class.
        k_coupl_x (float): Electrode-center coupling constant in x-direction.
        k_coupl_xy (float): Electrode-center coupling constant in xy-direction.
        k_el_x (float): Coupling constant within the electrode in x-direction.
        k_el_y (float): Coupling constant within the electrode in y-direction.
        k_el_xy (float): Coupling constant within the electrode in xy-direction.
        N_q (int): Number of q-points in reciprocal space.
        N_y_scatter (int): Number of atoms in y-direction of the central part.

    Attributes:
        All attributes of Electrode motherclass.
        k_coupl_x (float): Electrode-center coupling constant in x-direction.
        k_coupl_xy (float): Electrode-center coupling constant in xy-direction.
        k_el_x (float): Coupling constant within the electrode in x-direction.
        k_el_y (float): Coupling constant within the electrode in y-direction.
        k_el_xy (float): Coupling constant within the electrode in xy-direction.
        N_q (int): Number of q-points in reciprocal space.
        N_y_scatter (int): Number of atoms in y-direction of the central part.
        g0 (np.ndarray): Uncoupled surface Green's function for each frequency.
        g (np.ndarray): Coupled surface Greens's function for each frequency
        dos (np.ndarray): Density of States (DOS) for each frequency.
        dos_real (np.ndarray): Real-part Re(DOS) for each frequency.
        dos_cpld (np.ndarray): DOS of the coupled system for each frequency.
        dos_real_cpld (np.ndarray): Real-part Re(DOS_cpld) of the coupled system for each frequency.

    """

    def __init__(self, w, interaction_range, interact_potential, atom_type, lattice_constant, 
                 N_q, k_el_x, k_el_y, k_el_xy, k_coupl_x, k_coupl_xy, batch_size=100):
        super().__init__(w, interaction_range, interact_potential, atom_type, lattice_constant)
        self.q = np.linspace(-np.pi, np.pi, N_q, endpoint=False)
        self.batch_size = batch_size
        self.k_values = mg.ranged_force_constant(k_el_x=k_el_x, k_el_y=k_el_y, k_el_xy=k_el_xy, 
                                                 k_coupl_x=k_coupl_x, k_coupl_xy=k_coupl_xy, 
                                                 interaction_range=interaction_range)
        self.k_el_x = k_el_x
        self.k_el_y = k_el_y
        self.k_xy = k_el_xy
        self.k_coupl_x = k_coupl_x
        self.k_coupl_xy = k_coupl_xy
        self.g0 = self._calculate_g0()
        self.g, self.dos, self.dos_real, self.dos_cpld, self.dos_real_cpld = self._calculate_g()

    def _calculate_g0(self) -> np.ndarray:
        """
        Calculates the uncoupled surface Green's function for a 2D infinite square lattice. 

        Returns:
            g0 (np.ndarray): Uncoupled surface Green's function g0.

        """

        #def calc_g0_w(w, k_el_x, k_el_y):  
        """
                Calculates the surface greens function g0_q in reciprocal space for a 2D infinite square lattice electrode.

                Args:
                    q (float): Wave vector in reciprocal space (y-direction).
                    w (float): Frequency in reciprocal space.
                    k_y (float): Force constant in y direction.
                    k_x (float): Force constant in x direction.
                
                Returns:
                    g0_q (float): Surface Green's function g0_q in reciprocal space.
            \"\"\"
                

            def g0_q(q, w):

                w = w + (1j * 1E-24)

                y = k_el_y * np.sin(q / 2)**2
                g0_q = 2 * (w**2 - 4 * y + np.sqrt((w**2 - 4 * y) * (w**2 - 4 * k_el_x - 4 * y)))**(-1) 

                return g0_q
            
            g0_q_vals = np.array([g0_q(q_val, w) for q_val in self.q])
            g0 = (1 / (2 * np.pi)) * simpson(g0_q_vals, self.q)

            return g0
        
        def batch_worker(freqs):
            out = list()
            for w in freqs:
                out.append(calc_g0_w(w=w, k_el_x=self.k_el_x, k_el_y=self.k_el_y))
            return out"""


        def calc_g0_w(w): 
    
            # Integrate over q
            def g0_q_integrand(q, w_val, k_el_x_val, k_el_y_val):
                w_comp = w_val + (1j * 1E-24) 
                y = k_el_y_val * np.sin(q / 2)**2
                
                # Analytical expression
                g0_val = 2 * (w_comp**2 - 4 * y + np.sqrt((w_comp**2 - 4 * y) * (w_comp**2 - 4 * k_el_x_val - 4 * y)))**(-1) 

                return g0_val

            q_min = self.q[0]#-np.pi
            q_max = self.q[-1]#np.pi
        
            args = (w, self.k_el_x, self.k_el_y)
            
            # Complex integration neccessary for seperately real- and imaginary part!
            integral_real, err_real = quad(lambda q, *args: np.real(g0_q_integrand(q, *args)), 
                                        q_min, q_max, 
                                        args=args, limit=1000)
            
            integral_imag, err_imag = quad(lambda q, *args: np.imag(g0_q_integrand(q, *args)), 
                                        q_min, q_max, 
                                        args=args, limit=1000)
            
            g0_integral = integral_real + 1j * integral_imag
            g0 = (1 / (2 * np.pi)) * g0_integral
            
            return g0
        
        def batch_worker(freqs):
            out = list()
            for w in freqs:
                out.append(calc_g0_w(w=w))
            return out
        
        # batch the frequencies
        freq_batches = [self.w[i:i+self.batch_size] for i in range(0, len(self.w), self.batch_size)]

        g0_batches = Parallel(n_jobs=-1)(
            delayed(batch_worker)(batch) for batch in freq_batches
        )

        g0 = np.array([res for batch in g0_batches for res in batch])

        return g0

    def _calculate_g(self) -> tuple[np.ndarray]:
        """
        Calculates the surface Green's function in real space to represent an infinite 2D square lattice electrode. 

        Returns:
            g (np.ndarray): Coupled surface Greens's function for each frequency
            dos (np.ndarray): Density of States (DOS) for each frequency.
            dos_real (np.ndarray): Real-part Re(DOS) for each frequency.
            dos_cpld (np.ndarray): DOS of the coupled system for each frequency.
            dos_real_cpld (np.ndarray): Real-part Re(DOS_cpld) of the coupled system for each frequency.

        """

        all_k_coupl = self.k_values["k_coupl_x"]
        k_coupl = sum(k_c for k_c in all_k_coupl)
        
        g = self.g0 / (1 + k_coupl * self.g0)
        
        dos = (-1 / np.pi) * np.imag(self.g0)
        dos_real = np.real(self.g0)
        dos_cpld = (-1 / np.pi) * np.imag(g)
        dos_real_cpld = np.real(g)

        return g, dos, dos_real, dos_cpld, dos_real_cpld
    
class Ribbon2D(Electrode):
    """
    Set up the electrode via the Green's function description g0 and g for a finite 2D-Ribbon electrode.
    Inherits from the Electrode class.
    Based on: "Highly convergent schemes for the calculation of bulk and surface Green functions", 
    M P Lopez Sancho et al 1985 J.Phys.F:Met.Phys. 15 851,

    M. Bürkle, Thomas J. Hellmuth, F. Pauly, Y. Asai, First-principles calculation of the 
    thermoelectric figure of merit for [2,2]paracyclophane-based single-molecule junctions, 
    PHYSICAL REVIEW B 91, 165419 (2015)
    DOI: 10.1103/PhysRevB.91.165419

    Args:
        Electrode (object): Inherits arguments from Electrode class.
        k_coupl_x (float): Electrode-center coupling constant in x-direction.
        k_coupl_xy (float): Electrode-center coupling constant in xy-direction.
        k_el_x (float): Coupling constant within the electrode in x-direction.
        k_el_y (float): Coupling constant within the electrode in y-direction.
        k_el_xy (float): Coupling constant within the electrode in xy-direction.
        N_q (int): Number of q-points in reciprocal space.
        N_y_scatter (int): Number of atoms in y-direction of the central part.
        batch_size (int): batch-size for task parallelism.
        M_E, M_C (float): Mass of the electrode (E) and center (C) atoms. (Not needed for now)

    Attributes:
        All attributes of Electrode motherclass.
        k_coupl_x (float): Electrode-center coupling constant in x-direction.
        k_coupl_xy (float): Electrode-center coupling constant in xy-direction.
        k_el_x (float): Coupling constant within the electrode in x-direction.
        k_el_y (float): Coupling constant within the electrode in y-direction.
        k_el_xy (float): Coupling constant within the electrode in xy-direction.
        N_q (int): Number of q-points in reciprocal space.
        N_y (int): Number of atoms in y-direction of the electrode.
        N_y_scatter (int): Number of atoms in y-direction of the central part.
        g0 (np.ndarray): Uncoupled surface Green's function for each frequency.
        g (np.ndarray): Coupled surface Greens's function for each frequency
        center_coupling (np.ndarray): Matrix of the electrode-center coupling.
        direct_interaction (np.ndarray): Matrix of the direct electrode-center interaction.
        dos (np.ndarray): Density of States (DOS) for each frequency.
        dos_real (np.ndarray): Real-part Re(DOS) for each frequency.
        dos_cpld (np.ndarray): DOS of the coupled system for each frequency.
        dos_real_cpld (np.ndarray): Real-part Re(DOS_cpld) of the coupled system for each frequency.
        batch_size (int): batch-size for task parallelism.
        M_E, M_C (float): Mass of the electrode (E) and center (C) atoms. (Not needed for now)
        H_01 (np.ndarray): H_01 == k_LR_C if the electrode has the same width in y-direction as the center.

    """

    def __init__(self, w, interaction_range, interact_potential, atom_type, lattice_constant, left, right,
                 N_y, N_y_scatter, M_E, M_C, k_el_x, k_el_y, k_el_xy, k_coupl_x, k_coupl_xy, 
                 batch_size=100): 
        super().__init__(w, interaction_range, interact_potential, atom_type, lattice_constant, left, right)
        self.N_y = N_y
        self.N_y_scatter = N_y_scatter
        self.k_values = mg.ranged_force_constant(k_el_x=k_el_x, k_el_y=k_el_y, k_el_xy=k_el_xy,
                                                 k_coupl_x=k_coupl_x, k_coupl_xy=k_coupl_xy,
                                                 interaction_range=interaction_range)
        self.M_E = M_E
        self.M_C = M_C
        self.eps = 1E-50
        self.batch_size = batch_size
        self.g0, self.H_01 = self._calculate_g0()  # H_01 only needed if N_y == N_y_scatter
        self.g, self.center_coupling, self.direct_interaction, \
        self.dos, self.dos_real, self.dos_cpld, self.dos_real_cpld = self._calculate_g()

        assert self.N_y - self.N_y_scatter >= 0, (
            "The number of atoms in the scattering region must be smaller than the number of atoms in the electrode. "
             "Please check your input parameters."
        )
        assert (self.N_y - self.N_y_scatter) % 2 == 0, (
            "The configuration must be symmetric in y-direction. Please check your input parameters."
        )

    def _calculate_g0(self) -> tuple[np.ndarray]:
        """
        Calculates uncoupled surface Green's functions using the decimation method with batched parallelization.
        
        Returns:
            g0 (np.ndarray): Uncoupled surface Green's function for each frequency.
            H_01 (np.ndarray): Coupling array from the decimation technique.

        """

        H_NN = mg.build_H_NN(self.N_y, self.interaction_range, k_values=self.k_values)
        H_00 = mg.build_H_00(self.N_y, self.interaction_range, k_values=self.k_values)
        H_01 = mg.build_H_01(self.N_y, self.interaction_range, k_values=self.k_values)

        assert (0 <= np.abs(np.sum(H_00 + H_01)) < 1E-10), (
            "Sum rule violated! H_00 + H_01 is not zero! Check the force constants and the interaction range."
        )
        assert (0 <= np.abs(np.sum(H_NN + 2 * H_01)) < 1E-10), (
            "Sum rule violated! H_NN + 2 * H_01 is not zero! Check the force constants and the interaction range."
        )

        def batch_worker(w_data):
            """Worker function for parallelized decimation"""
            results = []
            for w_idx, w in w_data:
                matrix = decimation(w=w, H_00=H_00, H_01=H_01, H_NN=H_NN, eps=self.eps)
                results.append((w_idx, matrix))
            return results

        # Create w data with indices
        w_data = [(w_idx, w) for w_idx, w in enumerate(self.w)]
        
        # Batch the frequencies
        batch_size = self.batch_size
        batches = [w_data[i:i+batch_size] for i in range(0, len(w_data), batch_size)]

        # Parallel processing
        batch_results = Parallel(n_jobs=-1)(
            delayed(batch_worker)(batch) for batch in batches
        )
        
        # Initialize result array
        matrix_shape = H_00.shape
        g0 = np.zeros((len(self.w), matrix_shape[0], matrix_shape[1]), dtype=np.complex64)
        
        # Fill the indexed results
        for batch in batch_results:
            for w_idx, matrix in batch:
                g0[w_idx] = matrix

        return g0, H_01

    def _calculate_g(self) -> tuple[np.ndarray]: 
                                    
        """
        Calculates coupled surface Green's functions using Dyson equation with batched parallelization.
        
        Returns:
            g (np.ndarray): Coupled surface Green's function.
            center_coupling (np.ndarray): Matrix of the electrode-center coupling.
            direct_interaction (np.ndarray): Matrix of the direct electrode-center interaction.
            dos (np.ndarray): Density of States (DOS) for each frequency.
            dos_real (np.ndarray): Real-part Re(DOS) for each frequency.
            dos_cpld (np.ndarray): DOS of the coupled system for each frequency.
            dos_real_cpld (np.ndarray): Real-part Re(DOS_cpld) of the coupled system for each frequency.

        """

        # Build coupling/interaction matrix between electrode and scattering region
        direct_interaction = mg.calc_direct_interaction(N_y_scatter=self.N_y_scatter,
                                                        k_values=self.k_values)
        center_coupling = mg.calc_center_coupl(N_y=self.N_y,
                                               N_y_scatter=self.N_y_scatter,
                                               interaction_range=self.interaction_range,
                                               k_values=self.k_values)

        def batch_worker(g0_data):
            """Worker function for parallelized g calculation"""
            results = []
            for w_idx, g0_matrix in g0_data:
                A = np.identity(g0_matrix.shape[0]) + np.dot(g0_matrix, center_coupling)
                g_matrix = np.dot(np.linalg.inv(A), g0_matrix)
                results.append((w_idx, g_matrix))
            return results

        # Create batched data with indices
        g0_data = [(w_idx, self.g0[w_idx]) for w_idx in range(len(self.w))]
        
        # Batch the data
        batch_size = self.batch_size
        batches = [g0_data[i:i+batch_size] for i in range(0, len(g0_data), batch_size)]
        
        # Parallel processing
        batch_results = Parallel(n_jobs=-1)(
            delayed(batch_worker)(batch) for batch in batches
        )
        
        # Initialize result array
        g = np.zeros_like(self.g0, dtype=np.complex64)
        
        # Fill the indexed results
        for batch in batch_results:
            for w_idx, g_matrix in batch:
                g[w_idx] = g_matrix

        dos = (-1 / np.pi) * np.imag(np.trace(self.g0, axis1=1, axis2=2))
        dos_real = np.real(np.trace(self.g0, axis1=1, axis2=2))
        dos_cpld = (-1 / np.pi) * np.imag(np.trace(g, axis1=1, axis2=2))
        dos_real_cpld = np.real(np.trace(g, axis1=1, axis2=2))

        return g, center_coupling, direct_interaction, dos, dos_real, dos_cpld, dos_real_cpld

class DecimationFourier(Electrode):
    """
    Set up the electrode via the Green's function description g0 and g for a infinite 2D-Ribbon electrode.
    Inherits from the Electrode class. Using Fourier transformation to take y-periodicity into account.

    Args:
        Electrode (object): Inherits arguments from Electrode class.
        k_coupl_x (float): Electrode-center coupling constant in x-direction.
        k_coupl_xy (float): Electrode-center coupling constant in xy-direction.
        k_el_x (float): Coupling constant within the electrode in x-direction.
        k_el_y (float): Coupling constant within the electrode in y-direction.
        k_el_xy (float): Coupling constant within the electrode in xy-direction.
        N_q (int): Number of q-points in reciprocal space.
        N_y (int): Number of atoms in y-direction of the electrode.
        N_y_scatter (int): Number of atoms in y-direction of the central part.
        batch_size (int): batch-size for task parallelism.
        N_q (int): Number of q-points for the Fourier transformation
        M_E, M_C (float): Mass of the electrode (E) and center (C) atoms. (Not needed for now)

    Attributes:
        All attributes of Electrode motherclass.
        k_coupl_x (float): Electrode-center coupling constant in x-direction.
        k_coupl_xy (float): Electrode-center coupling constant in xy-direction.
        k_el_x (float): Coupling constant within the electrode in x-direction.
        k_el_y (float): Coupling constant within the electrode in y-direction.
        k_el_xy (float): Coupling constant within the electrode in xy-direction.
        N_q (int): Number of q-points in reciprocal space.
        N_y (int): Number of atoms in y-direction of the electrode.
        N_y_scatter (int): Number of atoms in y-direction of the central part.
        g0 (np.ndarray): Uncoupled surface Green's function for each frequency.
        g (np.ndarray): Coupled surface Greens's function for each frequency
        center_coupling (np.ndarray): Matrix of the electrode-center coupling.
        direct_interaction (np.ndarray): Matrix of the direct electrode-center interaction.
        dos (np.ndarray): Density of States (DOS) for each frequency.
        dos_real (np.ndarray): Real-part Re(DOS) for each frequency.
        dos_cpld (np.ndarray): DOS of the coupled system for each frequency.
        dos_real_cpld (np.ndarray): Real-part Re(DOS_cpld) of the coupled system for each frequency.
        batch_size (int): batch-size for task parallelism.
        M_E, M_C (float): Mass of the electrode (E) and center (C) atoms. (Not needed for now)
        H_01 (np.ndarray): H_01 == k_LR_C if the electrode has the same width in y-direction as the center.

    """
    
    # Density of states (DOS) does'nt really makes sense here. Maybe get rid of it.

    def __init__(self, w, interaction_range, interact_potential, atom_type, lattice_constant, left, right, 
                 N_y, N_y_scatter, M_E, M_C, k_el_x, k_el_y, k_el_xy, k_coupl_x, k_coupl_xy, N_q, batch_size=100): 
        
        super().__init__(w, interaction_range, interact_potential, atom_type, lattice_constant, left, right)
        #self.q_y = np.linspace(-np.pi, np.pi, N_q, endpoint=False)
        self.q_y = np.linspace(-np.pi, np.pi, endpoint=False)
        self.batch_size = batch_size
        self.N_y = N_y
        self.N_y_scatter = N_y_scatter
        self.k_values = mg.ranged_force_constant(k_el_x=k_el_x, k_el_y=k_el_y, k_el_xy=k_el_xy, k_coupl_x=k_coupl_x, k_coupl_xy=k_coupl_xy, 
                                                 interaction_range=interaction_range)
        self.M_E = M_E
        self.M_C = M_C
        self.eps = 1E-50
        self.g0, self.H_01 = self._calculate_g0() #H_01 only needed if N_y == N_y_scatter
        self.g, self.center_coupling, self.direct_interaction, self.dos, self.dos_real, self.dos_cpld, self.dos_real_cpld = self._calculate_g()
        
        assert self.N_y - self.N_y_scatter >= 0, (
            "The number of atoms in the scattering region must be smaller than the number of atoms in the electrode. Please check your input parameters."
        )
        assert (self.N_y - self.N_y_scatter) % 2 == 0, (
            "The configuration must be symmetric in y-direction. Please check your input parameters."
        )

    def _calculate_g0(self) -> tuple[np.ndarray]:
        """
        Calculates uncoupled surface Green's functions using the decimation method and 
        Fourier transformation with batched parallelization.
        
        Returns:
            g0 (np.ndarray): Uncoupled surface Green's function for each frequency and q-point.
                            Shape: (N_w, N_q, matrix_dim, matrix_dim)
            H_01 (np.ndarray): Coupling array from the decimation technique.

        """
    
        H_00 = mg.build_H_00(self.N_y, self.interaction_range, self.k_values)
        H_01 = mg.build_H_01(self.N_y, self.interaction_range, self.k_values)
        H_NN = mg.build_H_NN(self.N_y, self.interaction_range, self.k_values)

        assert (0 <= np.abs(np.sum(H_00 + H_01)) < 1E-10), (
            "Sum rule violated! H_00 + H_01 is not zero! Check the force constants and the interaction range."
        )
        assert (0 <= np.abs(np.sum(H_NN + 2 * H_01)) < 1E-10), (
            "Sum rule violated! H_NN + 2 * H_01 is not zero! Check the force constants and the interaction range."
        )

        def batch_worker(w_q_pairs):
            """Worker function for parallelized decimation"""
            results = []
            for w_idx, q_idx, w, q in w_q_pairs:
                matrix = decimation(w=w, H_00=H_00, H_01=H_01, H_NN=H_NN, eps=self.eps, 
                                q_y=q, k_values=self.k_values, N_y=self.N_y)
                results.append((w_idx, q_idx, matrix))
            return results
    
        # Create all (w,q) combinations with their indices
        w_q_combinations = []
        for w_idx, w in enumerate(self.w):
            for q_idx, q in enumerate(self.q_y):
                w_q_combinations.append((w_idx, q_idx, w, q))
        
        # Batch the combinations
        batch_size = self.batch_size
        batches = [w_q_combinations[i:i+batch_size] for i in range(0, len(w_q_combinations), batch_size)]

        # Parallel processing
        batch_results = Parallel(n_jobs=-1)(
            delayed(batch_worker)(batch) for batch in batches
        )
        
        # Initialize result array
        matrix_shape = H_00.shape  # Get matrix dimensions
        g0 = np.zeros((len(self.w), len(self.q_y), matrix_shape[0], matrix_shape[1]), dtype=np.complex64)
        
        # Fill the indexed results
        for batch in batch_results:
            for w_idx, q_idx, matrix in batch:
                g0[w_idx, q_idx] = matrix

        return g0, H_01

    def _calculate_g(self) -> tuple[np.ndarray]:
        """
        Calculates coupled surface Green's of 2D half infinite square lattice. Taking into account the interaction range.

        Returns:
            g (np.ndarray): Coupled surface Green's function. Shape: (N_w, N_q, matrix_dim, matrix_dim)
            center_coupling (np.ndarray): Matrix of the electrode-center coupling.
            direct_interaction (np.ndarray): Matrix of the direct electrode-center interaction.
            dos (np.ndarray): Density of States (DOS) for each frequency.
            dos_real (np.ndarray): Real-part Re(DOS) for each frequency.
            dos_cpld (np.ndarray): DOS of the coupled system for each frequency.
            dos_real_cpld (np.ndarray): Real-part Re(DOS_cpld) of the coupled system for each frequency.

        """

        # Build coupling/interaction matrix between electrode and scattering region
        direct_interaction = mg.calc_direct_interaction(N_y_scatter=self.N_y_scatter, k_values=self.k_values)
        center_coupling = mg.calc_center_coupl(N_y=self.N_y, N_y_scatter=self.N_y_scatter, 
                                            interaction_range=self.interaction_range, k_values=self.k_values)

        def batch_worker(g0_data):
            """Worker function for parallelized g calculation"""
            results = []
            for w_idx, q_idx, g0_matrix in g0_data:
                A = np.identity(g0_matrix.shape[0]) + np.dot(g0_matrix, center_coupling)
                g_matrix = np.dot(np.linalg.inv(A), g0_matrix)
                
                results.append((w_idx, q_idx, g_matrix))

            return results
        
        # Create batched data with indices
        g0_data = []
        for w_idx in range(self.g0.shape[0]):
            for q_idx in range(self.g0.shape[1]):
                g0_data.append((w_idx, q_idx, self.g0[w_idx, q_idx]))
        
        # Batch the data
        batch_size = self.batch_size
        batches = [g0_data[i:i+batch_size] for i in range(0, len(g0_data), batch_size)]
        
        # Parallel processing
        batch_results = Parallel(n_jobs=-1)(
            delayed(batch_worker)(batch) for batch in batches
        )
        
        # Initialize result array
        g = np.zeros_like(self.g0, dtype=np.complex64)
        
        # Fill the indexed results
        for batch in batch_results:
            for w_idx, q_idx, g_matrix in batch:
                g[w_idx, q_idx] = g_matrix

        # Calculate DOS (averaged over q-points)
        dos_q = (-1 / np.pi) * np.imag(np.trace(self.g0, axis1=2, axis2=3))
        dos_real_q = np.real(np.trace(self.g0, axis1=2, axis2=3))
        dos_cpld_q = (-1 / np.pi) * np.imag(np.trace(g, axis1=2, axis2=3))
        dos_real_cpld_q = np.real(np.trace(g, axis1=2, axis2=3))
        
        # Average over q-points for final DOS
        dos = np.mean(dos_q, axis=1)
        dos_real = np.mean(dos_real_q, axis=1)
        dos_cpld = np.mean(dos_cpld_q, axis=1)
        dos_real_cpld = np.mean(dos_real_cpld_q, axis=1)


        return g, center_coupling, direct_interaction, dos, dos_real, dos_cpld, dos_real_cpld


if __name__ == '__main__':

    N = 10000
    Nq = 50
    E_D = 80
    # convert to J
    E_D = E_D * constants.meV2J
    # convert to 1/s
    w_D = E_D / constants.h_bar
    # convert to har*s/(bohr**2*u)
    w_D = w_D / constants.unit2SI
    w = np.linspace(w_D * 1E-12, w_D * 1.1, N)
    k_c = 0.1 
    batch_size = max(1, int(N / os.cpu_count()))
    #batch_size = 100


    #electrode_Debye = DebyeModel(w, k_c, w_D)
    #electrode_chain1d = Chain1D(w, interaction_range=2, interact_potential='reciproke_squared', atom_type="Au", lattice_constant=3.0, k_x=0.1, k_c=0.1)
    #electrode_2dribbon = Ribbon2D(w, interaction_range=3, interact_potential='reciproke_squared', atom_type="Au", lattice_constant=3.0, left=True, right=False, N_y=1, N_y_scatter=1, M_E=1, M_C=1, k_x=900, k_y=900, k_xy=180, k_c=900, k_c_xy=180)
    #electrode_2dSancho = DecimationFourier(w, interaction_range=1, interact_potential='reciproke_squared', atom_type="Au", lattice_constant=3.0, left=True, right=False, N_y=3, N_y_scatter=1, M_E=1, M_C=1, k_x=900, k_y=900, k_xy=0, k_c=900, k_c_xy=0, N_q=Nq)
    #electrode_infinite = AnalyticalFourier(w, interaction_range=1, interact_potential='reciproke_squared', atom_type="Au", lattice_constant=3.0, N_q=100, N_y_scatter=1, k_x=180, k_y=180, k_xy=0, k_c=180, k_c_xy=0)

    import time

    t1 = time.time()
    print(batch_size)
    #electrode_2dribbon = Ribbon2D(w, interaction_range=1, interact_potential='reciproke_squared', atom_type="Au", lattice_constant=3.0, left=True, right=False, N_y=3, N_y_scatter=1, M_E=1, M_C=1, 
    #                               k_el_x=900, k_el_y=900, k_el_xy=10, k_coupl_x=900, k_coupl_xy=10, batch_size=batch_size)
    electrode_infinite = AnalyticalFourier(w, interaction_range=1, interact_potential='reciproke_squared', atom_type="Au", lattice_constant=3.0, N_q=Nq, 
                                            N_y_scatter=1, k_el_x=180, k_el_y=180, k_el_xy=0, k_coupl_x=180, k_coupl_xy=0, batch_size=batch_size)
    #electrode_2dSancho = DecimationFourier(w, interaction_range=1, interact_potential='reciproke_squared', atom_type="Au", lattice_constant=3.0, left=True, right=False, N_y=3, 
                                           #N_y_scatter=1, M_E=1, M_C=1, k_el_x=900, k_el_y=900, k_el_xy=90, k_coupl_x=900, k_coupl_xy=90, N_q=Nq, batch_size=batch_size)
    t2 = time.time()

    print(t2-t1)

    print("debug")