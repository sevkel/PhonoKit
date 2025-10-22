"""
Sigma Calculation Service

Service module for calculating self-energies for different electrode configurations.

Author: Severin Keller
Date: 2025
"""

import numpy as np
from joblib import Parallel, delayed
from utils import matrix_gen as mg


class SigmaCalculator:
    """
    Service class for calculating self-energies (sigma) for different electrode types.
    
    Supports:
    - DebyeModel electrodes
    - Chain1D electrodes
    - Ribbon2D electrodes
    - AnalyticalFourier electrodes
    - DecimationFourier electrodes
    """
    
    def __init__(self, electrode_L, electrode_R, scatter, electrode_dict_L, electrode_dict_R, 
                 w, N, batch_size):
        """
        Initialize the sigma calculator.
        
        Args:
            electrode_L: Left electrode object
            electrode_R: Right electrode object
            scatter: Scattering object
            electrode_dict_L (dict): Left electrode configuration
            electrode_dict_R (dict): Right electrode configuration
            w (np.ndarray): Frequency array
            N (int): Number of frequency points
            batch_size (int): Batch size for parallel processing
        """
        self.electrode_L = electrode_L
        self.electrode_R = electrode_R
        self.scatter = scatter
        self.electrode_dict_L = electrode_dict_L
        self.electrode_dict_R = electrode_dict_R
        self.w = w
        self.N = N
        self.batch_size = batch_size
        self.i = np.arange(0, N, dtype=int)
    
    def calculate(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Calculate self-energies based on electrode types.
        
        Returns:
            tuple: (sigma_L, sigma_R) - Self-energies for left and right electrodes
        """
        electrode_type_L = self.electrode_dict_L["type"]
        electrode_type_R = self.electrode_dict_R["type"]
        
        if (electrode_type_L, electrode_type_R) == ("Chain1D", "Chain1D"):
            return self._calculate_chain1d()
        elif (electrode_type_L, electrode_type_R) in [("DebyeModel", "DebyeModel"), 
                                                        ("AnalyticalFourier", "AnalyticalFourier")]:
            return self._calculate_debye_analytical()
        elif (electrode_type_L, electrode_type_R) == ("Ribbon2D", "Ribbon2D"):
            return self._calculate_ribbon2d()
        elif (electrode_type_L, electrode_type_R) == ("DecimationFourier", "DecimationFourier"):
            return self._calculate_decimation_fourier()
        else:
            raise ValueError(f"Unsupported electrode combination: {electrode_type_L}, {electrode_type_R}")
    
    def _calculate_chain1d(self) -> tuple[np.ndarray, np.ndarray]:
        """Calculate sigma for Chain1D electrodes."""
        k_coupl_x = sum(mg.ranged_force_constant(
            k_coupl_x=self.electrode_dict_L["k_coupl_x"])["k_coupl_x"])
        
        f_E = 0.5 * (self.w**2 - 2 * k_coupl_x - 
                     self.w * np.sqrt(self.w**2 - 4 * k_coupl_x, dtype=np.complex64))
        
        sigma_L = np.zeros((self.N, self.scatter.N, self.scatter.N), dtype=np.complex64)
        sigma_R = np.zeros((self.N, self.scatter.N, self.scatter.N), dtype=np.complex64)
        
        for i in range(self.N):
            sigma_L[i, 0, 0] = f_E[i]
            sigma_R[i, -1, -1] = f_E[i]
        
        return sigma_L, sigma_R
    
    def _calculate_debye_analytical(self) -> tuple[np.ndarray, np.ndarray]:
        """Calculate sigma for scalar Green's function electrodes (DebyeModel, AnalyticalFourier)."""
        g_L = self.electrode_L.g
        g_R = self.electrode_R.g
        
        k_coupl_l = self.electrode_L.k_coupl_x
        k_coupl_r = self.electrode_R.k_coupl_x
        
        sigma_nu_l = np.array([k_coupl_l**2 * g_L[i] for i in self.i])
        sigma_nu_r = np.array([k_coupl_r**2 * g_R[i] for i in self.i])
        
        hessian_shape = self.scatter.hessian.shape
        sigma_L = np.zeros((self.N, hessian_shape[0], hessian_shape[1]), dtype=np.complex64)
        sigma_R = np.zeros((self.N, hessian_shape[0], hessian_shape[1]), dtype=np.complex64)
        
        if self.scatter.type == "FiniteLattice2D":
            for k in range(self.N):
                for n in range(self.scatter.N_y):
                    sigma_L[k][n * 2, n * 2] = sigma_nu_l[k]
                    sigma_R[k][sigma_R.shape[1] - 2 - n * 2, 
                             sigma_R.shape[2] - 2 - n * 2] = sigma_nu_r[k]
        elif self.scatter.type == "Chain1D":
            for k in range(self.N):
                sigma_L[k][0, 0] = sigma_nu_l[k]
                sigma_R[k][-1, -1] = sigma_nu_r[k]
        
        return sigma_L, sigma_R
    
    def _calculate_ribbon2d(self) -> tuple[np.ndarray, np.ndarray]:
        """Calculate sigma for Ribbon2D electrodes."""
        g_L = self.electrode_L.g
        g_R = self.electrode_R.g
        
        k_LC, k_RC = self._build_coupling_matrices()
        
        sigma_L = np.array([np.dot(np.dot(k_LC.T, g_L[i]), k_LC) for i in self.i])
        sigma_R = np.array([np.dot(np.dot(k_RC.T, g_R[i]), k_RC) for i in self.i])
        
        return sigma_L, sigma_R
    
    def _calculate_decimation_fourier(self) -> tuple[np.ndarray, np.ndarray]:
        """Calculate sigma for DecimationFourier electrodes with parallelization."""
        g_L = self.electrode_L.g
        g_R = self.electrode_R.g
        
        k_LC, k_RC = self._build_coupling_matrices()
        
        N_q = len(self.electrode_L.q_y)
        D_shape = self.scatter.hessian.shape
        
        sigma_L = np.zeros((self.N, N_q, D_shape[0], D_shape[1]), dtype=np.complex64)
        sigma_R = np.zeros((self.N, N_q, D_shape[0], D_shape[1]), dtype=np.complex64)
        
        def sigma_worker(w_q_data):
            """Worker function for parallelized sigma computation."""
            results_L = []
            results_R = []
            
            for w_idx, q_idx, g_matrix in w_q_data:
                sigma_L_wq = np.dot(np.dot(k_LC.T, g_matrix), k_LC)
                sigma_R_wq = np.dot(np.dot(k_RC.T, g_matrix), k_RC)
                results_L.append((w_idx, q_idx, sigma_L_wq))
                results_R.append((w_idx, q_idx, sigma_R_wq))
            
            return results_L, results_R
        
        # Create indexed data
        w_q_data = []
        for w_idx in range(g_L.shape[0]):
            for q_idx in range(g_L.shape[1]):
                w_q_data.append((w_idx, q_idx, g_L[w_idx, q_idx]))
        
        # Batch and parallelize
        batches = [w_q_data[i:i+self.batch_size] 
                  for i in range(0, len(w_q_data), self.batch_size)]
        
        batch_results = Parallel(n_jobs=-1)(
            delayed(sigma_worker)(batch) for batch in batches
        )
        
        # Fill results
        for batch in batch_results:
            for (w_idx, q_idx, sigma_L_matrix), (_, _, sigma_R_matrix) in zip(batch[0], batch[1]):
                sigma_L[w_idx, q_idx] = sigma_L_matrix
                sigma_R[w_idx, q_idx] = sigma_R_matrix
        
        return sigma_L, sigma_R
    
    def _build_coupling_matrices(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Build coupling matrices for Ribbon2D and DecimationFourier electrodes.
        
        Returns:
            tuple: (k_LC, k_RC) - Coupling matrices for left and right electrodes
        """
        N_y_L = self.electrode_L.N_y
        N_y_R = self.electrode_R.N_y
        N_y_scatter = self.scatter.N_y
        
        interaction_range_L = self.electrode_L.interaction_range
        interaction_range_R = self.electrode_R.interaction_range
        
        # Initialize coupling matrices
        k_LC = np.zeros((2 * interaction_range_L * N_y_L, 
                        2 * self.scatter.N_x * N_y_scatter), dtype=float)
        k_RC = np.zeros((2 * interaction_range_R * N_y_R, 
                        2 * self.scatter.N_x * N_y_scatter), dtype=float)
        
        k_LC_temp = np.zeros((2 * interaction_range_L * N_y_L, 
                             2 * interaction_range_L * N_y_scatter), dtype=float)
        k_RC_temp = np.zeros((2 * interaction_range_R * N_y_R, 
                             2 * interaction_range_R * N_y_scatter), dtype=float)
        
        # Get force constants
        all_k_coupl_x_L = mg.ranged_force_constant(
            k_coupl_x=self.electrode_dict_L["k_coupl_x"], 
            interaction_range=interaction_range_L)["k_coupl_x"]
        all_k_coupl_x_R = mg.ranged_force_constant(
            k_coupl_x=self.electrode_dict_R["k_coupl_x"], 
            interaction_range=interaction_range_R)["k_coupl_x"]
        all_k_coupl_xy_L = mg.ranged_force_constant(
            k_coupl_xy=self.electrode_dict_L["k_coupl_xy"], 
            interaction_range=interaction_range_L)["k_coupl_xy"]
        all_k_coupl_xy_R = mg.ranged_force_constant(
            k_coupl_xy=self.electrode_dict_R["k_coupl_xy"], 
            interaction_range=interaction_range_R)["k_coupl_xy"]
        
        # Build left coupling matrix
        k_LC_temp = self._build_single_coupling_matrix(
            N_y_L, N_y_scatter, interaction_range_L, 
            all_k_coupl_x_L, all_k_coupl_xy_L)
        
        # Build right coupling matrix
        k_RC_temp = self._build_single_coupling_matrix(
            N_y_R, N_y_scatter, interaction_range_R, 
            all_k_coupl_x_R, all_k_coupl_xy_R)
        
        # Handle interaction range > 1 for right electrode
        if interaction_range_R > 1:
            mid_col_R = k_RC_temp.shape[1] // 2
            k_RC_temp = np.hstack([k_RC_temp[:, mid_col_R:], k_RC_temp[:, :mid_col_R]])
        
        # Fill final matrices
        k_LC[:k_LC_temp.shape[0], :k_LC_temp.shape[1]] = k_LC_temp
        k_RC[-k_RC_temp.shape[0]:, -k_RC_temp.shape[1]:] = k_RC_temp
        
        return k_LC, k_RC
    
    def _build_single_coupling_matrix(self, N_y_el, N_y_scatter, interaction_range, 
                                     all_k_coupl_x, all_k_coupl_xy) -> np.ndarray:
        """
        Build a single coupling matrix for one electrode.
        
        Args:
            N_y_el (int): Number of atoms in y-direction for electrode
            N_y_scatter (int): Number of atoms in y-direction for scatter
            interaction_range (int): Interaction range
            all_k_coupl_x (list): Force constants in x-direction
            all_k_coupl_xy (list): Force constants in xy-direction
            
        Returns:
            np.ndarray: Coupling matrix
        """
        k_temp = np.zeros((2 * interaction_range * N_y_el, 
                          2 * interaction_range * N_y_scatter), dtype=float)
        
        atomnr_el = 0
        
        for i in range(interaction_range):
            for at_el in range(1, N_y_el + 1):
                atomnr_el += 1
                
                if ((N_y_el - N_y_scatter) // 2) <= at_el <= ((N_y_el - N_y_scatter) // 2) + N_y_scatter + 1:
                    atomnr_sc = 0
                    
                    for j in range(i + 1, 0, -1):
                        for at_sc in range(1, N_y_scatter + 1):
                            atomnr_sc += 1
                            
                            # X-direction coupling
                            if at_el == at_sc + ((N_y_el - N_y_scatter) // 2):
                                k_temp[2 * (atomnr_el - 1), 2 * (atomnr_sc - 1)] += -all_k_coupl_x[-j]
                            
                            # XY-direction coupling (only for nearest x-neighbors)
                            if i == interaction_range - 1:
                                # Corner atoms
                                if ((at_el == ((N_y_el - N_y_scatter) // 2) and at_sc == 1) or 
                                    (at_el == ((N_y_el - N_y_scatter) // 2) + N_y_scatter + 1 and 
                                     at_sc == N_y_scatter)):
                                    k_temp[2 * (atomnr_el - 1), 2 * (at_sc - 1)] = -all_k_coupl_xy[0]
                                    k_temp[2 * (atomnr_el - 1) + 1, 2 * (at_sc - 1) + 1] = -all_k_coupl_xy[0]
                                
                                # Diagonal neighbors in y-direction
                                if at_el == at_sc + ((N_y_el - N_y_scatter) // 2) and N_y_scatter > 1:
                                    if at_sc > 1:
                                        k_temp[2 * (atomnr_el - 1), 2 * (at_sc - 2)] = -all_k_coupl_xy[0]
                                        k_temp[2 * (atomnr_el - 1) + 1, 2 * (at_sc - 2) + 1] = -all_k_coupl_xy[0]
                                    
                                    if at_sc < N_y_scatter:
                                        k_temp[2 * (atomnr_el - 1), 2 * at_sc] = -all_k_coupl_xy[0]
                                        k_temp[2 * (atomnr_el - 1) + 1, 2 * at_sc + 1] = -all_k_coupl_xy[0]
        
        return k_temp
