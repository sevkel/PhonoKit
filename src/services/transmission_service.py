"""
Transmission Calculation Service

Service module for calculating phonon transmission coefficients.

Author: Severin Keller
Date: 2025
"""

import os
import numpy as np
from joblib import Parallel, delayed
from utils import matrix_gen as mg


class TransmissionCalculator:
    """
    Service class for calculating phonon transmission.
    
    Supports:
    - Analytical 1D Chain transmission
    - Standard transmission calculations
    - DecimationFourier transmission with momentum-space integration
    """
    
    def __init__(self, w, sigma_L, sigma_R, g_CC_ret, g_CC_adv, scatter, 
                 electrode_dict_L, electrode_dict_R, scatter_dict, electrode_L=None, 
                 sys_descr="", data_path="", batch_size=None):
        """
        Initialize the transmission calculator.
        
        Args:
            w (np.ndarray): Frequency array
            sigma_L (np.ndarray): Left self-energy
            sigma_R (np.ndarray): Right self-energy
            g_CC_ret (np.ndarray): Retarded Green's function
            g_CC_adv (np.ndarray): Advanced Green's function
            scatter: Scattering object
            electrode_dict_L (dict): Left electrode configuration
            electrode_dict_R (dict): Right electrode configuration
            scatter_dict (dict): Scatter configuration
            electrode_L (optional): Left electrode object
            sys_descr (str): System description
            data_path (str): Data output path
            batch_size (int, optional): Batch size for parallel processing
        """
        self.w = w
        self.sigma_L = sigma_L
        self.sigma_R = sigma_R
        self.g_CC_ret = g_CC_ret
        self.g_CC_adv = g_CC_adv
        self.scatter = scatter
        self.electrode_dict_L = electrode_dict_L
        self.electrode_dict_R = electrode_dict_R
        self.scatter_dict = scatter_dict
        self.electrode_L = electrode_L
        self.sys_descr = sys_descr
        self.data_path = data_path
        self.batch_size = batch_size
        self.N = len(w)
        self.i = np.arange(0, self.N, dtype=int)
        self.D = scatter.hessian
    
    def calculate(self) -> np.ndarray:
        """
        Calculate transmission based on electrode and scatter configuration.
        
        Returns:
            np.ndarray: Phonon transmission values
        """
        electrode_type_L = self.electrode_dict_L["type"]
        electrode_type_R = self.electrode_dict_R["type"]
        scatter_type = self.scatter_dict["type"]
        
        if (electrode_type_L == "Chain1D" and electrode_type_R == "Chain1D" and 
            scatter_type == "Chain1D"):
            return self._calculate_chain1d_transmission()
        elif (electrode_type_L == "DecimationFourier" and 
              electrode_type_R == "DecimationFourier" and 
              scatter_type == "FiniteLattice2D"):
            return self._calculate_decimation_fourier_transmission()
        else:
            return self._calculate_standard_transmission()
    
    def _calculate_chain1d_transmission(self) -> np.ndarray:
        """Calculate analytical 1D Chain transmission."""
        k_coupl_x_L = sum(mg.ranged_force_constant(
            k_coupl_x=self.electrode_dict_L["k_coupl_x"])["k_coupl_x"])
        k_coupl_x_R = sum(mg.ranged_force_constant(
            k_coupl_x=self.electrode_dict_R["k_coupl_x"])["k_coupl_x"])
        
        g_E_L = np.where(4 * k_coupl_x_L - self.w**2 >= 0, 
                        self.w * np.sqrt(4 * k_coupl_x_L - self.w**2), 0)
        g_E_R = np.where(4 * k_coupl_x_R - self.w**2 >= 0, 
                        self.w * np.sqrt(4 * k_coupl_x_R - self.w**2), 0)
        
        lambda_L = np.zeros((self.N, self.scatter.N, self.scatter.N), dtype=np.complex64)
        lambda_R = np.zeros((self.N, self.scatter.N, self.scatter.N), dtype=np.complex64)
        
        for i in range(self.N):
            lambda_L[i, 0, 0] = g_E_L[i]
            lambda_R[i, -1, -1] = g_E_R[i]
        
        trans_prob_matrix = np.array([
            np.dot(np.dot(self.g_CC_ret[i], lambda_L[i]), 
                  np.dot(self.g_CC_adv[i], lambda_R[i]))
            for i in self.i
        ])
        
        self._save_transmission_data(trans_prob_matrix)
        
        tau_ph = np.array([np.real(np.trace(trans_prob_matrix[i])) for i in self.i])
        
        return tau_ph
    
    def _calculate_decimation_fourier_transmission(self) -> np.ndarray:
        """Calculate transmission for DecimationFourier electrodes."""
        def transmission_worker(w_q_data):
            """Worker function for parallelized transmission computation."""
            results = []
            
            for w_idx, q_idx in w_q_data:
                spectral_dens_L_q = 1j * (self.sigma_L[w_idx, q_idx] - 
                                         np.transpose(np.conj(self.sigma_L[w_idx, q_idx])))
                spectral_dens_R_q = 1j * (self.sigma_R[w_idx, q_idx] - 
                                         np.transpose(np.conj(self.sigma_R[w_idx, q_idx])))
                
                probmat_q = np.dot(np.dot(self.g_CC_ret[w_idx, q_idx], spectral_dens_L_q), 
                                  np.dot(self.g_CC_adv[w_idx, q_idx], spectral_dens_R_q))
                
                tau_wq = np.real(np.trace(probmat_q))
                results.append((w_idx, q_idx, tau_wq, probmat_q))
            
            return results
        
        # Create indexed data
        N_q = len(self.electrode_L.q_y)
        w_q_data = [(w_idx, q_idx) 
                    for w_idx in range(self.N) 
                    for q_idx in range(N_q)]
        
        # Batch and parallelize
        batches = [w_q_data[i:i+self.batch_size] 
                  for i in range(0, len(w_q_data), self.batch_size)]
        
        batch_results = Parallel(n_jobs=-1)(
            delayed(transmission_worker)(batch) for batch in batches
        )
        
        # Initialize result arrays
        tau_ph_wq = np.zeros((self.N, N_q))
        tau_ph_probmat_wq = np.zeros((self.N, N_q, self.D.shape[0], self.D.shape[1]), 
                                    dtype=np.complex64)
        
        # Fill results
        for batch in batch_results:
            for w_idx, q_idx, tau_val, probmat in batch:
                tau_ph_wq[w_idx, q_idx] = tau_val
                tau_ph_probmat_wq[w_idx, q_idx] = probmat
        
    
        tau_ph = np.mean(tau_ph_wq, axis=1)
        tau_ph_probmat = np.mean(tau_ph_probmat_wq, axis=1)
        
        self._save_decimation_fourier_data(tau_ph_probmat, tau_ph_wq)
        
        return tau_ph
    
    def _calculate_standard_transmission(self) -> np.ndarray:
        """Calculate standard transmission."""
        spectral_dens_L = 1j * (self.sigma_L - np.transpose(np.conj(self.sigma_L), (0, 2, 1)))
        spectral_dens_R = 1j * (self.sigma_R - np.transpose(np.conj(self.sigma_R), (0, 2, 1)))
        
        trans_prob_matrix = np.array([
            np.dot(np.dot(self.g_CC_ret[i], spectral_dens_L[i]), 
                  np.dot(self.g_CC_adv[i], spectral_dens_R[i]))
            for i in self.i
        ])
        
        self._save_transmission_data(trans_prob_matrix)
        
        tau_ph = np.array([np.real(np.trace(trans_prob_matrix[i])) for i in self.i])
        
        return tau_ph
    
    def _save_transmission_data(self, trans_prob_matrix):
        """Save transmission probability matrix to file."""
        trans_prob_mat_path = os.path.join(self.data_path, "trans_prob_matrices")
        if not os.path.exists(trans_prob_mat_path):
            os.makedirs(trans_prob_mat_path)
        
        filename = self._generate_filename("trans_prob_matrix.npz")
        
        np.savez(os.path.join(trans_prob_mat_path, filename), 
                w=self.w, 
                trans_prob_matrix=trans_prob_matrix)
    
    def _save_decimation_fourier_data(self, tau_ph_probmat, tau_ph_wq):
        """Save DecimationFourier transmission data."""
        trans_prob_mat_path = os.path.join(self.data_path, "trans_prob_matrices")
        if not os.path.exists(trans_prob_mat_path):
            os.makedirs(trans_prob_mat_path)
        
        filename = self._generate_filename("trans_prob_matrix.npz")
        
        np.savez(os.path.join(trans_prob_mat_path, filename), 
                w=self.w, 
                q_y=self.electrode_L.q_y,
                trans_prob_matrix=tau_ph_probmat,
                tau_ph_wq=tau_ph_wq)
    
    def _generate_filename(self, suffix):
        """Generate filename for saving data."""
        base = (f"{self.sys_descr}___PT_elL={self.electrode_dict_L['type']}_"
               f"elR={self.electrode_dict_R['type']}_"
               f"CC={self.scatter_dict['type']}_"
               f"intrange={self.electrode_L.interaction_range if self.electrode_L else 'N/A'}")
        
        try:
            base += f"_kcoupl_x={self.electrode_dict_L['k_coupl_x']}"
        except KeyError:
            pass
        
        try:
            base += f"_kcoupl_xy={self.electrode_dict_L['k_coupl_xy']}"
        except KeyError:
            pass
        
        return f"{base}_{suffix}"
