"""
Green's Function Calculation Service

Service module for calculating retarded and advanced Green's functions.

Author: Severin Keller
Date: 2025
"""

import numpy as np
from joblib import Parallel, delayed


class GreensFunctionCalculator:
    """
    Service class for calculating retarded and advanced Green's functions.
    
    Supports:
    - Standard 3D calculations for most electrode types
    - 4D calculations for DecimationFourier electrodes (frequency-momentum space)
    """
    
    def __init__(self, w, D, sigma_L, sigma_R, electrode_dict_L, electrode_dict_R, 
                 electrode_L=None, batch_size=None):
        """
        Initialize the Green's function calculator.
        
        Args:
            w (np.ndarray): Frequency array
            D (np.ndarray): Dynamical matrix
            sigma_L (np.ndarray): Left self-energy
            sigma_R (np.ndarray): Right self-energy
            electrode_dict_L (dict): Left electrode configuration
            electrode_dict_R (dict): Right electrode configuration
            electrode_L (optional): Left electrode object (for DecimationFourier)
            batch_size (int, optional): Batch size for parallel processing
        """
        self.w = w
        self.D = D
        self.sigma_L = sigma_L
        self.sigma_R = sigma_R
        self.electrode_dict_L = electrode_dict_L
        self.electrode_dict_R = electrode_dict_R
        self.electrode_L = electrode_L
        self.batch_size = batch_size
        self.N = len(w)
        self.i = np.arange(0, self.N, dtype=int)
    
    def calculate(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Calculate retarded and advanced Green's functions.
        
        Returns:
            tuple: (g_CC_ret, g_CC_adv) - Retarded and advanced Green's functions
        """
        electrode_type_L = self.electrode_dict_L["type"]
        electrode_type_R = self.electrode_dict_R["type"]
        
        if (electrode_type_L, electrode_type_R) == ("DecimationFourier", "DecimationFourier"):
            return self._calculate_decimation_fourier()
        else:
            return self._calculate_standard()
    
    def _calculate_standard(self) -> tuple[np.ndarray, np.ndarray]:
        """Calculate Green's functions for standard electrode types (3D arrays)."""
        g_CC_ret = np.array([
            np.linalg.inv((self.w[i] + 1E-8j)**2 * np.identity(self.D.shape[0]) - 
                         self.D - self.sigma_L[i] - self.sigma_R[i])
            for i in self.i
        ])
        
        g_CC_adv = np.transpose(np.conj(g_CC_ret), (0, 2, 1))
        
        return g_CC_ret, g_CC_adv
    
    def _calculate_decimation_fourier(self) -> tuple[np.ndarray, np.ndarray]:
        """Calculate Green's functions for DecimationFourier electrodes (4D arrays)."""
        N_q = len(self.electrode_L.q_y)
        g_CC_ret = np.zeros((self.N, N_q, self.D.shape[0], self.D.shape[1]), 
                           dtype=np.complex64)
        g_CC_adv = np.zeros((self.N, N_q, self.D.shape[0], self.D.shape[1]), 
                           dtype=np.complex64)
        
        def gcc_worker(w_q_data):
            """Worker function for parallelized G_CC computation."""
            results_ret = []
            results_adv = []
            
            for w_idx, q_idx, w_val in w_q_data:
                matrix_to_invert = ((w_val + 1E-8j)**2 * np.identity(self.D.shape[0]) - 
                                   self.D - self.sigma_L[w_idx, q_idx] - 
                                   self.sigma_R[w_idx, q_idx])
                
                g_CC_ret_wq = np.linalg.inv(matrix_to_invert)
                g_CC_adv_wq = np.transpose(np.conj(g_CC_ret_wq))
                
                results_ret.append((w_idx, q_idx, g_CC_ret_wq))
                results_adv.append((w_idx, q_idx, g_CC_adv_wq))
            
            return results_ret, results_adv
        
        # Create indexed data
        w_q_data = []
        for w_idx, w_val in enumerate(self.w):
            for q_idx in range(N_q):
                w_q_data.append((w_idx, q_idx, w_val))
        
        # Batch and parallelize
        batches = [w_q_data[i:i+self.batch_size] 
                  for i in range(0, len(w_q_data), self.batch_size)]
        
        batch_results = Parallel(n_jobs=-1)(
            delayed(gcc_worker)(batch) for batch in batches
        )
        
        # Fill results
        for batch in batch_results:
            for (w_idx, q_idx, g_ret_matrix), (_, _, g_adv_matrix) in zip(batch[0], batch[1]):
                g_CC_ret[w_idx, q_idx] = g_ret_matrix
                g_CC_adv[w_idx, q_idx] = g_adv_matrix
        
        return g_CC_ret, g_CC_adv
