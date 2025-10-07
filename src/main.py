__docformat__ = "google"

import sys
import json
import os
from unicodedata import name
from joblib import Parallel, delayed
import numpy as np
from model_systems import * 
import electrode as el
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import tmoutproc as top
import calculate_kappa as ck
from utils import constants as const, matrix_gen as mg
import scienceplots #Import needed?


# For plotting
matplotlib.rcParams['font.family'] = r'C://Users//sevke//Desktop//Dev//fonts//fira_sans//FiraSans-Regular.ttf'
prop = fm.FontProperties(fname=r'C://Users//sevke//Desktop//Dev//fonts//fira_sans//FiraSans-Regular.ttf')
plt.style.use(['science', 'notebook', 'no-latex'])


class PhononTransport:
	"""
	Class for phonon transport calculations according to:

	M. Bürkle, Thomas J. Hellmuth, F. Pauly, Y. Asai, First-principles calculation of the 
    thermoelectric figure of merit for [2,2]paracyclophane-based single-molecule junctions, 
    PHYSICAL REVIEW B 91, 165419 (2015)
	DOI: 10.1103/PhysRevB.91.165419

	Here, the electrodes Green's functions and the central parts can be set up in different configurations.

	Args:
		data_path (str): Path where the data will be saved.
		sys_descr (str): System description for the data path.
		electrode_dict (dict): Dictionary containing the configuration of the enabled electrode.
		scatter_dict (dict): Dictionary containing the configuration of the enabled scatter object.
		E_D (float): Debye energy in meV.
		M_E (str): Atom type in the reservoir.
		M_C (str): Atom type coupled to the reservoir.
		N (int): Number of grid points.
		T_min (float): Minimum temperature for thermal conductance calculation.
		T_max (float): Maximum temperature for thermal conductance calculation.
		kappa_grid_points (int): Number of grid points for thermal conductance.

	Attributes:
		data_path (str): Path where the data will be saved.
		sys_descr (str): System description for the data path.
		electrode_dict (dict): Dictionary containing the configuration of the enabled electrode.
		scatter_dict (dict): Dictionary containing the configuration of the enabled scatter object.
		E_D (float): Debye energy in meV.
		M_E (str): Atom type in the reservoir.
		M_C (str): Atom type coupled to the reservoir.
		N (int): Number of grid points.
		T_min (float): Minimum temperature for thermal conductance calculation.
		T_max (float): Maximum temperature for thermal conductance calculation.
		kappa_grid_points (int): Number of grid points for thermal conductance.

	
	Raises:
		ValueError: If the electrode-center models don't match reasonable.

	"""

	def __init__(self, data_path, sys_descr, electrode_dict_L, electrode_dict_R, scatter_dict, 
			  E_D, M_E, M_R, M_C, N, T_min, T_max, kappa_grid_points):
		self.data_path = data_path
		self.sys_descr = sys_descr
		self.electrode_dict_L = electrode_dict_L
		self.electrode_dict_R = electrode_dict_R
		self.scatter_dict = scatter_dict
		self.M_E = M_E
		self.M_C = M_C
		self.M_R = M_R
		self.N = N
		self.E_D = E_D
		self.batch_size = max(1, int(N / os.cpu_count()))

		# Check for allowed combinations of electrode and scatter types
		if (self.electrode_dict_L["type"], self.electrode_dict_R["type"], self.scatter_dict["type"]) not in [
			("DebyeModel", "DebyeModel", "FiniteLattice2D"),
			("DebyeModel", "DebyeModel", "Chain1D"),
			("Ribbon2D", "Ribbon2D", "FiniteLattice2D"), 
			("Ribbon2D", "Ribbon2D", "Chain1D"),
			("Chain1D", "Chain1D", "Chain1D"),
			("AnalyticalFourier", "AnalyticalFourier", "FiniteLattice2D"),
			("AnalyticalFourier", "AnalyticalFourier", "Chain1D"),
			("DecimationFourier", "DecimationFourier", "FiniteLattice2D"),
			("DecimationFourier", "DecimationFourier", "Chain1D")
		]:
			raise ValueError(f"Invalid combination of electrode type '{self.electrode_L.type}', '{self.electrode_R.type}' and scatter type '{self.scatter.type}'")

		self.temperature = np.linspace(T_min, T_max, kappa_grid_points)
		self.w = np.linspace(1E-3, self.E_D * 1.1, N) #new
		self.i = np.linspace(0, self.N, self.N, False, dtype=int)

		print("########## Setting up the scatter region ##########")
		self.scatter = self.__initialize_scatter(self.scatter_dict, self.electrode_dict_L, self.electrode_dict_R)
  
		print("########## Setting up the electrodes ##########")
		self.electrode_L = self.__initialize_electrode(self.electrode_dict_L)
		self.electrode_R = self.__initialize_electrode(self.electrode_dict_R)

		self.D = self.scatter.hessian 
		self.sigma_L, self.sigma_R = self.calculate_sigma()
		self.g_CC_ret, self.g_CC_adv = self.calculate_G_cc()
		self.T = self.calculate_transmission()
		self.kappa = self.calc_kappa()
	
	def __initialize_electrode(self, electrode_dict) -> object:
		"""
		Initializes the electrode based on the provided configuration.

		Args:
			electrode_dict (dict): Dictionary containing the electrode configuration.

		Returns:
			Electrode (object): Initialized electrode object.

		Raises:
			ValueError: If a electrode type is undefined or unsupported.

		"""
		
		match electrode_dict["type"]:

			case "DebyeModel":
				return el.DebyeModel(
					self.w,
					k_coupl_x = electrode_dict["k_coupl_x"],
					k_coupl_xy = electrode_dict["k_coupl_xy"],
					w_D = self.E_D 
				)
			
			case "Chain1D":
				return el.Chain1D(
					self.w,
					interaction_range=electrode_dict["interaction_range"],
					interact_potential=electrode_dict["interact_potential"],
     				atom_type=electrode_dict["atom_type"],
					lattice_constant=electrode_dict["lattice_constant"],
     				k_el_x=electrode_dict["k_el_x"],
					k_coupl_x=electrode_dict["k_coupl_x"]
				)
			
			case "Ribbon2D":
				return el.Ribbon2D(
					self.w,
					interaction_range=electrode_dict["interaction_range"],
					interact_potential=electrode_dict["interact_potential"],
					atom_type=electrode_dict["atom_type"],
					lattice_constant=electrode_dict["lattice_constant"],
					N_y=electrode_dict["N_y"],
					N_y_scatter=self.scatter.N_y,
					M_E=self.M_E,
					M_C=self.M_C,
					k_el_x=electrode_dict["k_el_x"],
					k_el_y=electrode_dict["k_el_y"],
					k_el_xy=electrode_dict["k_el_xy"],
					k_coupl_x=electrode_dict["k_coupl_x"],
					k_coupl_xy=electrode_dict["k_coupl_xy"],
					left=electrode_dict["left"],
					right=electrode_dict["right"],
					batch_size=self.batch_size
				)
			
			case "AnalyticalFourier":
				return el.AnalyticalFourier(
					self.w,
					interaction_range=electrode_dict["interaction_range"],
					interact_potential=electrode_dict["interact_potential"],
					atom_type=electrode_dict["atom_type"],
					lattice_constant=electrode_dict["lattice_constant"],
					N_q=electrode_dict["N_q"],
					k_el_x=electrode_dict["k_el_x"],
					k_el_y=electrode_dict["k_el_y"],
					k_el_xy=electrode_dict["k_el_xy"],
					k_coupl_x=electrode_dict["k_coupl_x"],
					k_coupl_xy=electrode_dict["k_coupl_xy"],
					N_y_scatter=self.scatter.N_y,
					batch_size=self.batch_size
				)
			
			case "DecimationFourier":
				return el.DecimationFourier(
					self.w,
					N_q=electrode_dict["N_q"],
					interaction_range=electrode_dict["interaction_range"],
					interact_potential=electrode_dict["interact_potential"],
					atom_type=electrode_dict["atom_type"],
					lattice_constant=electrode_dict["lattice_constant"],
					N_y=electrode_dict["N_y"],
					N_y_scatter=self.scatter.N_y,
					M_E=self.M_E,
					M_C=self.M_C,
					k_el_x=electrode_dict["k_el_x"],
					k_el_y=electrode_dict["k_el_y"],
					k_el_xy=electrode_dict["k_el_xy"],
					k_coupl_x=electrode_dict["k_coupl_x"],
					k_coupl_xy=electrode_dict["k_coupl_xy"],
					left=electrode_dict["left"],
					right=electrode_dict["right"],
					batch_size=self.batch_size
				)
	
			case _:
				raise ValueError(f"Unsupported electrode type: {electrode_dict['type']}")

	def __initialize_scatter(self, scatter_dict, electrode_dict_l, electrode_dict_r) -> object:
		"""
		Initializes the scatter object based on the provided configuration.

		Args:
			scatter_dict (dict): Dictionary containing the scatter configuration.

		Returns:
			Scatter (object): Initialized scatter object.

		Raises:
			ValueError: If scatter type is undefined or unsupported.

		"""
		match scatter_dict["type"]:

			case "FiniteLattice2D":
				return FiniteLattice2D(
					N_y=scatter_dict["N_y"],
					N_x=scatter_dict["N_x"],
					N_y_el_L=electrode_dict_l["N_y"],
					N_y_el_R=electrode_dict_r["N_y"],
					k_coupl_x_l=electrode_dict_l["k_coupl_x"],
					k_c_x=scatter_dict["k_c_x"],
					k_coupl_x_r=electrode_dict_r["k_coupl_x"],
					k_c_y=scatter_dict["k_c_y"],
					k_c_xy=scatter_dict["k_c_xy"],
					k_coupl_xy_l=electrode_dict_l["k_coupl_xy"],
					k_coupl_xy_r=electrode_dict_r["k_coupl_xy"],
					interact_potential=scatter_dict["interact_potential"],
					interaction_range=scatter_dict["interaction_range"],
					lattice_constant=scatter_dict["lattice_constant"],
					atom_type=scatter_dict["atom_type"]
				)
    
			case "Chain1D":
				return Chain1D(
					k_c=scatter_dict["k_c_x"],
					k_coupl_l=electrode_dict_l["k_el_x"],
					k_coupl_r=electrode_dict_r["k_el_x"],
					interact_potential=scatter_dict["interact_potential"],
					interaction_range=scatter_dict["interaction_range"],
					lattice_constant=scatter_dict["lattice_constant"],
					atom_type=scatter_dict["atom_type"],
					N=scatter_dict["N"]
				)
			
			case _:
				raise ValueError(f"Unsupported scatter type: {scatter_dict['type']}")

	def calculate_sigma(self) -> tuple[np.ndarray]:
		"""
		Calculates self-energies for the left (L) and right (R) electrodes.

		Returns:
			sigma_L, sigma_R (np.ndarray): Self-energy of the left (L) and right (R) electrode.

		Raises: ValueError if electrode configurations don't match.
		"""

		match (self.electrode_dict_L["type"], self.electrode_dict_R["type"]):

			case ("DebyeModel", "DebyeModel"):
				# Scalar Greens function
				g_L = self.electrode_L.g
				g_R = self.electrode_R.g

				k_coupl_l = self.electrode_L.k_coupl_x 
				k_coupl_r = self.electrode_R.k_coupl_x 
												
			case ("Chain1D", "Chain1D"):

				k_coupl_x = sum(mg.ranged_force_constant(k_el_x=self.electrode_dict_L["k_coupl_x"])["k_coupl_x"])
				f_E = 0.5 * (self.w**2 - 2 * k_coupl_x - self.w * np.sqrt(self.w**2 - 4 * k_coupl_x, dtype=np.complex64)) 

				sigma_L = np.zeros((self.N, self.scatter.N, self.scatter.N), dtype=np.complex64)
				sigma_R = np.zeros((self.N, self.scatter.N, self.scatter.N), dtype=np.complex64)

				for i in range(self.N):
    
					sigma_L[i, 0, 0] = f_E[i] 
					sigma_R[i, -1, -1] = f_E[i]
     
				return sigma_L, sigma_R
			
			case ("AnalyticalFourier", "AnalyticalFourier"):

				g_L = self.electrode_L.g
				g_R = self.electrode_R.g

				k_LC = self.electrode_L.center_coupling # center_coupling is enough as we allow only nearest neighbours here.
				k_RC = self.electrode_R.center_coupling
    			
			case ("Ribbon2D", "Ribbon2D") | ("DecimationFourier", "DecimationFourier"):
				
				g_L = self.electrode_L.g
				g_R = self.electrode_R.g

				#set up coupling matrices (L,R) x C dimensional
				k_LC = np.zeros((2 * self.electrode_L.interaction_range * self.electrode_L.N_y, 
					 2 * self.scatter.N_x * self.scatter.N_y), dtype=float)
				
				k_RC = np.zeros((2 * self.electrode_R.interaction_range * self.electrode_R.N_y, 
					 2 * self.scatter.N_x * self.scatter.N_y), dtype=float)
				
				k_LC_temp = np.zeros((2 * self.electrode_L.interaction_range * self.electrode_L.N_y, 
					 2 * self.electrode_L.interaction_range * self.scatter.N_y), dtype=float)
    
				k_RC_temp = np.zeros((2 * self.electrode_R.interaction_range * self.electrode_R.N_y, 
					2 * self.electrode_R.interaction_range * self.scatter.N_y), dtype=float)
    
				N_y_L = self.electrode_L.N_y
				N_y_R = self.electrode_R.N_y
				N_y_scatter = self.scatter.N_y

				interaction_range_L = self.electrode_L.interaction_range
				interaction_range_R = self.electrode_R.interaction_range
				all_k_coupl_x_L = mg.ranged_force_constant(k_coupl_x=self.electrode_dict_L["k_coupl_x"])["k_coupl_x"]
				all_k_coupl_x_R = mg.ranged_force_constant(k_coupl_x=self.electrode_dict_R["k_coupl_x"])["k_coupl_x"]
				all_k_coupl_xy_L = mg.ranged_force_constant(k_coupl_xy=self.electrode_dict_L["k_coupl_xy"])["k_coupl_xy"]
				all_k_coupl_xy_R = mg.ranged_force_constant(k_coupl_xy=self.electrode_dict_R["k_coupl_xy"])["k_coupl_xy"]
    	
				atomnr_el = 0

				# Set up LC interaction matrix	
				for i in range(interaction_range_L):
					for at_el in range(1, N_y_L + 1):
						atomnr_el += 1
      
						if ((N_y_L - N_y_scatter) // 2) <= at_el <= ((N_y_L - N_y_scatter) // 2) + N_y_scatter + 1:
							atomnr_sc = 0
							
							for j in range(i + 1, 0, -1):
								for at_sc in range(1, N_y_scatter + 1):
									atomnr_sc += 1

									# look if sc and el are aligned 
									if at_el == at_sc + ((N_y_L - N_y_scatter) // 2):
										# coupling in x-direction
										k_LC_temp[2 * (atomnr_el - 1), 2 * (atomnr_sc - 1)] += -all_k_coupl_x_L[-j]
		
									if i == interaction_range_L - 1:
										if (at_el == ((N_y_L - N_y_scatter) // 2) and at_sc == 1) or (at_el == ((N_y_L - N_y_scatter) // 2) + N_y_scatter + 1 and at_sc == N_y_scatter):
											# coupling also xy
											k_LC_temp[2 * (atomnr_el - 1), 2 * (at_sc - 1)] = -all_k_coupl_xy_L[0]
											k_LC_temp[2 * (atomnr_el - 1) + 1, 2 * (at_sc - 1) + 1] = -all_k_coupl_xy_L[0]
           
										# xy coupling only for diagonally opposite atoms (neighbors in y-direction)
										if at_el == at_sc + ((N_y_L - N_y_scatter) // 2) and N_y_scatter > 1:
           
											if at_sc > 1:
												# coupling to previous neighbor (diagonal)
												k_LC_temp[2 * (atomnr_el - 1), 2 * (at_sc - 2)] = -all_k_coupl_xy_L[0]
												k_LC_temp[2 * (atomnr_el - 1) + 1, 2 * (at_sc - 2) + 1] = -all_k_coupl_xy_L[0]

											if at_sc < N_y_scatter:
												# coupling to next neighbor (diagonal)
												k_LC_temp[2 * (atomnr_el - 1), 2 * at_sc] = -all_k_coupl_xy_L[0]
												k_LC_temp[2 * (atomnr_el - 1) + 1, 2 * at_sc + 1] = -all_k_coupl_xy_L[0]
					
				atomnr_el = 0
				
				# Set up RC interaction matrix	
				for i in range(interaction_range_R):
					for at_el in range(1, N_y_R + 1):
						atomnr_el += 1
      
						if ((N_y_R - N_y_scatter) // 2) <= at_el <= ((N_y_R - N_y_scatter) // 2) + N_y_scatter + 1:
							atomnr_sc = 0
								
							for j in range(i + 1, 0, -1):
								for at_sc in range(1, N_y_scatter + 1):
									atomnr_sc += 1
									
									# look if sc and el are aligned 
									if at_el == at_sc + ((N_y_R - N_y_scatter) // 2):
										# coupling in x-direction
										k_RC_temp[2 * (atomnr_el - 1), 2 * (atomnr_sc - 1)] = -all_k_coupl_x_R[-j]
		
									if i == interaction_range_R - 1:
             
										if (at_el == ((N_y_R - N_y_scatter) // 2) and at_sc == 1) or (at_el == ((N_y_R - N_y_scatter) // 2) + N_y_scatter + 1 and at_sc == N_y_scatter):
											# coupling also xy
											k_RC_temp[2 * (atomnr_el - 1) + 1, 2 * (at_sc - 1) + 1] = -all_k_coupl_xy_R[0]
											k_RC_temp[2 * (atomnr_el - 1), 2 * (at_sc - 1)] = -all_k_coupl_xy_R[0]
           
										# xy coupling only for diagonally opposite atoms (neighbors in y-direction)
										if at_el == at_sc + ((N_y_R - N_y_scatter) // 2) and N_y_scatter > 1:
           
											if at_sc > 1:
												# coupling to previous neighbor (diagonal)
												k_RC_temp[2 * (atomnr_el - 1), 2 * (at_sc - 2)] = -all_k_coupl_xy_R[0]
												k_RC_temp[2 * (atomnr_el - 1) + 1, 2 * (at_sc - 2) + 1] = -all_k_coupl_xy_R[0]

											if at_sc < N_y_scatter:
												# coupling to next neighbor (diagonal)
												k_RC_temp[2 * (atomnr_el - 1), 2 * at_sc] = -all_k_coupl_xy_R[0]
												k_RC_temp[2 * (atomnr_el - 1) + 1, 2 * at_sc + 1] = -all_k_coupl_xy_R[0]

				if interaction_range_R > 1:
					mid_col_R = k_RC_temp.shape[1] // 2
					k_RC_temp = np.hstack([k_RC_temp[:, mid_col_R:], k_RC_temp[:, :mid_col_R]])
					
				# Fill k_LC from top left and k_RC from bottom right
				k_LC[:k_LC_temp.shape[0], :k_LC_temp.shape[1]] = k_LC_temp
				k_RC[-k_RC_temp.shape[0]:, -k_RC_temp.shape[1]:] = k_RC_temp


		# Initialize sigma array with the same shape as Green's function g
		if (self.electrode_dict_L["type"], self.electrode_dict_R["type"]) == ("DecimationFourier", "DecimationFourier"):
        	# 4D arrays for DecimationFourier: (N_w, N_q, matrix_dim, matrix_dim)
			sigma_L = np.zeros((self.N, len(self.electrode_L.q_y), self.D.shape[0], self.D.shape[1]), dtype=np.complex64)
			sigma_R = np.zeros((self.N, len(self.electrode_R.q_y), self.D.shape[0], self.D.shape[1]), dtype=np.complex64)
		else:
			# 3D arrays for other electrode types: (N_w, matrix_dim, matrix_dim)
			sigma_L = np.zeros((self.N, self.D.shape[0], self.D.shape[1]), dtype=np.complex64)
			sigma_R = np.zeros((self.N, self.D.shape[0], self.D.shape[1]), dtype=np.complex64)
		
		# Build sigma matrix for each frequency depending on the (allowed) electrode model configuration
		# The 1D Chain transmission has an analytical expression an is covered directly there.
		
		# DebyeModel (Markussen)
		if (g_L.shape, g_R.shape) == ((self.N,), (self.N,)):

			sigma_nu_l = np.array(list(map(lambda i: k_coupl_l**2 * g_L[i], self.i)))
			sigma_nu_r = np.array(list(map(lambda i: k_coupl_r**2 * g_R[i], self.i)))

			match self.scatter_dict["type"]:
			
				case "FiniteLattice2D":
					# get N_y scatter and set sigma_nu on the positions of coupling atoms
					# !!! Not sure if this Debye model is valid for FiniteLattice2D !!!
					for k in range(self.N):

						for n in range(self.scatter.N_y):
							sigma_L[k][n * 2, n * 2] = sigma_nu_l[k]
							sigma_R[k][sigma_R.shape[1] - 2 - n * 2, sigma_R.shape[2] - 2 - n * 2] = sigma_nu_r[k]
				
				case "Chain1D":

					for k in range(self.N):

						sigma_L[k][0, 0] = sigma_nu_l[k]
						sigma_R[k][-1, -1] = sigma_nu_r[k]
						
		elif (electrode_dict_L["type"], electrode_dict_R["type"]) == ("Ribbon2D", "Ribbon2D"):
			
			# Does parallelizing makes sense here?
			sigma_L = np.array(list(map(lambda i: np.dot(np.dot(k_LC.T, g_L[i]), k_LC), self.i)))
			sigma_R = np.array(list(map(lambda i: np.dot(np.dot(k_RC.T, g_R[i]), k_RC), self.i)))
   
			print('debug')

		elif (electrode_dict_L["type"], electrode_dict_R["type"]) == ("DecimationFourier", "DecimationFourier"):
			
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
			
			# Create indexed data for parallelization
			w_q_data = []
			for w_idx in range(g_L.shape[0]):
				for q_idx in range(g_L.shape[1]):
					w_q_data.append((w_idx, q_idx, g_L[w_idx, q_idx]))
			
			# Batch the data
			batch_size = self.batch_size
			batches = [w_q_data[i:i+batch_size] for i in range(0, len(w_q_data), batch_size)]
			
			# Parallel processing
			batch_results = Parallel(n_jobs=-1)(
				delayed(sigma_worker)(batch) for batch in batches
			)
			
			# Fill the indexed results
			for batch in batch_results:
				for (w_idx, q_idx, sigma_L_matrix), (_, _, sigma_R_matrix) in zip(batch[0], batch[1]):
					sigma_L[w_idx, q_idx] = sigma_L_matrix
					sigma_R[w_idx, q_idx] = sigma_R_matrix

		# InfitineFourier2D case TODO: dimensionality problem due to xy coupling -> dim(D) == dim(sigmaL,R) ggf. < dim(g_L,R) 
		elif (electrode_dict_L["type"], electrode_dict_R["type"]) == ("AnalyticalFourier", "AnalyticalFourier"):

			sigma_L_temp = np.array(list(map(lambda i: np.dot(np.dot(k_LC.T, g_L[i]), k_LC), self.i)))
			sigma_R_temp = np.array(list(map(lambda i: np.dot(np.dot(k_RC.T, g_R[i]), k_RC), self.i)))

			for i in range(self.N):
				sigma_L[i, 0: sigma_L_temp.shape[1], 0: sigma_L_temp.shape[2]] = sigma_L_temp[i]
				sigma_R[i, sigma_R.shape[1] - sigma_R_temp.shape[1]: sigma_R.shape[1], sigma_R.shape[2] - sigma_R_temp.shape[2]: sigma_R.shape[2]] = sigma_R_temp[i]

		else:
			raise ValueError(f"Unsupported shape for g_L: {g_L.shape} or g_R: {g_R.shape}.\n Or unallowed combination of electrode types")

		return sigma_L, sigma_R

	def calculate_G_cc(self) -> tuple[np.ndarray]:
		"""
		Calculates retarded and advanced Green's functions for the central part with given parameters at given frequency w.

		Returns:
			g_CC_ret, g_CC_adv (np.ndarray): Greens function for the central part.

		"""

		if (self.electrode_dict_L["type"], self.electrode_dict_R["type"]) == ("DecimationFourier", "DecimationFourier"):

			N_q = len(self.electrode_L.q_y)
			g_CC_ret = np.zeros((self.N, N_q, self.D.shape[0], self.D.shape[1]), dtype=np.complex64)
			g_CC_adv = np.zeros((self.N, N_q, self.D.shape[0], self.D.shape[1]), dtype=np.complex64)

			def gcc_worker(w_q_data):
				"""Worker function for parallelized G_CC computation."""
				results_ret = []
				results_adv = []
				
				for w_idx, q_idx, w_val in w_q_data:
					# Matrix inversion for each (w, q) combination
					matrix_to_invert = ((w_val + 1E-16j)**2 * np.identity(self.D.shape[0]) - 
									self.D - self.sigma_L[w_idx, q_idx] - self.sigma_R[w_idx, q_idx])
					
					g_CC_ret_wq = np.linalg.inv(matrix_to_invert)
					g_CC_adv_wq = np.transpose(np.conj(g_CC_ret_wq))
					
					results_ret.append((w_idx, q_idx, g_CC_ret_wq))
					results_adv.append((w_idx, q_idx, g_CC_adv_wq))
				
				return results_ret, results_adv
			
			# Create indexed data for parallelization
			w_q_data = []
			for w_idx, w_val in enumerate(self.w):
				for q_idx in range(N_q):
					w_q_data.append((w_idx, q_idx, w_val))
			
			# Batch the data
			batch_size = self.batch_size
			batches = [w_q_data[i:i+batch_size] for i in range(0, len(w_q_data), batch_size)]
			
			# Parallel processing
			batch_results = Parallel(n_jobs=-1)(
				delayed(gcc_worker)(batch) for batch in batches
			)
			
			# Fill the indexed results
			for batch in batch_results:
				for (w_idx, q_idx, g_ret_matrix), (_, _, g_adv_matrix) in zip(batch[0], batch[1]):
					g_CC_ret[w_idx, q_idx] = g_ret_matrix
					g_CC_adv[w_idx, q_idx] = g_adv_matrix

		else:
			g_CC_ret = np.array(list(map(lambda i: np.linalg.inv((self.w[i] + 1E-16j)**2 * np.identity(self.D.shape[0]) - self.D - self.sigma_L[i] - self.sigma_R[i]), self.i)))
			g_CC_adv = np.transpose(np.conj(g_CC_ret), (0, 2, 1))

		return g_CC_ret, g_CC_adv	

	def calculate_transmission(self) -> np.ndarray:
		"""
		Calculates the phonon transmission for the given parameters at given frequency w.

		Returns:
			tau_ph (np.ndarray): Phonon transmission values.

		"""

		trans_prob_mat_path = os.path.join(self.data_path, "trans_prob_matrices")
		if not os.path.exists(trans_prob_mat_path):
			os.makedirs(trans_prob_mat_path)

		if self.electrode_dict_L["type"] == "Chain1D" and self.electrode_dict_R["type"] == "Chain1D" and scatter_dict["type"] == "Chain1D":
			# 1D Chain transmission has an analytical expression
			k_coupl_x_L = sum(mg.ranged_force_constant(k_el_x=self.electrode_dict_L["k_coupl_x"])["k_coupl_x"])
			k_coupl_x_R = sum(mg.ranged_force_constant(k_el_x=self.electrode_dict_R["k_coupl_x"])["k_coupl_x"])
		
			g_E_L = np.where(4 * k_coupl_x_L - self.w**2 >= 0, self.w * np.sqrt(4 * k_coupl_x_L - self.w**2), 0)
			g_E_R = np.where(4 * k_coupl_x_R - self.w**2 >= 0, self.w * np.sqrt(4 * k_coupl_x_R - self.w**2), 0)
			
			lambda_L = np.zeros((self.N, self.scatter.N, self.scatter.N), dtype=np.complex64)
			lambda_R = np.zeros((self.N, self.scatter.N, self.scatter.N), dtype=np.complex64)
   
			for i in range(self.N):
				lambda_L[i, 0, 0] = g_E_L[i]
				lambda_R[i, -1, -1] = g_E_R[i]

			trans_prob_matrix = np.array(list(map(lambda i: np.dot(np.dot(self.g_CC_ret[i], lambda_L[i]), np.dot(self.g_CC_adv[i], lambda_R[i])), self.i)))
			
			np.savez(os.path.join(trans_prob_mat_path, 
						f"{self.sys_descr}___PT_elL={self.electrode_dict_L["type"]}_elR={self.electrode_dict_R["type"]}_"
						f"CC={self.scatter_dict["type"]}_intrange={self.electrode_L.interaction_range}_kcoupl_x={electrode_dict_L["k_coupl_x"]}_trans_prob_matrix.npz"),
						w=self.w, trans_prob_matrix=trans_prob_matrix)

			tau_ph = np.array(list(map(lambda i: np.real(np.trace(trans_prob_matrix[i])), self.i)))
				
			return tau_ph

		elif self.electrode_dict_L["type"] == "DecimationFourier" and self.electrode_dict_R["type"] == "DecimationFourier" and scatter_dict["type"] == "FiniteLattice2D":

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
			
			# Create indexed data for parallelization
			N_q = len(self.electrode_L.q_y)
			w_q_data = []
			for w_idx in range(self.N):
				for q_idx in range(N_q):
					w_q_data.append((w_idx, q_idx))
			
			# Batch the data
			batch_size = self.batch_size
			batches = [w_q_data[i:i+batch_size] for i in range(0, len(w_q_data), batch_size)]
			
			# Parallel processing
			batch_results = Parallel(n_jobs=-1)(
				delayed(transmission_worker)(batch) for batch in batches
			)
			
			# Initialize result arrays
			tau_ph_wq = np.zeros((self.N, N_q))
			tau_ph_probmat_wq = np.zeros((self.N, N_q, self.D.shape[0], self.D.shape[1]), dtype=np.complex64)
			
			# Fill the indexed results
			for batch in batch_results:
				for w_idx, q_idx, tau_val, probmat in batch:
					tau_ph_wq[w_idx, q_idx] = tau_val
					tau_ph_probmat_wq[w_idx, q_idx] = probmat
			
			# Average over q-points for final transmission
			tau_ph = np.mean(tau_ph_wq, axis=1)
			tau_ph_probmat = np.mean(tau_ph_probmat_wq, axis=1)

			try:
				filename = (f"{self.sys_descr}___PT_elL={self.electrode_dict_L['type']}_elR={self.electrode_dict_R['type']}_"
						f"CC={self.scatter_dict['type']}_intrange={self.electrode_L.interaction_range}_"
						f"kcoupl_x={self.electrode_dict_L['k_coupl_x']}_"
						f"kcoupl_xy={self.electrode_dict_L['k_coupl_xy']}_trans_prob_matrix.npz")
				
			except KeyError as e:
				filename = (f"{self.sys_descr}___PT_elL={self.electrode_dict_L['type']}_elR={self.electrode_dict_R['type']}_"
						f"CC={self.scatter_dict['type']}_intrange={self.electrode_L.interaction_range}_"
						f"kcoupl_x={self.electrode_dict_L['k_coupl_x']}_trans_prob_matrix.npz")
				
			np.savez(os.path.join(trans_prob_mat_path, filename), w=self.w, trans_prob_matrix=tau_ph_probmat)
			
			return tau_ph
		
		# Standard case for other electrode combinations
		spectral_dens_L = 1j * (self.sigma_L - np.transpose(np.conj(self.sigma_L), (0, 2, 1)))
		spectral_dens_R = 1j * (self.sigma_R - np.transpose(np.conj(self.sigma_R), (0, 2, 1)))

		trans_prob_matrix = np.array(list(map(lambda i: np.dot(np.dot(self.g_CC_ret[i], spectral_dens_L[i]), np.dot(self.g_CC_adv[i], spectral_dens_R[i])), self.i)))

		try:
			filename = (f"{self.sys_descr}___PT_elL={self.electrode_dict_L['type']}_elR={self.electrode_dict_R['type']}_"
					f"CC={self.scatter_dict['type']}_intrange={self.electrode_L.interaction_range}_"
					f"kcoupl_x={self.electrode_dict_L['k_coupl_x']}_"
					f"kcoupl_xy={self.electrode_dict_L['k_coupl_xy']}_trans_prob_matrix.npz")
		except KeyError as e:
			filename = (f"{self.sys_descr}___PT_elL={self.electrode_dict_L['type']}_elR={self.electrode_dict_R['type']}_"
					f"CC={self.scatter_dict['type']}_intrange={self.electrode_L.interaction_range}_"
					f"kcoupl_x={self.electrode_dict_L['k_coupl_x']}_trans_prob_matrix.npz")
		
		np.savez(os.path.join(trans_prob_mat_path, filename), w=self.w, trans_prob_matrix=trans_prob_matrix)

		tau_ph = np.array(list(map(lambda i: np.real(np.trace(trans_prob_matrix[i])), self.i)))

		return tau_ph

	def calc_kappa(self) -> np.ndarray:
		"""
		Calculates the phonon thermal conductance.

		Returns:
			kappa (np.ndarray): Phonon thermal conductance values for each temperature in self.temperature.

		"""

		kappa = list()
  
		# w to SI
		w_kappa = self.w * const.unit2SI
		E = const.h_bar * w_kappa

		# joule to hartree
		E = E / const.har2J

		valid_indices = ~np.isnan(self.T)

		if False in valid_indices:
			notespath = os.path.join(self.data_path, "notes.txt")
			if not os.path.exists(notespath):
				with open(notespath, "w") as f:

					try:
						f.write(f"Invalid data points found in T in file {self.sys_descr}___PT_elL={self.electrode_dict_L["type"]}_elR={self.electrode_dict_R["type"]}_"
			  					f"CC={self.scatter_dict["type"]}_intrange={self.electrode_L.interaction_range}_kcoupl_x={electrode_dict_L["k_coupl_x"]}_"
								f"kcoupl_xy={self.electrode_dict_L["k_coupl_xy"]}_KAPPA.dat")

					except KeyError as e:
						f.write(f"Invalid data points found in T in file {self.sys_descr}___PT_elL={self.electrode_dict_L["type"]}_elR={self.electrode_dict_R["type"]}_"
			  					f"CC={self.scatter_dict["type"]}_intrange={self.electrode_L.interaction_range}_kcoupl_x={electrode_dict_L["k_coupl_x"]}_KAPPA.dat")
					f.close()

		valid_T = self.T[valid_indices]
		valid_E = E[valid_indices]

		for j in range(0, len(self.temperature)):
			kappa.append(ck.calculate_kappa(valid_T[1:len(valid_T)], valid_E[1:len(valid_E)], self.temperature[j]) * const.har2pJ)

		return kappa

	def plot_transport(self, write_data=True, plot_data=False) -> None:
		"""
		Writes out the raw data and plots the transport properties of the system.
		
		Args:
			write_data (bool): Flag if data write-out is wanted. Writes it to the data path.
			plot_data (bool): Flag if the data shall be plotted or not.

		"""

		if write_data:
			
			path_trans = os.path.join(self.data_path, "trans")
			path_transdos = os.path.join(self.data_path, "trans+dos")
			path_kappa = os.path.join(self.data_path, "kappa")

			if not os.path.exists(path_trans):
				os.makedirs(path_trans)
			if not os.path.exists(path_transdos):
				os.makedirs(path_transdos)
			if not os.path.exists(path_kappa):
				os.makedirs(path_kappa)

			try:
				top.write_plot_data(path_trans + 
						f"/{self.sys_descr}___PT_elL={self.electrode_dict_L["type"]}_elR={self.electrode_dict_R["type"]}_"
						f"CC={self.scatter_dict["type"]}_intrange={self.electrode_L.interaction_range}_kcoupl_x={electrode_dict_L["k_coupl_x"]}_"
						f"kcoupl_xy={self.electrode_dict_L["k_coupl_xy"]}.dat", 
						(self.w, self.T), "w (sqrt(har/(bohr**2*u))), T_vals")
							
				top.write_plot_data(path_transdos + 
						f"/{self.sys_descr}___PT_elL={self.electrode_dict_L["type"]}_elR={self.electrode_dict_R["type"]}_"
						f"CC={self.scatter_dict["type"]}_intrange={self.electrode_L.interaction_range}_kcoupl_x={electrode_dict_L["k_coupl_x"]}_"
						f"kcoupl_xy={self.electrode_dict_L["k_coupl_xy"]}.dat", 
						(self.w, self.T), "w (sqrt(har/(bohr**2*u))), T_vals")
				
				top.write_plot_data(path_kappa + 
						f"/{self.sys_descr}___PT_elL={self.electrode_dict_L["type"]}_elR={self.electrode_dict_R["type"]}_"
						f"CC={self.scatter_dict["type"]}_intrange={self.electrode_L.interaction_range}_kcoupl_x={electrode_dict_L["k_coupl_x"]}_"
						f"kcoupl_xy={self.electrode_dict_L["k_coupl_xy"]}_KAPPA.dat", 
						(self.temperature, self.kappa), "T (K), kappa (pW/K)")

			except KeyError as e:
				top.write_plot_data(path_trans + 
						f"/{self.sys_descr}___PT_elL={self.electrode_dict_L["type"]}_elR={self.electrode_dict_R["type"]}_"
						f"CC={self.scatter_dict["type"]}_intrange={self.electrode_L.interaction_range}_kcoupl_x={electrode_dict_L["k_coupl_x"]}.dat", 
						(self.w, self.T), "w (sqrt(har/(bohr**2*u))), T_vals")
				
				top.write_plot_data(path_transdos + 
						f"/{self.sys_descr}___PT_elL={self.electrode_dict_L["type"]}_elR={self.electrode_dict_R["type"]}_"
						f"CC={self.scatter_dict["type"]}_intrange={self.electrode_L.interaction_range}_kcoupl_x={electrode_dict_L["k_coupl_x"]}.dat", 
						(self.w, self.T), "w (sqrt(har/(bohr**2*u))), T_vals")
				
				top.write_plot_data(path_kappa + 
						f"/{self.sys_descr}___PT_elL={self.electrode_dict_L["type"]}_elR={self.electrode_dict_R["type"]}_"
						f"CC={self.scatter_dict["type"]}_intrange={self.electrode_L.interaction_range}_kcoupl_x={electrode_dict_L["k_coupl_x"]}_KAPPA.dat", 
						(self.temperature, self.kappa), "T (K), kappa (pW/K)")


		print(f'TauMax = {max(self.T)}, TauMin = {min(self.T)}, T_0 = {self.T[0]}')
		print(f'KappaMax = {max(self.kappa)}, KappaMin = {min(self.kappa)}')
		
		
		if plot_data:
			fig, ax1 = plt.subplots(1, 1)
			fig.tight_layout()
			ax1.plot(self.w, self.T)
			ax1.set_xlabel(r'Phonon Energy ($\mathrm{meV}$)', fontsize=12, fontproperties=prop)
			ax1.set_ylabel(r'$\tau_{\mathrm{ph}}$', fontsize=12, fontproperties=prop)
			ax1.set_xlim(0, 0.6 * E_D)
			ax1.set_xticklabels(ax1.get_xticks(), fontproperties=prop)
			ax1.set_yticklabels(ax1.get_yticks(), fontproperties=prop)
			ax1.grid()
	
		
			plt.rc('xtick', labelsize=12)
			plt.rc('ytick', labelsize=12)
			plt.xticks(fontproperties=prop)
			plt.yticks(fontproperties=prop)
	
			try:
				plt.savefig(self.data_path + 
						f"/{self.sys_descr}___PT_elL={self.electrode_dict_L["type"]}_elR={self.electrode_dict_R["type"]}_"
						f"CC={self.scatter_dict["type"]}_intrange={self.electrode_L.interaction_range}_kcoupl_x={self.electrode_dict_L["k_coupl_x"]}_"
						f"kcoupl_xy={self.electrode_dict_L["k_coupl_xy"]}.pdf", 
						bbox_inches='tight')
				
			except KeyError as e:
				plt.savefig(self.data_path + 
						f"/{self.sys_descr}___PT_elL={self.electrode_dict_L["type"]}_elR={self.electrode_dict_R["type"]}_"
						f"CC={self.scatter_dict["type"]}_intrange={self.electrode_L.interaction_range}_"
						f"kcoupl_x={self.electrode_dict_L["k_coupl_x"]}.pdf", 
						bbox_inches='tight')
	
			plt.clf()

	def plot_dos(self, write_data=True, plot_dos=False) -> None:
		"""
		Plots the density of states (DOS) for the left and right electrode.

		Args:
			write_data (bool): Flag if data write-out is wanted. Writes it to the data path.
			plot_dos (bool): Flag if the DOS-data shall be plotted or not.

		"""
  		
		dos_L = self.electrode_L.dos
		dos_real_L = self.electrode_L.dos_real
		dos_L_cpld = self.electrode_L.dos_cpld
		dos_real_L_cpld = self.electrode_L.dos_real_cpld
  
		dos_R = self.electrode_R.dos
		dos_real_R = self.electrode_R.dos_real
		dos_R_cpld = self.electrode_R.dos_cpld
		dos_real_R_cpld = self.electrode_R.dos_real_cpld
  
		print(f'DOS Left electrode max/min: {max(dos_L)}, {min(dos_L)}')
		print(f'DOS Right electrode max/min: {max(dos_R)}, {min(dos_R)}')
		
		if write_data:

			path_dos = os.path.join(self.data_path, "dos")
			path_transdos = os.path.join(self.data_path, "trans+dos")

			if not os.path.exists(path_dos):
				os.makedirs(path_dos)

			if not os.path.exists(path_transdos):
				os.makedirs(path_transdos)

			try:
				top.write_plot_data(path_dos + 
						f"/{self.sys_descr}___PT_elL={self.electrode_dict_L["type"]}_elR={self.electrode_dict_R["type"]}_"
						f"CC={self.scatter_dict["type"]}_intrange={self.electrode_L.interaction_range}_kcoupl_x={self.electrode_dict_L["k_coupl_x"]}_"
						f"kcoupl_xy={self.electrode_dict_L["k_coupl_xy"]}_DOS.dat", 
						(self.w, dos_L, dos_real_L, dos_R, dos_real_R, dos_L_cpld, dos_real_L_cpld, dos_R_cpld, dos_real_R_cpld),
						"w (sqrt(har/(bohr**2*u))), DOS_L (a.u.), DOS_L_real (a.u.), DOS_R (a.u.), DOS_R_real (a.u.), DOS_L_cpld (a.u.), DOS_L_real_cpld (a.u.), DOS_R_cpld (a.u.), DOS_R_real_cpld (a.u.)")
				
				top.write_plot_data(path_transdos + 
						f"/{self.sys_descr}___PT_elL={self.electrode_dict_L["type"]}_elR={self.electrode_dict_R["type"]}_"
						f"CC={self.scatter_dict["type"]}_intrange={self.electrode_L.interaction_range}_kcoupl_x={self.electrode_dict_L["k_coupl_x"]}_"
						f"kcoupl_xy={self.electrode_dict_L["k_coupl_xy"]}_DOS.dat", 
						(self.w, dos_L, dos_real_L, dos_R, dos_real_R, dos_L_cpld, dos_real_L_cpld, dos_R_cpld, dos_real_R_cpld),
						"w (sqrt(har/(bohr**2*u))), DOS_L (a.u.), DOS_L_real (a.u.), DOS_R (a.u.), DOS_R_real (a.u.), DOS_L_cpld (a.u.), DOS_L_real_cpld (a.u.), DOS_R_cpld (a.u.), DOS_R_real_cpld (a.u.)")
				
			except KeyError as e:
				top.write_plot_data(path_dos + 
						f"/{self.sys_descr}___PT_elL={self.electrode_dict_L["type"]}_elR={self.electrode_dict_R["type"]}_"
						f"CC={self.scatter_dict["type"]}_intrange={self.electrode_L.interaction_range}_kcoupl_x={self.electrode_dict_L["k_coupl_x"]}_DOS.dat",
						(self.w, dos_L, dos_real_L, dos_R, dos_real_R, dos_L_cpld, dos_real_L_cpld, dos_R_cpld, dos_real_R_cpld), 
						"w (sqrt(har/(bohr**2*u))), DOS_L (a.u.), DOS_L_real (a.u.), DOS_R (a.u.), DOS_R_real (a.u.), DOS_L_cpld (a.u.), DOS_L_real_cpld (a.u.), DOS_R_cpld (a.u.), DOS_R_real_cpld (a.u.)")
				
				top.write_plot_data(path_transdos + 
						f"/{self.sys_descr}___PT_elL={self.electrode_dict_L["type"]}_elR={self.electrode_dict_R["type"]}_"
						f"CC={self.scatter_dict["type"]}_intrange={self.electrode_L.interaction_range}_kcoupl_x={self.electrode_dict_L["k_coupl_x"]}_DOS.dat",
						(self.w, dos_L, dos_real_L, dos_R, dos_real_R, dos_L_cpld, dos_real_L_cpld, dos_R_cpld, dos_real_R_cpld), 
						"w (sqrt(har/(bohr**2*u))), DOS_L (a.u.), DOS_L_real (a.u.), DOS_R (a.u.), DOS_R_real (a.u.), DOS_L_cpld (a.u.), DOS_L_real_cpld (a.u.), DOS_R_cpld (a.u.), DOS_R_real_cpld (a.u.)")
			
		if plot_dos:
			fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
			fig.tight_layout()
			ax1.set_title('DOS Left electrode', fontproperties=prop)
			ax1.plot(self.w, dos_real_L_cpld, label=r'$\Re(d)$', color='blue', linestyle='--')
			ax1.plot(self.w, dos_L_cpld, label=r'$\Im(d)$', color='red')
			ax1.set_ylabel('DOS', fontsize=12, fontproperties=prop)
			#ax1.set_xlabel(r'Phonon Energy ($\mathrm{meV}$)', fontsize=12, fontproperties=prop)
			ax1.set_xticklabels(ax1.get_xticks(), fontproperties=prop)
			ax1.set_yticklabels(ax1.get_yticks(), fontproperties=prop)
			ax1.grid()
			ax1.legend(fontsize=12, prop=prop)

			ax2.set_title('DOS Right electrode', fontproperties=prop)
			ax2.plot(self.w, dos_real_R_cpld, label=r'$\Re(d)$', color='blue', linestyle='--')
			ax2.plot(self.w, dos_R_cpld, label=r'$\Im(d)$', color='red')
			ax2.set_ylabel('DOS', fontsize=12, fontproperties=prop)
			ax2.set_xlabel(r'Phonon Energy ($\mathrm{meV}$)', fontsize=12, fontproperties=prop)
			ax2.set_xticklabels(ax2.get_xticks(), fontproperties=prop)
			ax2.set_yticklabels(ax2.get_yticks(), fontproperties=prop)
			ax2.grid()
			ax2.legend(fontsize=12, prop=prop)

			try:
				plt.savefig(self.data_path + 
						f"/{self.sys_descr}___PT_elL={self.electrode_dict_L["type"]}_elR={self.electrode_dict_R["type"]}_"
						f"CC={self.scatter_dict["type"]}_intrange={self.electrode_L.interaction_range}_kcoupl_x={self.electrode_dict_L["k_coupl_x"]}_"
						f"kcoupl_xy={self.electrode_dict_L["k_coupl_xy"]}_DOS.pdf", 
						bbox_inches='tight')
				
			except KeyError as e:
				plt.savefig(self.data_path + 
						f"/{self.sys_descr}___PT_elL={self.electrode_dict_L["type"]}_elR={self.electrode_dict_R["type"]}_"
						f"CC={self.scatter_dict["type"]}_intrange={self.electrode_L.interaction_range}_kcoupl_x={self.electrode_dict_L["k_coupl_x"]}_DOS.pdf", 
						bbox_inches='tight')

	
if __name__ == '__main__':

    # Load the .json configuration file
    config_path = sys.argv[1]

    try:
        with open(config_path, 'r') as f:
             config = json.load(f)
    except FileNotFoundError:
        print(f"Configuration file '{config_path}' not found.")
        sys.exit(1)

	# Extract the enabled electrode
    for electrode in ["ELECTRODE_L", "ELECTRODE_R"]:
        for electrode_type, params in config[electrode].items():
            if params.get("enabled", False):  # check if enabled is true 
                
                if electrode == "ELECTRODE_L":
                    electrode_dict_L = params
                    electrode_dict_L["left"] = True
                    electrode_dict_L["right"] = False
                    electrode_dict_L["type"] = electrode_type
                    
                elif electrode == "ELECTRODE_R":
                    electrode_dict_R = params
                    electrode_dict_R["left"] = False
                    electrode_dict_R["right"] = True
                    electrode_dict_R["type"] = electrode_type

    if not (electrode_dict_L and electrode_dict_R):
        raise ValueError(f"No enabled electrode found in the configuration for {electrode}.")

    # Extract the enabled scatter object
    scatter_dict = None
    if "SCATTER" in config:
        for scatter_type, params in config["SCATTER"].items():
            if params.get("enabled", False):  # check if enabled is true
                scatter_dict = params
                scatter_dict["type"] = scatter_type
                break
    if not scatter_dict:
        raise ValueError("No enabled scatter object found in the configuration.")

    # General parameters
    data_path = config["CALCULATION"]["data_path"]
    sys_descr = config["CALCULATION"]["sys_descr"]
    E_D = config["CALCULATION"]["E_D"]
    M_E = config["CALCULATION"]["M_E"]
    M_R = config["CALCULATION"]["M_R"]
    M_C = config["CALCULATION"]["M_C"]
    N = config["CALCULATION"]["N"]
    T_min = config["CALCULATION"]["T_min"]
    T_max = config["CALCULATION"]["T_max"]
    kappa_grid_points = config["CALCULATION"]["kappa_grid_points"]

    # Initialize PhononTransort class object
    PT = PhononTransport(
        data_path = data_path,
        sys_descr = sys_descr,
        electrode_dict_L = electrode_dict_L,
		electrode_dict_R = electrode_dict_R,
        scatter_dict = scatter_dict,
        E_D = E_D,
        M_E = M_E,
		M_R = M_R,
        M_C = M_C,
        N = N,
        T_min = T_min,
        T_max = T_max,
        kappa_grid_points = kappa_grid_points
    )
    
    PT.plot_transport(plot_data=config["data_output"]["plot_transmission"])
    PT.plot_dos(plot_dos=config["data_output"]["plot_dos"])
        
    print("debug")