import copy
import os.path
import sys
import json

import numpy as np
from model_systems import * 
import electrode as el
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from matplotlib.ticker import FormatStrFormatter
import tmoutproc as top
import calculate_kappa as ck
from utils import eigenchannel_utils as eu
from utils import constants as const
import scienceplots

# does'nt work that well yet
matplotlib.rcParams['font.family'] = r'C://Users//sevke//Desktop//Dev//fonts//fira_sans//FiraSans-Regular.ttf'
#matplotlib.rcParams['mathtext.rm'] = r'C://Users//sevke//Desktop//Dev//fonts/fira_sans//FiraSans-Regular.ttf'
#matplotlib.rcParams['mathtext.it'] = r'C://Users//sevke//Desktop//Dev//fonts//fira_sans//FiraSans-Italic.ttf'
#matplotlib.rcParams['mathtext.bf'] = r'C://Users//sevke//Desktop//Dev//fonts//fira_sans//FiraSan-Bold.ttf'
prop = fm.FontProperties(fname=r'C://Users//sevke//Desktop//Dev//fonts//fira_sans//FiraSans-Regular.ttf')
plt.style.use(['science', 'notebook', 'no-latex'])


class PhononTransport:
	"""Class for phonon transport calculations

	This class can be used for phonon transport calculations using a decimation technique to set up the electrodes. 
	Also describes the hessian matrix of the center part ab initio.
	"""

	def __init__(self, data_path, electrode_dict_L, electrode_dict_R, scatter_dict, E_D, M_L, M_R, M_C, N, T_min, T_max, kappa_grid_points):
		"""
		Args:
			electrode_dict (dict): Dictionary containing the configuration of the enabled electrode.
			scatter_dict (dict): Dictionary containing the configuration of the enabled scatter object.
			E_D (float): Debeye energy in meV.
			M_L (str): Atom type in the reservoir.
			M_C (str): Atom type coupled to the reservoir.
			N (int): Number of grid points.
			T_min (float): Minimum temperature for thermal conductance calculation.
			T_max (float): Maximum temperature for thermal conductance calculation.
			kappa_grid_points (int): Number of grid points for thermal conductance.
		"""

		self.data_path = data_path
		self.electrode_dict_L = electrode_dict_L
		self.electrode_dict_R = electrode_dict_R
		self.scatter_dict = scatter_dict
		self.M_L = M_L
		self.M_C = M_C
		self.M_R = M_R
		self.N = N
		self.E_D = E_D

        # Convert to har * s / (bohr**2 * u)
		#self.w_D = (E_D * const.meV2J / const.h_bar) / const.unit2SI

		self.temperature = np.linspace(T_min, T_max, kappa_grid_points)
		#self.w = np.linspace(0, self.w_D * 1.1, N)
		#self.E = self.w * const.unit2SI * const.h_bar * const.J2meV

		#TODO did a change in starting point w here
		self.w = np.linspace(1E-3, self.E_D * 1.1, N) #new
		self.i = np.linspace(0, self.N, self.N, False, dtype=int)

		print("########## Setting up the scatter region ##########")
		self.scatter = self.__initialize_scatter(self.scatter_dict, self.electrode_dict_L, self.electrode_dict_R)
  
		print("########## Setting up the electrodes ##########")
		self.electrode_L = self.__initialize_electrode(self.electrode_dict_L)
		self.electrode_R = self.__initialize_electrode(self.electrode_dict_R)

		# Check for allowed combinations of electrode and scatter types
		if (self.electrode_dict_L["type"], self.electrode_dict_R["type"], self.scatter_dict["type"]) not in [
			("DebeyeModel", "DebeyeModel", "FiniteLattice2D"),
			("DebeyeModel", "DebeyeModel", "Chain1D"),
			("Ribbon2D", "Ribbon2D", "FiniteLattice2D"), #TODO: you can set it up to get a "2D" 1Dchain.
			("Ribbon2D", "Ribbon2D", "Chain1D"),
			("Chain1D", "Chain1D", "Chain1D"),
			("InfiniteFourier2D", "InfiniteFourier2D", "FiniteLattice2D"),
			("InfiniteFourier2D", "InfiniteFourier2D", "Chain1D")
		]:
			raise ValueError(f"Invalid combination of electrode type '{self.electrode_L.type}', '{self.electrode_R.type}' and scatter type '{self.scatter.type}'")

		self.D = self.scatter.hessian #* top.atom_weight(self.M_C) * (const.eV2hartree / const.ang2bohr ** 2)
  
		# Test for extended electrode partitioning
		'''self.D = np.array([[100, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                  [0, 100, 0, -100, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 200, 0, 0, 0, -100, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                  [0, -100, 0, 200, 0, -100, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 100, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, -100, 0, 100, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                  [0, 0, -100, 0, 0, 0, 200, 0, -100, 0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, -100, 0, 200, 0, 0, 0, -100, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 100, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 100, 0, -100, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0, -100, 0, 0, 0, 200, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -100, 0, 200, 0, -100],
                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 100, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -100, 0, 100]])'''
                  
		'''self.D = np.array([[ 200.5,    0., -100.,    0.],
							[   0.,    0.,    0.,    0.],
							[-100.,    0.,  200.5,    0.],
							[   0.,    0.,    0.,    0.]])'''
                  
		self.sigma_L, self.sigma_R = self.calculate_sigma()
		self.g_CC_ret, self.g_CC_adv = self.calculate_G_cc()
		self.T = self.calculate_transmission()
		self.kappa = self.calc_kappa()
	
	def __initialize_electrode(self, electrode_dict):
		"""
		Initializes the electrode based on the provided configuration.

		Args:
			electrode_dict (dict): Dictionary containing the electrode configuration.

		Returns:
			Electrode: Initialized electrode object.
		"""
		
		match electrode_dict["type"]:

			case "DebeyeModel":
				return el.DebeyeModel(
					self.w,
					k_c = electrode_dict["k_x"],
					w_D = self.E_D #/ const.h_bar
				)
			
			case "Chain1D":
				return el.Chain1D(
					self.w,
					interaction_range = electrode_dict["interaction_range"],
					interact_potential = electrode_dict["interact_potential"],
     				atom_type = electrode_dict["atom_type"],
					lattice_constant=electrode_dict["lattice_constant"],
     				k_x = electrode_dict["k_x"],
					k_c = electrode_dict["k_c"]
				)
			
			case "Ribbon2D":
				return el.Ribbon2D(
					self.w,
					interaction_range = electrode_dict["interaction_range"],
					interact_potential = electrode_dict["interact_potential"],
					atom_type = electrode_dict["atom_type"],
					lattice_constant=electrode_dict["lattice_constant"],
					N_y = electrode_dict["N_y"],
					N_y_scatter = self.scatter.N_y,
					M_L = self.M_L,
					M_C = self.M_C,
					k_x = electrode_dict["k_x"],
					k_y = electrode_dict["k_y"],
					k_xy = electrode_dict["k_xy"],
					k_c = electrode_dict["k_c"],
					k_c_xy = electrode_dict["k_c_xy"]
				)
			
			case "InfiniteFourier2D":
				return el.InfiniteFourier2D(
					self.w,
					interaction_range = electrode_dict["interaction_range"],
					interact_potential = electrode_dict["interact_potential"],
					atom_type = electrode_dict["atom_type"],
					lattice_constant=electrode_dict["lattice_constant"],
					N_q = electrode_dict["N_q"],
					k_x = electrode_dict["k_x"],
					k_y = electrode_dict["k_y"],
					k_xy = electrode_dict["k_xy"],
					k_c = electrode_dict["k_c"],
					k_c_xy = electrode_dict["k_c_xy"],
					N_y_scatter = self.scatter.N_y
				)
	
			case _:
				raise ValueError(f"Unsupported electrode type: {electrode_dict['type']}")

	def __initialize_scatter(self, scatter_dict, electrode_dict_l, electrode_dict_r):
		"""
		Initializes the scatter object based on the provided configuration.

		Args:
			scatter_dict (dict): Dictionary containing the scatter configuration.

		Returns:
			Scatter: Initialized scatter object.
		"""
		match scatter_dict["type"]:

			case "FiniteLattice2D":
				return FiniteLattice2D(
					N_y = scatter_dict["N_y"],
					N_x = scatter_dict["N_x"],
					k_l_x = electrode_dict_l["k_x"],
					k_c_x = scatter_dict["k_x"],
					k_r_x = electrode_dict_r["k_x"],
					k_c_y = scatter_dict["k_y"],
					k_c_xy = scatter_dict["k_xy"],
					k_l_xy = electrode_dict_l["k_xy"],
					k_r_xy = electrode_dict_r["k_xy"],
					interact_potential = scatter_dict["interact_potential"],
					interaction_range = scatter_dict["interaction_range"],
					lattice_constant = scatter_dict["lattice_constant"],
					atom_type = scatter_dict["atom_type"]
				)
    
			case "Chain1D":
				return Chain1D(
					k_c = scatter_dict["k_x"],
					k_l = electrode_dict_l["k_x"],
					k_r = electrode_dict_r["k_x"],
					interact_potential = scatter_dict["interact_potential"],
					interaction_range = scatter_dict["interaction_range"],
					lattice_constant = scatter_dict["lattice_constant"],
					atom_type = scatter_dict["atom_type"],
					N = scatter_dict["N"]
				)
			
			case _:
				raise ValueError(f"Unsupported scatter type: {scatter_dict['type']}")

	def calculate_sigma(self):
		"""Calculates self energy according to: 
		First-principles calculation of the thermoelectric figure of merit for [2,2]paracyclophane-based single-molecule junctions. PHYSICAL REVIEW B 91, 165419 (2015)

		Args:
			self: self object

		Returns:
			sigma (array_like): self energy 
		"""
		#extend to electrode L, R
		match (self.electrode_dict_L["type"], self.electrode_dict_R["type"]):

			case ("DebeyeModel", "DebeyeModel"):
				# Scalar Greens function
				g_L = self.electrode_L.g
				g_R = self.electrode_R.g

				k_c_l = self.electrode_L.k_c #* (1 / np.sqrt(top.atom_weight(self.M_C) * top.atom_weight(self.M_L)))
				k_c_r = self.electrode_R.k_c #* (1 / np.sqrt(top.atom_weight(self.M_C) * top.atom_weight(self.M_R)))
							
				'''if self.scatter_dict["type"] == "Chain1D":
					k_c_l = np.zeros((self.electrode_L.interaction_range, self.electrode_L.interaction_range), dtype=float) #* (1 / np.sqrt(top.atom_weight(self.M_C) * top.atom_weight(self.M_L)))
					k_c_r = np.zeros((self.electrode_R.interaction_range, self.electrode_R.interaction_range), dtype=float) #* (1 / np.sqrt(top.atom_weight(self.M_C) * top.atom_weight(self.M_R)))

					all_k_c_x_L = self.electrode_L.ranged_force_constant()[0]
					all_k_c_x_R = self.electrode_R.ranged_force_constant()[0]

					# Indices can be taken as atom numbers since it's 1D, i is electrode, j is scatter
					for i in range(1, self.electrode_L.interaction_range + 1):
						for j in range(1, self.electrode_L.interaction_range + 1):
							if i >= j:
								try:
									k_c_l[i-1, j-1] = all_k_c_x_L[-(i - j + 1)][1]
								except TypeError:
									k_c_l[i-1, j-1] = all_k_c_x_L[-(i - j + 1)]
							
					for i in range(1, self.electrode_R.interaction_range + 1):
						for j in range(1, self.electrode_R.interaction_range + 1):
							if i >= j:
								try:
									k_c_r[i-1, j-1] = all_k_c_x_R[-(i - j + 1)][1]
								except TypeError:
									k_c_r[i-1, j-1] = all_k_c_x_R[-(i - j + 1)]'''
							
							
			case ("Chain1D", "Chain1D"):
				#1D Jan PhD Thesis p.21 (analytical solution)

				k_x = sum(self.electrode_L.ranged_force_constant()[0][i][1] for i in range(self.electrode_L.interaction_range))
				f_E = 0.5 * (self.w**2 - 2 * k_x - self.w * np.sqrt(self.w**2 - 4 * k_x, dtype=complex)) 

				sigma_L = np.zeros((self.N, self.scatter.N, self.scatter.N), dtype=complex)
				sigma_R = np.zeros((self.N, self.scatter.N, self.scatter.N), dtype=complex)

				for i in range(self.N):
    
					sigma_L[i, 0, 0] = f_E[i] 
					sigma_R[i, -1, -1] = f_E[i]
     
				return sigma_L, sigma_R
    			
			case ("Ribbon2D", "Ribbon2D"):
				#2D
				g_L = self.electrode_L.g
				g_R = self.electrode_R.g

				direct_interaction_L = self.electrode_L.direct_interaction
				direct_interaction_R = self.electrode_R.direct_interaction

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
				all_k_c_x_L = self.electrode_L.ranged_force_constant()[3]
				all_k_c_x_R = self.electrode_R.ranged_force_constant()[3]
				all_k_c_xy_L = self.electrode_L.ranged_force_constant()[4]
				all_k_c_xy_R = self.electrode_R.ranged_force_constant()[4]
    
				#set up LC interaction matrix		
				atomnr_el = 0
				
				for i in range(interaction_range_L):
					for at_el in range(1, N_y_L + 1):
						atomnr_el += 1
      
						if ((N_y_L - N_y_scatter) // 2) <= at_el <= ((N_y_L - N_y_scatter) // 2) + N_y_scatter + 1:
		
							atomnr_sc = 0
							for j in range(i + 1):
								for at_sc in range(1, N_y_scatter + 1):
									atomnr_sc += 1
									# look if sc and el are aligned 
									if at_el == at_sc + ((N_y_L - N_y_scatter) // 2):
										# coupling in x-direction
										k_LC_temp[2 * (atomnr_el - 1), 2 * (atomnr_sc - 1)] = -all_k_c_x_L[-(j + 1)][1]
		
									if i == interaction_range_L - 1:
										if (at_el == ((N_y_L - N_y_scatter) // 2) and at_sc == 1) or (at_el == ((N_y_L - N_y_scatter) // 2) + N_y_scatter + 1 and at_sc == N_y_scatter):
											# coupling also xy
											k_LC_temp[2 * (atomnr_el - 1), 2 * (at_sc - 1) + 1] = -all_k_c_xy_L[0][1]
											k_LC_temp[2 * (atomnr_el - 1) + 1, 2 * (at_sc - 1)] = -all_k_c_xy_L[0][1]

										if at_el == at_sc + ((N_y_L - N_y_scatter) // 2) and N_y_scatter > 1:
           
											if at_sc == 1:
												# coupling also xy-direction
												k_LC_temp[2 * (atomnr_el - 1) + 1, 2 * at_sc] = -all_k_c_xy_L[0][1]
												k_LC_temp[2 * (atomnr_el - 1), 2 * at_sc + 1] = -all_k_c_xy_L[0][1]
											elif at_sc == N_y_scatter:
												# coupling also xy-direction
												k_LC_temp[2 * (atomnr_el - 1) + 1, 2 * (at_sc - 2)] = -all_k_c_xy_L[0][1]
												k_LC_temp[2 * (atomnr_el - 1), 2 * (at_sc - 2) + 1] = -all_k_c_xy_L[0][1]
										
            
				atomnr_el = 0
				
				for i in range(interaction_range_R):
					for at_el in range(1, N_y_R + 1):
						atomnr_el += 1
      
						if ((N_y_R - N_y_scatter) // 2) <= at_el <= ((N_y_R - N_y_scatter) // 2) + N_y_scatter + 1:
		
							atomnr_sc = 0
							for j in range(i + 1):
								for at_sc in range(1, N_y_scatter + 1):
									atomnr_sc += 1
									# look if sc and el are aligned 
									if at_el == at_sc + ((N_y_R - N_y_scatter) // 2):
										# coupling in x-direction
										k_RC_temp[2 * (atomnr_el - 1), 2 * (atomnr_sc - 1)] = -all_k_c_x_R[-(j + 1)][1]
		
									if i == interaction_range_R - 1:
										if (at_el == ((N_y_R - N_y_scatter) // 2) and at_sc == 1) or (at_el == ((N_y_R - N_y_scatter) // 2) + N_y_scatter + 1 and at_sc == N_y_scatter):
											# coupling also xy
											k_RC_temp[2 * (atomnr_el - 1), 2 * (at_sc - 1) + 1] = -all_k_c_xy_R[0][1]
											k_RC_temp[2 * (atomnr_el - 1) + 1, 2 * (at_sc - 1)] = -all_k_c_xy_R[0][1]

										if at_el == at_sc + ((N_y_R - N_y_scatter) // 2) and N_y_scatter > 1:
           
											if at_sc == 1:
												# coupling also xy-direction
												k_RC_temp[2 * (atomnr_el - 1) + 1, 2 * at_sc] = -all_k_c_xy_R[0][1]
												k_RC_temp[2 * (atomnr_el - 1), 2 * at_sc + 1] = -all_k_c_xy_R[0][1]
											elif at_sc == N_y_scatter:
												# coupling also xy-direction
												k_RC_temp[2 * (atomnr_el - 1) + 1, 2 * (at_sc - 2)] = -all_k_c_xy_R[0][1]
												k_RC_temp[2 * (atomnr_el - 1), 2 * (at_sc - 2) + 1] = -all_k_c_xy_R[0][1]

				# k_LC fill from left top
				k_LC[:k_LC_temp.shape[0], :k_LC_temp.shape[1]] = k_LC_temp
				# k_RC fill from bottom right
				k_RC[-k_RC_temp.shape[0]:, -k_RC_temp.shape[1]:] = k_RC_temp

				'''# Test for extendet electrode partitioning
    			k_x_l = all_k_c_x_L[0][1]
				k_x_r = all_k_c_x_R[0][1]
				k_xy_l = all_k_c_xy_L[0][1]
				k_xy_r = all_k_c_xy_R[0][1]

				k_LC = np.array([[-k_x_l, 0, 0, -k_xy_l, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
								[0, 0, -k_xy_l, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
								[0, -k_xy_l, -k_x_l, 0, 0, -k_xy_l, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
								[-k_xy_l, 0, 0, 0, -k_xy_l, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
								[0, 0, 0, -k_xy_l, -k_x_l, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
								[0, 0, -k_xy_l, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
								])
				k_RC = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -k_x_r, 0, 0, -k_xy_r, 0, 0],
								[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -k_xy_r, 0, 0, 0],
								[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -k_xy_r, -k_x_r, 0, 0, -k_xy_r],
								[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -k_xy_r, 0, 0, 0, -k_xy_r, 0],
								[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -k_xy_r, -k_x_r, 0],
								[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -k_xy_r, 0, 0, 0]
								])'''

			case ("InfiniteFourier2D", "InfiniteFourier2D"):

				g_L = self.electrode_L.g
				g_R = self.electrode_R.g

				k_LC = self.electrode_L.k_lc_LL
				k_RC = self.electrode_R.k_lc_LL

		# Initialize sigma array with the same shape as g
		sigma_L = np.zeros((self.N, self.D.shape[0], self.D.shape[1]), dtype=complex)
		sigma_R = np.zeros((self.N, self.D.shape[0], self.D.shape[1]), dtype=complex)

		# Build sigma matrix for each frequency depending on the (allowed) electrode model configuration
		# The 1D Chain transmission has an analytical expression an is covered directly there.
		
		# DebeyeModel (Markussen)
		if (g_L.shape, g_R.shape) == ((self.N,), (self.N,)): # and self.electrode_L.interaction_range > 1 and self.electrode_R.interaction_range > 1: 

			sigma_nu_l = np.array(list(map(lambda i: k_c_l**2 * g_L[i], self.i)))
			sigma_nu_r = np.array(list(map(lambda i: k_c_r**2 * g_R[i], self.i)))

			match self.scatter_dict["type"]:
			
				case "FiniteLattice2D":
					# get N_y scatter and set sigma_nu on the positions of coupling atoms
					for k in range(self.N):
						for n in range(self.scatter.N_y):
							sigma_L[k][n * 2, n * 2] = sigma_nu_l[k]
							sigma_R[k][sigma_R.shape[1] - 2 - n * 2, sigma_R.shape[2] - 2 - n * 2] = sigma_nu_r[k]
				
				case "Chain1D":

					for k in range(self.N):
						#sigma_L[k] = np.dot(np.dot(k_c_l.T, g_L[k]), k_c_l)
						#sigma_R[k] = np.dot(np.dot(k_c_r.T, g_R[k]), k_c_r)
						sigma_L[k][0, 0] = sigma_nu_l[k]
						sigma_R[k][-1, -1] = sigma_nu_r[k]
						
  
		# 2D case (decimation technique)
		elif (electrode_dict_L["type"], electrode_dict_R["type"]) == ("Ribbon2D", "Ribbon2D") and \
			(g_L.shape, g_R.shape) == ((self.N, 2 * self.electrode_L.interaction_range * self.electrode_L.N_y, 2 * self.electrode_L.interaction_range * self.electrode_L.N_y), \
			(self.N, 2 * self.electrode_R.interaction_range * self.electrode_R.N_y, 2 * self.electrode_R.interaction_range * self.electrode_R.N_y)):

			#TODO: check if this is correct >> k_LC dimension
			#sigma_L_temp = np.array(list(map(lambda i: np.dot(np.dot(k_LC.T, g_L[i]), k_LC), self.i)))
			#sigma_R_temp = np.array(list(map(lambda i: np.dot(np.dot(k_RC.T, g_R[i]), k_RC), self.i)))
			
			#sigma_L_temp = np.array(list(map(lambda i: np.dot(np.dot(self.electrode_L.k_lc_LL.T, g_L[i]), self.electrode_L.k_lc_LL), self.i)))
			#sigma_R_temp = np.array(list(map(lambda i: np.dot(np.dot(self.electrode_R.k_lc_LL.T, g_R[i]), self.electrode_R.k_lc_LL), self.i)))
			sigma_L = np.array(list(map(lambda i: np.dot(np.dot(k_LC.T, g_L[i]), k_LC), self.i)))
			sigma_R = np.array(list(map(lambda i: np.dot(np.dot(k_RC.T, g_R[i]), k_RC), self.i)))
			# TODO: FIX
			"""for i in range(self.N):
				sigma_L[i, 0: sigma_L_temp.shape[1], 0: sigma_L_temp.shape[2]] = sigma_L_temp[i]
				sigma_R[i, sigma_R.shape[1] - sigma_R_temp.shape[1]: sigma_R.shape[1], sigma_R.shape[2] - sigma_R_temp.shape[2]: sigma_R.shape[2]] = sigma_R_temp[i]"""

		# 3D case (decimation technique) #TODO: Implementation
		elif (electrode_dict_L["type"], electrode_dict_R["type"]) == ("Ribbon3D", "Ribbon3D") and \
			(g_L.shape, g_R.shape) == ((self.N, 3 * self.electrode_L.interaction_range * self.electrode_L.N_y, 3 * self.electrode_L.interaction_range * self.electrode_L.N_y), \
			(self.N, 3 * self.electrode_R.interaction_range * self.electrode_R.N_y, 3 * self.electrode_R.interaction_range * self.electrode_R.N_y)):
		
			sigma_L = np.array(list(map(lambda i: np.dot(np.dot(k_LC.T, g_L[i]), k_LC), self.i)))
			sigma_R = np.array(list(map(lambda i: np.dot(np.dot(k_RC.T, g_R[i]), k_RC), self.i)))

		# InfitineFourier2D case TODO: dimensionality problem due to xy coupling -> dim(D) == dim(sigmaL,R) ggf. < dim(g_L,R)
		elif (electrode_dict_L["type"], electrode_dict_R["type"]) == ("InfiniteFourier2D", "InfiniteFourier2D") and \
			(g_L.shape, g_R.shape) == ((self.N, 2 * (self.scatter.N_y + 2), 2 * (self.scatter.N_y + 2)), (self.N, 2 * (self.scatter.N_y + 2), 2 * (self.scatter.N_y + 2))):

			sigma_L_temp = np.array(list(map(lambda i: np.dot(np.dot(k_LC.T, g_L[i]), k_LC), self.i)))
			sigma_R_temp = np.array(list(map(lambda i: np.dot(np.dot(k_RC.T, g_R[i]), k_RC), self.i)))
			
			#sigma_L_temp = np.array(list(map(lambda i: np.dot(np.dot(self.electrode_L.k_lc_LL.T, g_L[i]), self.electrode_L.k_lc_LL), self.i)))
			#sigma_R_temp = np.array(list(map(lambda i: np.dot(np.dot(self.electrode_R.k_lc_LL.T, g_R[i]), self.electrode_R.k_lc_LL), self.i)))

			for i in range(self.N):
				sigma_L[i, 0: sigma_L_temp.shape[1], 0: sigma_L_temp.shape[2]] = sigma_L_temp[i]
				sigma_R[i, sigma_R.shape[1] - sigma_R_temp.shape[1]: sigma_R.shape[1], sigma_R.shape[2] - sigma_R_temp.shape[2]: sigma_R.shape[2]] = sigma_R_temp[i]

		else:
			raise ValueError(f"Unsupported shape for g_L: {g_L.shape} or g_R: {g_R.shape}")

		return sigma_L, sigma_R

	def calculate_G_cc(self):
		"""Calculates Greens function for the central with given parameters at given frequency w.

		Args:
			self: self object

		Returns:
			g_cc (np.ndarray): Greens function for the central part
		"""

		g_CC_ret = np.array(list(map(lambda i: np.linalg.inv((self.w[i] + 1E-16j)**2 * np.identity(self.D.shape[0]) - self.D - self.sigma_L[i] - self.sigma_R[i]), self.i)))
		g_CC_adv = np.transpose(np.conj(g_CC_ret), (0, 2, 1))

		return g_CC_ret, g_CC_adv	

	def calculate_transmission(self):
		"""Calculates the transmission for the given parameters at given frequency w.

		Args:
			self: self object

		Returns:
			T (np.ndarray): Transmission
		"""
		if self.electrode_dict_L["type"] == "Chain1D" and self.electrode_dict_R["type"] == "Chain1D" and scatter_dict["type"] == "Chain1D":
			# 1D Chain transmission has an analytical expression
			k_x_L = sum(self.electrode_L.ranged_force_constant()[0][i][1] for i in range(self.electrode_L.interaction_range))
			k_x_R = sum(self.electrode_R.ranged_force_constant()[0][i][1] for i in range(self.electrode_R.interaction_range))
		
			g_E_L = np.where(4 * k_x_L - self.w**2 >= 0, self.w * np.sqrt(4 * k_x_L - self.w**2), 0)
			g_E_R = np.where(4 * k_x_R - self.w**2 >= 0, self.w * np.sqrt(4 * k_x_R - self.w**2), 0)
			
			lambda_L = np.zeros((self.N, self.scatter.N, self.scatter.N), dtype=complex)
			lambda_R = np.zeros((self.N, self.scatter.N, self.scatter.N), dtype=complex)
   
			for i in range(self.N):
				lambda_L[i, 0, 0] = g_E_L[i]
				lambda_R[i, -1, -1] = g_E_R[i]

			trans_prob_matrix = np.array(list(map(lambda i: np.dot(np.dot(self.g_CC_ret[i], lambda_L[i]), np.dot(self.g_CC_adv[i], lambda_R[i])), self.i)))
			tau_ph = np.array(list(map(lambda i: np.real(np.trace(trans_prob_matrix[i])), self.i)))
				
			return tau_ph
  
		#spectral_dens_L = -2 * np.imag(self.sigma_L)
		spectral_dens_L = 1j * (self.sigma_L - np.transpose(np.conj(self.sigma_L), (0, 2, 1)))
		#spectral_dens_R = -2 * np.imag(self.sigma_R)
		spectral_dens_R = 1j * (self.sigma_R - np.transpose(np.conj(self.sigma_R), (0, 2, 1)))

		trans_prob_matrix = np.array(list(map(lambda i: np.dot(np.dot(self.g_CC_ret[i], spectral_dens_L[i]), np.dot(self.g_CC_adv[i], spectral_dens_R[i])), self.i)))
  
		tau_ph = np.array(list(map(lambda i: np.real(np.trace(trans_prob_matrix[i])), self.i)))
  
		return tau_ph

	def calc_kappa(self):
		"""Calculates the thermal conductance.

		Returns:
			np.ndarray: array of thermal conductance values for each temperature in self.temperature.
		"""
		kappa = list()
  
		# w to SI
		w_kappa = self.w * const.unit2SI
		E = const.h_bar * w_kappa

		# joule to hartree
		E = E / const.har2J

		for j in range(0, len(self.temperature)):
			kappa.append(ck.calculate_kappa(self.T[1:len(self.T)], E[1:len(E)], self.temperature[j]) * const.har2pJ)
		
		return kappa

	def	plot_transport(self, write_data=True):
		"""Writes out the raw data and plots the transport properties of the system."""

		if write_data:
			try:
				top.write_plot_data(self.data_path + f"/PT_elL={self.electrode_dict_L["type"]}_elR={self.electrode_dict_R["type"]}_CC={self.scatter_dict["type"]}_kc={self.scatter_dict["k_x"]}_kc_xy={self.scatter_dict["k_xy"]}.dat", (self.w, self.T), "w (sqrt(har/(bohr**2*u))), T_vals")
			except KeyError as e:
				top.write_plot_data(self.data_path + f"/PT_elL={self.electrode_dict_L["type"]}_elR={self.electrode_dict_R["type"]}_CC={self.scatter_dict["type"]}_kc={self.scatter_dict["k_x"]}.dat", (self.w, self.T), "w (sqrt(har/(bohr**2*u))), T_vals")
			#top.write_plot_data(self.data_path + f"/debye.dat", (self.w, self.T), "w (sqrt(har/(bohr**2*u))), T_vals")
			top.write_plot_data(self.data_path + "/kappa.dat", (self.temperature, self.kappa), "T (K), kappa (pW/K)")

		print(f'TauMax = {max(self.T)}, TauMin = {min(self.T)}, T_0 = {self.T[0]}')
		print(f'KappaMax = {max(self.kappa)}, KappaMin = {min(self.kappa)}')
		#print(max(self.E), min(self.E))
		#fig, (ax1, ax2) = plt.subplots(2, 1)
		fig, ax1 = plt.subplots(1, 1)
		fig.tight_layout()
		#ax1.plot(self.E, self.T)
		ax1.plot(self.w, self.T)
		#ax1.set_yscale('log')
		ax1.set_xlabel(r'Phonon Energy ($\mathrm{meV}$)', fontsize=12, fontproperties=prop)
		ax1.set_ylabel(r'$\tau_{\mathrm{ph}}$', fontsize=12, fontproperties=prop)
		#ax1.axvline(self.w_D * const.unit2SI * const.h_bar / const.meV2J, ls="--", color="black")
		ax1.axhline(1, ls="--", color="black")
		#ax1.set_ylim(0, 1.5)
		#ax1.set_ylim(0, 4)
		ax1.set_xlim(0, 0.5 * E_D)
		ax1.set_xticklabels(ax1.get_xticks(), fontproperties=prop)
		ax1.set_yticklabels(ax1.get_yticks(), fontproperties=prop)
		ax1.xaxis.set_major_formatter(FormatStrFormatter('%.1f'))
		ax1.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
		ax1.grid()
  
		"""ax2.plot(self.temperature, self.kappa)
		ax2.set_xlabel('Temperature ($K$)', fontsize=12, fontproperties=prop)
		ax2.set_ylabel(r'$\kappa_{\mathrm{ph}}\;(\mathrm{pw/K})$', fontsize=12, fontproperties=prop)
		ax2.grid()
		ax2.set_xticklabels(ax1.get_xticks(), fontproperties=prop)
		ax2.set_yticklabels(ax1.get_yticks(), fontproperties=prop)
		ax2.xaxis.set_major_formatter(FormatStrFormatter('%.1f'))
		ax2.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))"""
     
		plt.rc('xtick', labelsize=12)
		plt.rc('ytick', labelsize=12)
		plt.xticks(fontproperties=prop)
		plt.yticks(fontproperties=prop)
  
		try:
			plt.savefig(self.data_path + f"/PT_elL={self.electrode_dict_L["type"]}_elR={self.electrode_dict_R["type"]}_CC={self.scatter_dict["type"]}_intrange={self.electrode_L.interaction_range}_kc={self.scatter_dict["k_x"]}_kc_xy={self.scatter_dict["k_xy"]}.pdf", bbox_inches='tight')
		except KeyError as e:
			plt.savefig(self.data_path + f"/PT_elL={self.electrode_dict_L["type"]}_elR={self.electrode_dict_R["type"]}_CC={self.scatter_dict["type"]}_intrange={self.electrode_L.interaction_range}_kc={self.scatter_dict["k_x"]}.pdf", bbox_inches='tight')
		#plt.savefig(self.data_path + f"/debye.pdf", bbox_inches='tight')
		plt.clf()

	def plot_dos(self, write_data=True):
		"""Plots the density of states (DOS) for the left and right electrode.
		Args:
			write_data (bool): If True, writes the DOS data to a file.
		"""
  		
		dos_L = self.electrode_L.dos
		dos_real_L = self.electrode_L.dos_real
		dos_R = self.electrode_R.dos
		dos_real_R = self.electrode_R.dos_real
  
		print(f'DOS Left electrode: {max(dos_L)}, {min(dos_L)}')
		print(f'DOS Right electrode: {max(dos_R)}, {min(dos_R)}')
  
		if write_data:
			try:
				top.write_plot_data(self.data_path + f"/PT_elL={self.electrode_dict_L["type"]}_elR={self.electrode_dict_R["type"]}_CC={self.scatter_dict["type"]}_kc={self.scatter_dict["k_x"]}_kc_xy={self.scatter_dict["k_xy"]}_DOS.dat", (self.w, dos_L, dos_real_L, dos_R, dos_real_R), "w (sqrt(har/(bohr**2*u))), DOS_L (a.u.), DOS_L_real (a.u.), DOS_R (a.u.), DOS_R_real (a.u.)")
			except KeyError as e:	
				top.write_plot_data(self.data_path + f"/PT_elL={self.electrode_dict_L["type"]}_elR={self.electrode_dict_R["type"]}_CC={self.scatter_dict["type"]}_kc={self.scatter_dict["k_x"]}_DOS.dat", (self.w, dos_L, dos_real_L, dos_R, dos_real_R), "w (sqrt(har/(bohr**2*u))), DOS_L (a.u.), DOS_L_real (a.u.), DOS_R (a.u.), DOS_R_real (a.u.)")

		fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
		fig.tight_layout()
		ax1.set_title('DOS Left electrode', fontproperties=prop)
		ax1.plot(self.w, dos_real_L, label=r'$\Re(d)$', color='red', linestyle='--')
		ax1.plot(self.w, dos_L, label=r'$\Im(d)$', color='blue')
		ax1.set_ylabel('DOS', fontsize=12, fontproperties=prop)
		#ax1.set_xlabel(r'Phonon Energy ($\mathrm{meV}$)', fontsize=12, fontproperties=prop)
		ax1.set_xticklabels(ax1.get_xticks(), fontproperties=prop)
		ax1.set_yticklabels(ax1.get_yticks(), fontproperties=prop)
		'''ax1.xaxis.set_major_formatter(FormatStrFormatter('%.1f'))
		ax1.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))'''
		#ax1.set_xlim(0, self.E_D)
		ax1.grid()
		ax1.legend(fontsize=12, prop=prop)

		ax2.set_title('DOS Right electrode', fontproperties=prop)
		ax2.plot(self.w, dos_real_R, label=r'$\Re(d)$', color='red', linestyle='--')
		ax2.plot(self.w, dos_R, label=r'$\Im(d)$', color='blue')
		ax2.set_ylabel('DOS', fontsize=12, fontproperties=prop)
		ax2.set_xlabel(r'Phonon Energy ($\mathrm{meV}$)', fontsize=12, fontproperties=prop)
		ax2.set_xticklabels(ax2.get_xticks(), fontproperties=prop)
		ax2.set_yticklabels(ax2.get_yticks(), fontproperties=prop)
		'''ax2.xaxis.set_major_formatter(FormatStrFormatter('%.1f'))
		ax2.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))'''
		#ax2.set_xlim(0, self.E_D)
		ax2.grid()
		ax2.legend(fontsize=12, prop=prop)
  
		'''plt.rc('xtick', labelsize=12)
		plt.rc('ytick', labelsize=12)
		plt.xticks(fontproperties=prop)
		plt.yticks(fontproperties=prop)'''

		try:
			plt.savefig(self.data_path + f"/PT_elL={self.electrode_dict_L["type"]}_elR={self.electrode_dict_R["type"]}_CC={self.scatter_dict["type"]}_intrange={self.electrode_L.interaction_range}_kc={self.scatter_dict["k_x"]}_kc_xy={self.scatter_dict["k_xy"]}_DOS.pdf", bbox_inches='tight')
		except KeyError as e:
			plt.savefig(self.data_path + f"/PT_elL={self.electrode_dict_L["type"]}_elR={self.electrode_dict_R["type"]}_CC={self.scatter_dict["type"]}_intrange={self.electrode_L.interaction_range}_kc={self.scatter_dict["k_x"]}_DOS.pdf", bbox_inches='tight')

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
                    electrode_dict_L["type"] = electrode_type
                    
                elif electrode == "ELECTRODE_R":
                    electrode_dict_R = params
                    electrode_dict_R["type"] = electrode_type

    if not (electrode_dict_L and electrode_dict_R):
        raise ValueError(f"No enabled electrode found in the configuration for {electrode}.")

    # Extract the enabled scatter object
    scatter_dict = None
    if "SCATTER" in config:
        for scatter_type, params in config["SCATTER"].items():
            if params.get("enabled", False):  # Prüfe auf 'enabled: true'
                scatter_dict = params
                scatter_dict["type"] = scatter_type
                break
    if not scatter_dict:
        raise ValueError("No enabled scatter object found in the configuration.")

    # General parameters
    data_path = config["CALCULATION"]["data_path"]
    E_D = config["CALCULATION"]["E_D"]
    M_L = config["CALCULATION"]["M_L"]
    M_R = config["CALCULATION"]["M_R"]
    M_C = config["CALCULATION"]["M_C"]
    N = config["CALCULATION"]["N"]
    T_min = config["CALCULATION"]["T_min"]
    T_max = config["CALCULATION"]["T_max"]
    kappa_grid_points = config["CALCULATION"]["kappa_grid_points"]

    # Initialize PhononTransort class object
    PT = PhononTransport(
        data_path = data_path,
        electrode_dict_L = electrode_dict_L,
		electrode_dict_R = electrode_dict_R,
        scatter_dict = scatter_dict,
        E_D = E_D,
        M_L = M_L,
		M_R = M_R,
        M_C = M_C,
        N = N,
        T_min = T_min,
        T_max = T_max,
        kappa_grid_points = kappa_grid_points
    )
    
    if config["data_output"]["plot_transmission"]:
        PT.plot_transport()
        
    print("debug")
    
    

    
