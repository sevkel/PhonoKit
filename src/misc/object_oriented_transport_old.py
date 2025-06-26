import copy
import os.path
import sys
import json

import numpy as np
from model_systems_old import * 
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
matplotlib.rcParams['mathtext.fontset'] = 'custom'
matplotlib.rcParams['font.family'] = '/hpc/gpfs2/scratch/u/kellerse/Masterarbeit/fonts/fira_sans/FiraSans Regular.ttf'
matplotlib.rcParams['mathtext.rm'] = '/hpc/gpfs2/scratch/u/kellerse/Masterarbeit/fonts/fira_sansFiraSans Regular.ttf'
matplotlib.rcParams['mathtext.it'] = '/hpc/gpfs2/scratch/u/kellerse/Masterarbeit/fonts/fira_sansFiraSans Italic.ttf'
matplotlib.rcParams['mathtext.bf'] = '/hpc/gpfs2/scratch/u/kellerse/Masterarbeit/fonts/fira_sansFiraSan Bold.ttf'
prop = fm.FontProperties(fname='/hpc/gpfs2/scratch/u/kellerse/Masterarbeit/fonts/fira_sans/FiraSans Regular.ttf')
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
  
		self.w = np.linspace(1E-10, self.E_D * 1.1, N) #new
		self.i = np.linspace(0, self.N, self.N, False, dtype=int)
		
		print("########## Setting up the scatter region ##########")
		self.scatter = self.__initialize_scatter(self.scatter_dict)
  
		print("########## Setting up the electrodes ##########")
		self.electrode_L = self.__initialize_electrode(self.electrode_dict_L)
		self.electrode_R = self.__initialize_electrode(self.electrode_dict_R)

		# Check for allowed combinations of electrode and scatter types
		if (self.electrode_dict_L["type"], self.electrode_dict_R["type"], self.scatter_dict["type"]) not in [
			("DebeyeModel", "DebeyeModel", "FiniteLattice2D"),
			("DebeyeModel", "DebeyeModel", "Chain1D"),
			("Ribbon2D", "Ribbon2D", "FiniteLattice2D"), #TODO: you can set it up to get a "2D" 1Dchain.
			("Chain1D", "Chain1D", "Chain1D")
		]:
			raise ValueError(f"Invalid combination of electrode type '{self.electrode_L.type}', '{self.electrode_R.type}' and scatter type '{self.scatter.type}'")

		self.D = self.scatter.hessian #* top.atom_weight(self.M_C) * (const.eV2hartree / const.ang2bohr ** 2)
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
					k_c = electrode_dict["k_c"],
					w_D = self.w_D
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
			
			case _:
				raise ValueError(f"Unsupported electrode type: {electrode_dict['type']}")

	def __initialize_scatter(self, scatter_dict):
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
					k_x = scatter_dict["k_x"],
					k_y = scatter_dict["k_y"],
					k_xy = scatter_dict["k_xy"],
					k_c = scatter_dict["k_c"],
					k_c_xy = scatter_dict["k_c_xy"],
				)
    
			case "Chain1D":
				return Chain1D(
					k_c = scatter_dict["k_c"],
					interact_potential = scatter_dict["interact_potential"],
					interaction_range = scatter_dict["interaction_range"],
					lattice_constant = scatter_dict["lattice_constant"],
					atom_type = scatter_dict["atom_type"],
					k_x = scatter_dict["k_x"],
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
				k_c = self.electrode_L.k_c * (1 / np.sqrt(top.atom_weight(self.M_C) * top.atom_weight(self.M_L)))

				g_R = self.electrode_R.g
				k_c = self.electrode_R.k_c * (1 / np.sqrt(top.atom_weight(self.M_C) * top.atom_weight(self.M_R)))

			case ("Chain1D", "Chain1D"):
				#1D Jan PhD Thesis p.21

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
					 2 * self.electrode_L.interaction_range * self.scatter.N_y), dtype=float)
				
				k_RC = np.zeros((2 * self.electrode_R.interaction_range * self.electrode_R.N_y, 
					 2 * self.electrode_R.interaction_range * self.scatter.N_y), dtype=float)
				
				N_y_L = self.electrode_L.N_y
				N_y_R = self.electrode_R.N_y
				N_y_scatter = self.scatter.N_y
				interaction_range_L = self.electrode_L.interaction_range
				interaction_range_R = self.electrode_R.interaction_range
				all_k_c_x_L = self.electrode_L.ranged_force_constant()[3]
				all_k_c_x_R = self.electrode_R.ranged_force_constant()[3]

				#set up LC interaction matrix
				scatter_temp = 0
				for i in range(interaction_range_L):
					for j in range(i * 2 * N_y_L, i * 2 * N_y_L + 2 * N_y_L, 2):

						if j % 2 == 0:
							atomnr_el = np.ceil(float(j + 1) / 2)
							
							for l in range(i * 2 * N_y_scatter, i * 2 * N_y_scatter + 2 * N_y_scatter, 2):
								atomnr_scatter = np.ceil(float(l + 1) / 2)
								

							if i * N_y_L + 1 + (N_y_L - N_y_scatter) / 2 <= atomnr_el <= i * N_y_L + N_y_L - (N_y_L - N_y_scatter) / 2 and atomnr_scatter > scatter_temp:
			
								k_LC[j, l] = -all_k_c_x_L[-1][1]
								scatter_temp = atomnr_scatter
								break

							if i == interaction_range_L - 1 and atomnr_el == i * N_y_L +  (N_y_L - N_y_scatter) / 2:
								atomnr_scatter = int(1 + (N_y_L - N_y_scatter) / 2)
								k_LC[j: j + direct_interaction_L.shape[0], 0: 0 + direct_interaction_L.shape[1]] = direct_interaction_L
				
				#set up CR interaction matrix
				scatter_temp = 0
				for i in range(interaction_range_R):
					for j in range(i * 2 * N_y_R, i * 2 * N_y_R + 2 * N_y_R, 2):

						if j % 2 == 0:
							atomnr_el = np.ceil(float(j + 1) / 2)
							
							for l in range(i * 2 * N_y_scatter, i * 2 * N_y_scatter + 2 * N_y_scatter, 2):
								atomnr_scatter = np.ceil(float(l + 1) / 2)
								

							if i * N_y_R + 1 + (N_y_R - N_y_scatter) / 2 <= atomnr_el <= i * N_y_R + N_y_R - (N_y_R - N_y_scatter) / 2 and atomnr_scatter > scatter_temp:
			
								k_RC[j, l] = -all_k_c_x_R[-1][1]
								scatter_temp = atomnr_scatter
								break

							if i == interaction_range_R - 1 and atomnr_el == i * N_y_R + (N_y_R - N_y_scatter) / 2:
								atomnr_scatter = int(1 + (N_y_R - N_y_scatter) / 2)
								k_RC[j: j + direct_interaction_R.shape[0], 0: 0 + direct_interaction_R.shape[1]] = direct_interaction_R

				#k_LC = self.electrode_L.k_lc_LL
				#k_RC = self.electrode_R.k_lc_LL
	
		# Initialize sigma array with the same shape as g
		sigma_L = np.zeros((self.N, self.D.shape[0], self.D.shape[1]), dtype=complex)
		sigma_R = np.zeros((self.N, self.D.shape[0], self.D.shape[1]), dtype=complex)

		# Build sigma matrix for each frequency depending on the (allowed) electrode model configuration

		# The 1D Chain transmission has an analytical expression an is covered directly there.
		
		# DebeyeModel (Markussen)
		if (g_L.shape, g_R.shape) == ((self.N,), (self.N,)) and self.electrode_L.interaction_range > 1 and self.electrode_R.interaction_range > 1: 
			sigma_L[0: k_LC.shape[0], 0: k_LC.shape[1]] = k_LC
			sigma_R[sigma_R.shape[0] - k_RC.shape[0]: sigma_R.shape[0], sigma_R.shape[1] - k_RC.shape[1]: sigma_R.shape[1]] = k_RC
   
  
		# 2D case (decimation technique)
		elif (g_L.shape, g_R.shape) == ((self.N, 2 * self.electrode_L.interaction_range * self.electrode_L.N_y, 2 * self.electrode_L.interaction_range * self.electrode_L.N_y), \
			(self.N, 2 * self.electrode_R.interaction_range * self.electrode_R.N_y, 2 * self.electrode_R.interaction_range * self.electrode_R.N_y)):

			#TODO: check if this is correct >> k_LC dimension
			#sigma_L_temp = np.array(list(map(lambda i: np.dot(np.dot(k_LC.T, g_L[i]), k_LC), self.i)))
			#sigma_R_temp = np.array(list(map(lambda i: np.dot(np.dot(k_RC.T, g_R[i]), k_RC), self.i)))
			
			sigma_L_temp = np.array(list(map(lambda i: np.dot(np.dot(self.electrode_L.k_lc_LL.T, g_L[i]), self.electrode_L.k_lc_LL), self.i)))
			sigma_R_temp = np.array(list(map(lambda i: np.dot(np.dot(self.electrode_R.k_lc_LL.T, g_R[i]), self.electrode_R.k_lc_LL), self.i)))

			for i in range(self.N):
				sigma_L[i, 0: sigma_L_temp.shape[1], 0: sigma_L_temp.shape[2]] = sigma_L_temp[i]
				sigma_R[i, sigma_R.shape[1] - sigma_R_temp.shape[1]: sigma_R.shape[1], sigma_R.shape[2] - sigma_R_temp.shape[2]: sigma_R.shape[2]] = sigma_R_temp[i]

		# 3D case (decimation technique)
		elif (g_L.shape, g_R.shape) == ((self.N, 3 * self.electrode_L.interaction_range * self.electrode_L.N_y, 3 * self.electrode_L.interaction_range * self.electrode_L.N_y), \
			(self.N, 3 * self.electrode_R.interaction_range * self.electrode_R.N_y, 3 * self.electrode_R.interaction_range * self.electrode_R.N_y)):
		
			sigma_L = np.array(list(map(lambda i: np.dot(np.dot(k_LC.T, g_L[i]), k_LC), self.i)))
			sigma_R = np.array(list(map(lambda i: np.dot(np.dot(k_RC.T, g_R[i]), k_RC), self.i)))

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

	def plot_eigenchannels(self):
		# top.write_plot_data(data_path + "/transmission_channels.dat", (T, T_val_tuple), "T (K), T_c")
		fig, ax = plt.subplots()
		for i in range(self.T_channel_vals.shape[1]):
			ax.plot(self.E, self.T_channel_vals[:, i], label=i + 1, ls="--")
		#ax.set_yscale('log')
		ax.set_xlabel('Phonon Energy ($\mathrm{meV}$)', fontsize=12)
		ax.set_ylabel(r'Transmission $\tau_{\mathrm{ph}}$', fontsize=12)
		ax.axvline(self.w_D * const.unit2SI * const.h_bar / (const.meV2J), ls="--", color="black")
		ax.axhline(1, ls="--", color="black")
		ax.set_ylim(1E-4, 2)
		plt.rc('xtick', labelsize=12)
		plt.rc('ytick', labelsize=12)
		plt.legend(fontsize=12)
		plt.savefig(self.data_path + "/transport_channels.pdf", bbox_inches='tight')

	def	plot_transport(self, write_data=True):
		"""Writes out the raw data and plots the transport properties of the system."""

		if write_data:
			top.write_plot_data(self.data_path + "/phonon_trans.dat", (self.w, self.T), "w (sqrt(har/(bohr**2*u))), T_vals")
			top.write_plot_data(self.data_path + "/kappa.dat", (self.temperature, self.kappa), "T (K), kappa (pW/K)")

		print(f'TauMax = {max(self.T)}, TauMin = {min(self.T)}, T_0 = {self.T[0]}')
		print(f'KappaMax = {max(self.kappa)}, KappaMin = {min(self.kappa)}')
		#print(max(self.E), min(self.E))
		fig, (ax1, ax2) = plt.subplots(2, 1)
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
		ax1.set_xlim(0, 0.5 * self.E_D)
		ax1.set_xticklabels(ax1.get_xticks(), fontproperties=prop)
		ax1.set_yticklabels(ax1.get_yticks(), fontproperties=prop)
		ax1.xaxis.set_major_formatter(FormatStrFormatter('%.1f'))
		ax1.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
		ax1.grid()
  
		ax2.plot(self.temperature, self.kappa)
		ax2.set_xlabel('Temperature ($K$)', fontsize=12, fontproperties=prop)
		ax2.set_ylabel(r'$\kappa_{\mathrm{ph}}\;(\mathrm{pw/K})$', fontsize=12, fontproperties=prop)
		ax2.grid()
		ax2.set_xticklabels(ax1.get_xticks(), fontproperties=prop)
		ax2.set_yticklabels(ax1.get_yticks(), fontproperties=prop)
		ax2.xaxis.set_major_formatter(FormatStrFormatter('%.1f'))
		ax2.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
     
		plt.rc('xtick', labelsize=12)
		plt.rc('ytick', labelsize=12)
		plt.xticks(fontproperties=prop)
		plt.yticks(fontproperties=prop)
		plt.savefig(self.data_path + "/transport.pdf", bbox_inches='tight')
		plt.clf()

	def tranport_calc(self):
     
		#self.calculate_G_cc()
		#self.calculate_T()
		#self.calc_kappa()
		#self.plot_transport()
  
		if self.eigenchannel == True:
      
			self.plot_eigenchannels()
			data = list([self.w])
   
			for j in range(0, self.channel_max):
				data.append(self.T_channel_vals[:,j])
			top.write_plot_data(self.data_path + "/phonon_trans_channel.dat", data, "T (K), kappa (pW/K)")

		top.write_plot_data(self.data_path + "/phonon_trans.dat", (self.w, self.T), "w (sqrt(har/(bohr**2*u))), T_vals")
		top.write_plot_data(self.data_path + "/kappa.dat", (self.temperature, self.kappa), "T (K), kappa (pW/K)")


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
        data_path=data_path,
        electrode_dict_L=electrode_dict_L,
		electrode_dict_R=electrode_dict_R,
        scatter_dict=scatter_dict,
        E_D=E_D,
        M_L=M_L,
		M_R=M_R,
        M_C=M_C,
        N=N,
        T_min=T_min,
        T_max=T_max,
        kappa_grid_points=kappa_grid_points
    )

    # Start calculation    
    print('debug')
    #PT.tranport_calc()
