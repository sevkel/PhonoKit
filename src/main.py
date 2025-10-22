"""
Phononic Transport Calculations

Main module for phonon transport calculations including data output and visualization.
Implements the theoretical framework from:

M. Bürkle, Thomas J. Hellmuth, F. Pauly, Y. Asai, First-principles calculation of the 
thermoelectric figure of merit for [2,2]paracyclophane-based single-molecule junctions, 
PHYSICAL REVIEW B 91, 165419 (2015)
DOI: 10.1103/PhysRevB.91.165419

Features:
- Comprehensive phonon transport calculations
- Multiple electrode configurations
- Thermal conductance calculations
- Data visualization and output
- Support for various scattering objects

Author: Severin Keller
Date: 2025
"""

# Standard library imports
import sys
import json
import os

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import scienceplots

# Local imports
from model_systems import * 
import electrode as el
import calculate_kappa as ck
from utils import constants as const

# Service imports
from services import (
    SigmaCalculator,
    GreensFunctionCalculator,
    TransmissionCalculator,
    PlotService
) 


# ============================================================================
# PLOTTING CONFIGURATION
# ============================================================================

# For plotting
matplotlib.rcParams['font.family'] = r'C://Users//sevke//Desktop//Dev//fonts//fira_sans//FiraSans-Regular.ttf'
prop = fm.FontProperties(fname=r'C://Users//sevke//Desktop//Dev//fonts//fira_sans//FiraSans-Regular.ttf')
plt.style.use(['science', 'notebook', 'no-latex'])


# ============================================================================
# MAIN TRANSPORT CLASS
# ============================================================================

class PhononTransport:
	"""
	Phonon Transport Calculation Engine
	
	Implements phonon transport calculations based on the theoretical framework from:
	M. Bürkle, Thomas J. Hellmuth, F. Pauly, Y. Asai, First-principles calculation of the 
	thermoelectric figure of merit for [2,2]paracyclophane-based single-molecule junctions, 
	PHYSICAL REVIEW B 91, 165419 (2015)
	DOI: 10.1103/PhysRevB.91.165419

	Supports multiple electrode configurations and scattering objects with:
	- 4D indexing for frequency-momentum space calculations
	- Parallel computation for performance optimization
	- Comprehensive thermal conductance calculations
	- Automatic data output and visualization

	Args:
		data_path (str): Output directory for calculated data
		sys_descr (str): System description identifier for data organization
		electrode_dict (dict): Electrode configuration parameters
		scatter_dict (dict): Scattering object configuration parameters
		E_D (float): Debye energy in meV
		M_E (str): Atom type in the reservoir electrodes
		M_C (str): Atom type in the central scattering region
		N (int): Number of frequency grid points
		T_min (float): Minimum temperature for thermal conductance calculation
		T_max (float): Maximum temperature for thermal conductance calculation
		kappa_grid_points (int): Number of temperature grid points for thermal conductance

	Attributes:
		data_path (str): Output directory path
		sys_descr (str): System description string
		electrode_dict (dict): Electrode configuration dictionary
		scatter_dict (dict): Scattering object configuration dictionary
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
			  E_D, M_E, M_C, N, T_min, T_max, kappa_grid_points):
		self.data_path = data_path
		self.sys_descr = sys_descr
		self.electrode_dict_L = electrode_dict_L
		self.electrode_dict_R = electrode_dict_R
		self.scatter_dict = scatter_dict
		self.M_E = M_E
		self.M_C = M_C
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

				if not (electrode_dict_l['type'] == 'DebyeModel' and electrode_dict_r['type'] == 'DebyeModel'):

					return Chain1D(
						k_c_x=scatter_dict["k_c_x"],
						k_coupl_x_l=electrode_dict_l["k_el_x"],
						k_coupl_x_r=electrode_dict_r["k_el_x"],
						interact_potential=scatter_dict["interact_potential"],
						interaction_range=scatter_dict["interaction_range"],
						lattice_constant=scatter_dict["lattice_constant"],
						atom_type=scatter_dict["atom_type"],
						N=scatter_dict["N"]
					)
				
				else:
					return Chain1D(
						k_c_x=scatter_dict["k_c_x"],
						k_coupl_x_l=electrode_dict_l["k_coupl_x"],
						k_coupl_x_r=electrode_dict_r["k_coupl_x"],
						interact_potential=scatter_dict["interact_potential"],
						interaction_range=scatter_dict["interaction_range"],
						lattice_constant=scatter_dict["lattice_constant"],
						atom_type=scatter_dict["atom_type"],
						N=scatter_dict["N"]
					)
			
			case _:
				raise ValueError(f"Unsupported scatter type: {scatter_dict['type']}")

	def calculate_sigma(self) -> tuple[np.ndarray, np.ndarray]:
		"""
		Calculate self-energies for the left (L) and right (R) electrodes using SigmaCalculator service.

		Returns:
			tuple: (sigma_L, sigma_R) - Self-energies for left and right electrodes
		"""
		sigma_calculator = SigmaCalculator(
			self.electrode_L,
			self.electrode_R,
			self.scatter,
			self.electrode_dict_L,
			self.electrode_dict_R,
			self.w,
			self.N,
			self.batch_size
		)
		
		return sigma_calculator.calculate()

	def calculate_G_cc(self) -> tuple[np.ndarray, np.ndarray]:
		"""
		Calculate retarded and advanced Green's functions using GreensFunctionCalculator service.

		Returns:
			tuple: (g_CC_ret, g_CC_adv) - Retarded and advanced Green's functions
		"""
		gf_calculator = GreensFunctionCalculator(
			self.w,
			self.D,
			self.sigma_L,
			self.sigma_R,
			self.electrode_dict_L,
			self.electrode_dict_R,
			self.electrode_L,
			self.batch_size
		)
		
		return gf_calculator.calculate()	

	def calculate_transmission(self) -> np.ndarray:
		"""
		Calculate phonon transmission using TransmissionCalculator service.

		Returns:
			np.ndarray: Phonon transmission values
		"""
		transmission_calculator = TransmissionCalculator(
			self.w,
			self.sigma_L,
			self.sigma_R,
			self.g_CC_ret,
			self.g_CC_adv,
			self.scatter,
			self.electrode_dict_L,
			self.electrode_dict_R,
			self.scatter_dict,
			self.electrode_L,
			self.sys_descr,
			self.data_path,
			self.batch_size
		)
		
		return transmission_calculator.calculate()
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
		Write and plot transport properties using PlotService.

		Args:
			write_data (bool): Flag if data write-out is wanted
			plot_data (bool): Flag if the data shall be plotted
		"""
		plot_service = PlotService(
			self.data_path,
			self.sys_descr,
			self.electrode_dict_L,
			self.electrode_dict_R,
			self.scatter_dict["type"],
			self.electrode_L,
			self.E_D,
			prop
		)
		
		plot_service.plot_transport(
			self.w,
			self.T,
			self.temperature,
			self.kappa,
			write_data=write_data,
			plot_data=plot_data
		)

	def plot_dos(self, write_data=True, plot_dos=False) -> None:
		"""
		Write and plot DOS using PlotService.

		Args:
			write_data (bool): Flag if data write-out is wanted
			plot_dos (bool): Flag if the DOS-data shall be plotted
		"""
		plot_service = PlotService(
			self.data_path,
			self.sys_descr,
			self.electrode_dict_L,
			self.electrode_dict_R,
			self.scatter_dict["type"],
			self.electrode_L,
			self.E_D,
			prop
		)
		
		plot_service.plot_dos(
			self.w,
			self.electrode_L,
			self.electrode_R,
			write_data=write_data,
			plot_dos=plot_dos
		)

	def write_coupled_surface_greens_functions(self):
		"""
		Write coupled surface Green's functions for both electrodes to npz files.
		Files will be saved in the 'cpld_sfg' subdirectory of the data path.
		"""
		plot_service = PlotService(
			self.data_path,
			self.sys_descr,
			self.electrode_dict_L,
			self.electrode_dict_R,
			self.scatter_dict["type"],
			self.electrode_L,
			self.E_D,
			prop
		)
		
		plot_service.write_coupled_surface_greens_functions(
			self.w,
			self.electrode_L,
			self.electrode_R
		)

	
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
        M_C = M_C,
        N = N,
        T_min = T_min,
        T_max = T_max,
        kappa_grid_points = kappa_grid_points
    )
    
    PT.plot_transport(plot_data=config["data_output"]["plot_transmission"])
    PT.plot_dos(plot_dos=config["data_output"]["plot_dos"])
        
    print("debug")