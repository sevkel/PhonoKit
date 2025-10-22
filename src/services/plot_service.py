"""
Plot Service

Service module for plotting and data output operations.

Author: Severin Keller
Date: 2025
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import tmoutproc as top


class PlotService:
    """
    Service class for plotting and data output.
    
    Handles:
    - Transmission plots
    - DOS plots  
    - Data file generation
    """
    
    def __init__(self, data_path, sys_descr, electrode_dict_L, electrode_dict_R, 
                 scatter_type, electrode_L, E_D, prop):
        """
        Initialize the plot service.
        
        Args:
            data_path (str): Output directory path
            sys_descr (str): System description
            electrode_dict_L (dict): Left electrode configuration
            electrode_dict_R (dict): Right electrode configuration
            scatter_type (str): Scattering object type
            electrode_L: Left electrode object
            E_D (float): Debye energy
            prop: Font properties
        """
        self.data_path = data_path
        self.sys_descr = sys_descr
        self.electrode_dict_L = electrode_dict_L
        self.electrode_dict_R = electrode_dict_R
        self.scatter_type = scatter_type
        self.electrode_L = electrode_L
        self.E_D = E_D
        self.prop = prop
    
    def plot_transport(self, w, T, kappa, temperature, write_data=True, plot_data=False):
        """
        Plot and save transport data.
        
        Args:
            w (np.ndarray): Frequency array
            T (np.ndarray): Transmission array
            kappa (np.ndarray): Thermal conductance array
            temperature (np.ndarray): Temperature array
            write_data (bool): Whether to write data files
            plot_data (bool): Whether to generate plots
        """
        if write_data:
            self._write_transport_data(w, T, kappa, temperature)
        
        print(f'TauMax = {max(T)}, TauMin = {min(T)}, T_0 = {T[0]}')
        print(f'KappaMax = {max(kappa)}, KappaMin = {min(kappa)}')
        
        if plot_data:
            self._create_transport_plot(w, T)
    
    def plot_dos(self, w, electrode_L, electrode_R, write_data=True, plot_dos=False):
        """
        Plot and save density of states data.
        
        Args:
            w (np.ndarray): Frequency array
            electrode_L: Left electrode object
            electrode_R: Right electrode object
            write_data (bool): Whether to write data files
            plot_dos (bool): Whether to generate plots
        """
        dos_L = electrode_L.dos
        dos_real_L = electrode_L.dos_real
        dos_L_cpld = electrode_L.dos_cpld
        dos_real_L_cpld = electrode_L.dos_real_cpld
        
        dos_R = electrode_R.dos
        dos_real_R = electrode_R.dos_real
        dos_R_cpld = electrode_R.dos_cpld
        dos_real_R_cpld = electrode_R.dos_real_cpld
        
        print(f'DOS Left electrode max/min: {max(dos_L)}, {min(dos_L)}')
        print(f'DOS Right electrode max/min: {max(dos_R)}, {min(dos_R)}')
        
        if write_data:
            self._write_dos_data(w, dos_L, dos_real_L, dos_R, dos_real_R, 
                               dos_L_cpld, dos_real_L_cpld, dos_R_cpld, dos_real_R_cpld)
        
        if plot_dos:
            self._create_dos_plot(w, dos_real_L_cpld, dos_L_cpld, 
                                dos_real_R_cpld, dos_R_cpld)
    
    def _write_transport_data(self, w, T, kappa, temperature):
        """Write transport data to files."""
        path_trans = os.path.join(self.data_path, "trans")
        path_transdos = os.path.join(self.data_path, "trans+dos")
        path_kappa = os.path.join(self.data_path, "kappa")
        
        for path in [path_trans, path_transdos, path_kappa]:
            if not os.path.exists(path):
                os.makedirs(path)
        
        base_filename = self._generate_base_filename()
        
        top.write_plot_data(
            os.path.join(path_trans, f"{base_filename}.dat"),
            (w, T), "w (sqrt(har/(bohr**2*u))), T_vals")
        
        top.write_plot_data(
            os.path.join(path_transdos, f"{base_filename}.dat"),
            (w, T), "w (sqrt(har/(bohr**2*u))), T_vals")
        
        top.write_plot_data(
            os.path.join(path_kappa, f"{base_filename}_KAPPA.dat"),
            (temperature, kappa), "T (K), kappa (pW/K)")
    
    def _write_dos_data(self, w, dos_L, dos_real_L, dos_R, dos_real_R,
                       dos_L_cpld, dos_real_L_cpld, dos_R_cpld, dos_real_R_cpld):
        """Write DOS data to files."""
        path_dos = os.path.join(self.data_path, "dos")
        path_transdos = os.path.join(self.data_path, "trans+dos")
        
        for path in [path_dos, path_transdos]:
            if not os.path.exists(path):
                os.makedirs(path)
        
        base_filename = self._generate_base_filename()
        
        data_tuple = (w, dos_L, dos_real_L, dos_R, dos_real_R, 
                     dos_L_cpld, dos_real_L_cpld, dos_R_cpld, dos_real_R_cpld)
        header = ("w (sqrt(har/(bohr**2*u))), DOS_L (a.u.), DOS_L_real (a.u.), "
                 "DOS_R (a.u.), DOS_R_real (a.u.), DOS_L_cpld (a.u.), "
                 "DOS_L_real_cpld (a.u.), DOS_R_cpld (a.u.), DOS_R_real_cpld (a.u.)")
        
        top.write_plot_data(
            os.path.join(path_dos, f"{base_filename}_DOS.dat"),
            data_tuple, header)
        
        top.write_plot_data(
            os.path.join(path_transdos, f"{base_filename}_DOS.dat"),
            data_tuple, header)
    
    def _create_transport_plot(self, w, T):
        """Create and save transmission plot."""
        fig, ax1 = plt.subplots(1, 1)
        fig.tight_layout()
        
        ax1.plot(w, T)
        ax1.set_xlabel(r'Phonon Energy ($\mathrm{meV}$)', fontsize=12, fontproperties=self.prop)
        ax1.set_ylabel(r'$\tau_{\mathrm{ph}}$', fontsize=12, fontproperties=self.prop)
        ax1.set_xlim(0, 1 * self.E_D)
        ax1.set_xticklabels(ax1.get_xticks(), fontproperties=self.prop)
        ax1.set_yticklabels(ax1.get_yticks(), fontproperties=self.prop)
        ax1.grid()
        
        plt.rc('xtick', labelsize=12)
        plt.rc('ytick', labelsize=12)
        plt.xticks(fontproperties=self.prop)
        plt.yticks(fontproperties=self.prop)
        
        base_filename = self._generate_base_filename()
        plt.savefig(os.path.join(self.data_path, f"{base_filename}.pdf"), 
                   bbox_inches='tight')
        plt.clf()
    
    def _create_dos_plot(self, w, dos_real_L_cpld, dos_L_cpld, 
                        dos_real_R_cpld, dos_R_cpld):
        """Create and save DOS plot."""
        fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
        fig.tight_layout()
        
        ax1.set_title('DOS Left electrode', fontproperties=self.prop)
        ax1.plot(w, dos_real_L_cpld, label=r'$\Re(d)$', color='blue', linestyle='--')
        ax1.plot(w, dos_L_cpld, label=r'$\Im(d)$', color='red')
        ax1.set_ylabel('DOS', fontsize=12, fontproperties=self.prop)
        ax1.set_xticklabels(ax1.get_xticks(), fontproperties=self.prop)
        ax1.set_yticklabels(ax1.get_yticks(), fontproperties=self.prop)
        ax1.grid()
        ax1.legend(fontsize=12, prop=self.prop)
        
        ax2.set_title('DOS Right electrode', fontproperties=self.prop)
        ax2.plot(w, dos_real_R_cpld, label=r'$\Re(d)$', color='blue', linestyle='--')
        ax2.plot(w, dos_R_cpld, label=r'$\Im(d)$', color='red')
        ax2.set_ylabel('DOS', fontsize=12, fontproperties=self.prop)
        ax2.set_xlabel(r'Phonon Energy ($\mathrm{meV}$)', fontsize=12, fontproperties=self.prop)
        ax2.set_xticklabels(ax2.get_xticks(), fontproperties=self.prop)
        ax2.set_yticklabels(ax2.get_yticks(), fontproperties=self.prop)
        ax2.grid()
        ax2.legend(fontsize=12, prop=self.prop)
        
        base_filename = self._generate_base_filename()
        plt.savefig(os.path.join(self.data_path, f"{base_filename}_DOS.pdf"), 
                   bbox_inches='tight')
    
    def _generate_base_filename(self):
        """Generate base filename for saving."""
        base = (f"{self.sys_descr}___PT_elL={self.electrode_dict_L['type']}_"
               f"elR={self.electrode_dict_R['type']}_"
               f"CC={self.scatter_type}_"
               f"intrange={self.electrode_L.interaction_range}")
        
        try:
            base += f"_kcoupl_x={self.electrode_dict_L['k_coupl_x']}"
        except (KeyError, AttributeError):
            pass
        
        try:
            base += f"_kcoupl_xy={self.electrode_dict_L['k_coupl_xy']}"
        except (KeyError, AttributeError):
            pass
        
        return base
    
    def write_coupled_surface_greens_functions(self, w, electrode_L, electrode_R):
        """
        Write coupled surface Green's functions for left and right electrodes to npz files.
        
        Args:
            w (np.ndarray): Frequency array
            electrode_L: Left electrode object with g attribute
            electrode_R: Right electrode object with g attribute
        """
        # Create output directory
        cpld_sfg_path = os.path.join(self.data_path, "cpld_sfg")
        if not os.path.exists(cpld_sfg_path):
            os.makedirs(cpld_sfg_path)
        
        base_filename = self._generate_base_filename()
        
        # Save left electrode coupled surface Green's function
        npz_filename_L = os.path.join(cpld_sfg_path, f"{base_filename}_cpld_g_L.npz")
        np.savez(npz_filename_L, 
                w=w,
                g_cpld_L=electrode_L.g,
                electrode_type=self.electrode_dict_L['type'])
        
        # Save right electrode coupled surface Green's function  
        npz_filename_R = os.path.join(cpld_sfg_path, f"{base_filename}_cpld_g_R.npz")
        np.savez(npz_filename_R,
                w=w, 
                g_cpld_R=electrode_R.g,
                electrode_type=self.electrode_dict_R['type'])
        
        print(f"Coupled surface Green's functions saved to: {cpld_sfg_path}")
        print(f"  - Left electrode: {base_filename}_cpld_g_L.npz")
        print(f"  - Right electrode: {base_filename}_cpld_g_R.npz")
