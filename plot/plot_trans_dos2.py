import matplotlib
import matplotlib.pyplot as plt 
import matplotlib.font_manager as fm
from matplotlib.ticker import FormatStrFormatter
import numpy as np
import glob
import re
import os
import scienceplots

# Font und Stil
matplotlib.rcParams['font.family'] = r'C://Users//sevke//Desktop//Dev//fonts//fira_sans//FiraSans-Regular.ttf'
prop = fm.FontProperties(fname=r'C://Users//sevke//Desktop//Dev//fonts//fira_sans//FiraSans-Regular.ttf')
plt.style.use(['science', 'notebook', 'no-latex'])

# Ordnerpfad anpassen
dest_fld = r"C:\Users\sevke\Desktop\Dev\MA\phonokit\src\plot\Chain2Layer_eL=2y_Sc=2x2y_elR=2y\trans+dos"
ordner =  dest_fld + "/*.dat"

dateien = glob.glob(ordner)

# Gruppiere Dateien nach kc_xy
kc_pattern = re.compile(r'kc_xy=(\d+(?:\.\d+)?)')
gruppen = {}

for datei in dateien:
    match = kc_pattern.search(datei)
    if match:
        kc_val = match.group(1)
        if kc_val not in gruppen:
            gruppen[kc_val] = {'trans': None, 'dos': None}
        if datei.endswith("_DOS.dat"):
            gruppen[kc_val]['dos'] = datei
        else:
            gruppen[kc_val]['trans'] = datei

# Für jede Gruppe einen Plot erzeugen
for kc_val, files in gruppen.items():
    transmission_freq = transmission_val = dosL_freq = dosL_val = dosR_freq = dosR_val = None

    # Transmission laden
    if files['trans'] is not None:
        daten = np.loadtxt(files['trans'], skiprows=1)
        transmission_freq = daten[:, 0]
        transmission_val = daten[:, 1]

    # DOS laden
    if files['dos'] is not None:
        daten = np.loadtxt(files['dos'], skiprows=1)
        dosL_freq = daten[:, 0]
        dosL_val = daten[:, 1]
        dosR_freq = daten[:, 0]
        dosR_val = daten[:, 3]

    # Plot nur wenn Transmission und DOS vorhanden
    if transmission_freq is not None and dosL_freq is not None:
        fig, ax1 = plt.subplots(1, 1, figsize=(8, 6))
        fig.tight_layout()

        # Transmission plotten
        ax1.plot(transmission_freq, transmission_val, label=r"$\tau_{\text{ph}}$", color='tab:blue', linewidth=1.3)

        # Zweite y-Achse für DOS
        ax2 = ax1.twinx()
        ax2.plot(dosL_freq, dosL_val, color='tab:green', label=r"$\rho_{\text{ph,L}}$", linewidth=1.3)
        ax2.plot(dosR_freq, dosR_val, '--', color='tab:orange', label=r"$\rho_{\text{ph,R}}$", linewidth=1.3)

        # Kraftkonstanten-Info in Plot einfügen (oben rechts)
        kc_xy_info = f"$k_{{c}}^{{xy}}$ = {kc_val} meV$^2$/Å$^2$"
        ax1.text(
            0.98, 0.9, kc_xy_info,
            transform=ax1.transAxes,
            fontsize=11,
            fontproperties=prop,
            ha='right', va='top',
            bbox=dict(facecolor='white', alpha=0.7, edgecolor='none')
        )

        # Achsen & Formatierungen
        ax1.set_xlabel(r'Phonon Energy ($\mathrm{meV}$)', fontsize=12, fontproperties=prop)
        ax1.set_ylabel(r'$\tau_{\mathrm{ph}}$', fontsize=12, fontproperties=prop)
        ax1.axhline(1, ls="--", color="black")
        E_D = 80  # Beispiel für Debye-Energie in meV
        ax1.set_xlim(0, E_D)

        for label in ax1.get_xticklabels():
            label.set_fontproperties(prop)
        for label in ax1.get_yticklabels():
            label.set_fontproperties(prop)
        for label in ax2.get_yticklabels():
            label.set_fontproperties(prop)

        ax1.xaxis.set_major_formatter(FormatStrFormatter('%.1f'))
        ax1.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
        ax1.grid()

        ax2.set_ylabel(r"$\rho_{\text{ph}}\,\text{(a.u.)}$", fontsize=12, fontproperties=prop)
        ax2.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))

        # Legenden kombinieren
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, prop=prop, loc='center right')

        # PDF-Dateiname
        pdf_name = f"transmission_dos_kcxy_{kc_val}.pdf"
        plt.savefig(os.path.join(dest_fld, pdf_name), bbox_inches='tight')
        plt.close(fig)