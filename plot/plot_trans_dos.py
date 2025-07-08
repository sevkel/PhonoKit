import matplotlib
import matplotlib.pyplot as plt 
import matplotlib.font_manager as fm
from matplotlib.ticker import FormatStrFormatter
import numpy as np
import glob
import re
import scienceplots

# does'nt work that well yet
matplotlib.rcParams['font.family'] = r'C://Users//sevke//Desktop//Dev//fonts//fira_sans//FiraSans-Regular.ttf'
prop = fm.FontProperties(fname=r'C://Users//sevke//Desktop//Dev//fonts//fira_sans//FiraSans-Regular.ttf')

plt.style.use(['science', 'notebook', 'no-latex'])

# Ordnerpfad anpassen
dest_fld = r"C:\Users\sevke\Desktop\Dev\MA\phonokit\src\plot\elL=3y_Sc=2x1y_elR=3y\trans+dos"
ordner =  dest_fld + "/*.dat"

# Dateien sammeln
dateien = glob.glob(ordner)

# Initialisierung für die drei Kurven
transmission_freq = None
transmission_val = None
dosL_freq = None
dosL_val = None
dosR_freq = None
dosR_val = None
kc_xy_info = ""

# kc_xy aus Dateiname extrahieren (für Kraftkonstanten-Info)
kc_pattern = re.compile(r'kc_xy=(\d+(?:\.\d+)?)')

for datei in dateien:
    # Transmission: Datei, die NICHT auf _DOS.dat endet
    if not datei.endswith("_DOS.dat"):
        daten = np.loadtxt(datei, skiprows=1)
        transmission_freq = daten[:, 0]
        transmission_val = daten[:, 1]
        match = kc_pattern.search(datei)
        if match:
            kc_xy_info = f"$k_{{c}}^{{xy}}$ = {match.group(1)} meV$^2$/Å$^2$"
    # DOS: Datei, die auf _DOS.dat endet
    elif datei.endswith("_DOS.dat"):
        daten = np.loadtxt(datei, skiprows=1)
        dosL_freq = daten[:, 0]
        dosL_val = daten[:, 1]
        dosR_freq = daten[:, 0]
        dosR_val = daten[:, 3]

# Plot vorbereiten
fig, ax1 = plt.subplots(1, 1, figsize=(8, 6))
fig.tight_layout()

# Transmission plotten (linke y-Achse)
if transmission_freq is not None and transmission_val is not None:
    ax1.plot(transmission_freq, transmission_val, label=r"$\tau_{\text{ph}}$", color='tab:blue', linewidth=1.3)

# Zweite y-Achse für DOS
ax2 = ax1.twinx()

# DOS Left plotten (rechte y-Achse)
if dosL_freq is not None and dosL_val is not None:
    ax2.plot(dosL_freq, dosL_val, color='tab:green', label=r"$\rho_{\text{ph,L}}$", linewidth=1.3)
# DOS Right plotten (rechte y-Achse)
if dosR_freq is not None and dosR_val is not None:
    ax2.plot(dosR_freq, dosR_val, '--', color='tab:orange', label=r"$\rho_{\text{ph,R}}$", linewidth=1.3)

if kc_xy_info:
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

# Fonts auf Ticks anwenden
for label in ax1.get_xticklabels():
    label.set_fontproperties(prop)
for label in ax1.get_yticklabels():
    label.set_fontproperties(prop)
for label in ax2.get_yticklabels():
    label.set_fontproperties(prop)

ax1.xaxis.set_major_formatter(FormatStrFormatter('%.1f'))
ax1.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
ax1.grid()

# Zweite y-Achse für DOS
ax2.set_ylabel(r"$\rho_{\text{ph}}\,\text{(a.u.)}$", fontsize=12, fontproperties=prop)
ax2.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))

# Legenden kombinieren
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, prop=prop, loc='center right')

plt.savefig(dest_fld + "/transmission_dos_combined.pdf", bbox_inches='tight')
plt.close(fig)