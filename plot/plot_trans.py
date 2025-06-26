import matplotlib
import matplotlib.pyplot as plt 
import matplotlib.font_manager as fm
from matplotlib.ticker import FormatStrFormatter
import numpy as np
import glob
import re
import scienceplots


matplotlib.rcParams['mathtext.fontset'] = 'custom'
matplotlib.rcParams['font.family'] = '/hpc/gpfs2/scratch/u/kellerse/Masterarbeit/fonts/fira_sans/FiraSans Regular.ttf'
matplotlib.rcParams['mathtext.rm'] = '/hpc/gpfs2/scratch/u/kellerse/Masterarbeit/fonts/fira_sansFiraSans Regular.ttf'
matplotlib.rcParams['mathtext.it'] = '/hpc/gpfs2/scratch/u/kellerse/Masterarbeit/fonts/fira_sansFiraSans Italic.ttf'
matplotlib.rcParams['mathtext.bf'] = '/hpc/gpfs2/scratch/u/kellerse/Masterarbeit/fonts/fira_sansFiraSan Bold.ttf'
prop = fm.FontProperties(fname='/hpc/gpfs2/scratch/u/kellerse/Masterarbeit/fonts/fira_sans/FiraSans Regular.ttf')

plt.style.use(['science', 'notebook', 'no-latex'])

# Ordnerpfad anpassen
dest_fld = "/hpc/gpfs2/scratch/u/kellerse/Masterarbeit/Dev/phonon_transport/object_oriented/src/plot/Scatter_CH1D_Nx=2"
ordner =  dest_fld + "/*.dat"


# Plot vorbereiten
fig, ax = plt.subplots(1, 1, figsize=(8, 6))
fig.tight_layout()

# Dateien sammeln
dateien = glob.glob(ordner)

# Ergebnisse speichern für Sortierung
plots = []

for datei in dateien:
    # kc-Wert extrahieren
    match = re.search(r'kc=(\d+(?:\.\d+)?).dat', datei)
    if match:
        kc_wert = float(match.group(1))
    else:
        kc_wert = None

    # Daten einlesen
    daten = np.loadtxt(datei, skiprows=1)
    frequenz = daten[:, 0]
    transmission = daten[:, 1]

    plots.append((kc_wert, frequenz, transmission))

# Nach kc sortieren, None-Werte zuletzt
plots.sort(key=lambda x: (x[0] is None, x[0]))

# Alles plotten
for kc_wert, frequenz, transmission in plots:
    label = r"$k_\mathrm{c(x,y)}$ = " + f"{int(kc_wert / 9)}" + r"$\,\frac{\text{meV}^2}{Å^2}$" if kc_wert is not None else "kc unbekannt"
    ax.plot(frequenz, transmission, label=label, linewidth=1.3)

# Achsen & Formatierungen
ax.set_xlabel(r'Phonon Energy ($\mathrm{meV}$)', fontsize=12, fontproperties=prop)
ax.set_ylabel(r'$\tau_{\mathrm{ph}}$', fontsize=12, fontproperties=prop)
ax.axhline(1, ls="--", color="black")

# Beispielwerte für Begrenzungen (falls du die aus Konstanten berechnest, hier einsetzen)
E_D = 60  # Beispiel für Debye-Energie in meV
ax.set_xlim(0, 0.5 * E_D)

# Fonts auf Ticks anwenden
for label in ax.get_xticklabels():
    label.set_fontproperties(prop)
for label in ax.get_yticklabels():
    label.set_fontproperties(prop)

ax.xaxis.set_major_formatter(FormatStrFormatter('%.1f'))
ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
ax.grid()

# Legende
ax.legend(prop=prop)
plt.savefig(dest_fld + "/transmission_comb.pdf", bbox_inches='tight')
