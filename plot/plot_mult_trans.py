import matplotlib
import matplotlib.pyplot as plt 
import matplotlib.font_manager as fm
from matplotlib.ticker import FormatStrFormatter, FuncFormatter
import numpy as np
import glob
import re
import scienceplots

# Filter-Einstellungen
filter = False # True = Filtere Transmissionswerte außerhalb der Schwellenwerte aus
threshold_upper = 10.0  # Obere Grenze für Filterung (Werte > threshold_upper werden entfernt)
threshold_lower = -0.001  # Untere Grenze für Filterung (Werte < threshold_lower werden entfernt)

# Font und Stil
matplotlib.rcParams['font.family'] = r'C://Users//sevke//Desktop//Dev//fonts//fira_sans//FiraSans-Regular.ttf'
prop = fm.FontProperties(fname=r'C://Users//sevke//Desktop//Dev//fonts//fira_sans//FiraSans-Regular.ttf', size=16)
plt.style.use(['science', 'notebook', 'no-latex'])

# Legenden-Einstellungen
legend_fontsize_main = 20  # Schriftgröße für Hauptlegende
legend_fontsize_const = 14  # Schriftgröße für Konstanten-Legende
legend_markerscale = 1.5  # Größe der Legenden-Marker (1.0 = normal, 2.0 = doppelt so groß)
legend_handlelength = 2.5  # Länge der Linien in der Legende (Standard: 2.0)
legend_handletextpad = 0.8  # Abstand zwischen Symbol und Text (Standard: 0.8)
legend_columnspacing = 2.0  # Abstand zwischen Spalten bei ncol > 1 (Standard: 2.0)

# Ordnerpfad anpassen
dest_fld = r"C:\Users\sevke\Desktop\Dev\MA\phonokit\plot\DecimationFourier\100y_1y2x_100y\trans"
dim1 = False
dim2 = True
logscale = False
legend_below = False  # True = Legende transparent (Graphen sichtbar), False = Legende über Graphen (verdeckt sie)
sep_label_pos = 'upper right'
ylim_upper = 2.5
ylim_lower = 0.0
ylim_lower_log = 1e-12  # Für logscale: untere Grenze für y-Achse
E_D = 80  
xlim_upper = E_D * 0.5

# Legendenposition im Plot
# Verfügbare Optionen:
# 'upper right', 'upper left', 'lower left', 'lower right', 'right', 'center left', 
# 'center right', 'lower center', 'upper center', 'center', 'best'
legend_position = 'lower right'
legend_ncol = 1  # Anzahl Spalten für die Hauptlegende


ordner =  dest_fld + "/*.dat"


# Plot vorbereiten
fig, ax = plt.subplots(1, 1, figsize=(12, 8))  # Größerer Plot für bessere Legende
fig.tight_layout()

# Dateien sammeln
dateien = glob.glob(ordner)

# Ergebnisse speichern für Sortierung
plots = []

for datei in dateien:
    # kc-Wert extrahieren
    if dim2 == True:
        match = re.search(r'kc_xy=(\d+(?:\.\d+)?)', datei) #xy
    elif dim1 == True:
        match = re.search(r'kc=(\d+(?:\.\d+)?)', datei)  # x
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

# Info über verarbeitete Dateien
print(f"Info: {len(plots)} Dateien verarbeitet")

# Bessere Farbpalette für klar unterscheidbare Farben (max 15)
import matplotlib.cm as cm
from matplotlib.colors import ListedColormap

# Klar unterscheidbare Farbpalette mit max 15 Farben
n_plots = min(len(plots), 15)  # Maximal 15 Plots
if n_plots <= 9:
    colors = plt.cm.Set1(np.linspace(0, 1, n_plots))  # Sehr kräftige, unterscheidbare Farben
elif n_plots <= 15:
    # Kombiniere Set1 (9 Farben) + ausgewählte Dark2 Farben (6 weitere)
    colors1 = plt.cm.Set1(np.linspace(0, 1, 9))
    colors2 = plt.cm.Dark2(np.linspace(0, 1, n_plots - 9))
    colors = np.vstack([colors1, colors2])
else:
    # Falls mehr als 15 Plots: Verwende nur die ersten 15
    colors = plt.cm.Set1(np.linspace(0, 1, 9))
    colors2 = plt.cm.Dark2(np.linspace(0, 1, 6))
    colors = np.vstack([colors, colors2])
    print(f"Warnung: Mehr als 15 Plots ({len(plots)}). Nur die ersten 15 werden geplottet.")
    plots = plots[:15]

# Verschiedene Linienstile für noch bessere Unterscheidung
# line_styles = ['-', '--', '-.', ':', '-', '--', '-.', ':']
# line_widths = [1.3, 1.5, 1.7, 1.3, 1.5, 1.7, 1.3, 1.5]

# Alle Linien gleiche Form
line_style = '-'
line_width = 1.3

# Alles plotten
# Konstante Werte als separate Legende (rechts oben) mit Text-only Handles
from matplotlib.patches import Rectangle

const_handles = []
const_labels = []
if dim2 == True:
    const_handles.append(Rectangle((0,0), 2, 2, fc="w", fill=False, edgecolor='none', linewidth=0))
    const_labels.append(r'$k_{\text{L,C,R}}^{\text{x}} = k_{\text{L,C,R}}^{\text{y}} = 100\,\frac{\text{meV}^2}{\text{Å}^2}$')
    '''const_handles.append(Rectangle((0,0), 2, 2, fc="w", fill=False, edgecolor='none', linewidth=0))
    const_labels.append(r'$k_{\text{R}}^{\text{x}} = k_{\text{R}}^{\text{y}} = 100\,\frac{\text{meV}^2}{\text{Å}^2}$')
    const_handles.append(Rectangle((0,0), 2, 2, fc="w", fill=False, edgecolor='none', linewidth=0))
    const_labels.append(r'$k_{\text{C}}^{\text{x}} = k_{\text{C}}^{\text{y}} = 100\,\frac{\text{meV}^2}{\text{Å}^2}$')'''

if dim1 == True:
    const_handles.append(Rectangle((0,0), 2, 2, fc="w", fill=False, edgecolor='none', linewidth=0))
    const_labels.append(r'$k_{\text{L}}^{\text{x}} = k_{\text{R}}^{\text{x}} = 100\,\frac{\text{meV}^2}{\text{Å}^2}$')

# Konstanten-Legende erstellen
const_legend = ax.legend(const_handles, const_labels,
                        prop=prop,
                        loc=sep_label_pos,
                        frameon=True,
                        fancybox=True,
                        shadow=True,
                        fontsize=legend_fontsize_const,
                        markerscale=legend_markerscale,
                        handlelength=0,  # Keine Liniensymbole für Konstanten
                        handletextpad=0)  # Kein Abstand zwischen Symbol und Text für Konstanten
const_legend.set_zorder(5)  # Zwischen Grid (1) und Graphen (10)

for i, (kc_wert, frequenz, transmission) in enumerate(plots):
    #label = r"$k_\mathrm{C}^{\text{xy}}$ = " + f"{int(kc_wert / 9)}" + r"$\,\frac{\text{meV}^2}{Å^2}$" if kc_wert is not None else "kc unbekannt"

    if dim2 == True:
        label = r"$k_\mathrm{L,C,R}^{\text{xy}}$ = " + f"{int(kc_wert / 9)}" + r"$\,\frac{\text{meV}^2}{Å^2}$" if kc_wert is not None else "kc unbekannt"
    elif dim1 == True:
        #label = r"$k_\mathrm{C}$ = " + f"{int(kc_wert / 9)}" + r"$\,\frac{\text{meV}^2}{Å^2}$" if kc_wert is not None else "kc unbekannt"
        if i == 1:
            label = r"$k_\mathrm{C}^{\text{x}}$ = 100.5" + r"$\,\frac{\text{meV}^2}{Å^2}$" if kc_wert is not None else "kc unbekannt" # mit ^x
        else:
            label = r"$k_\mathrm{C}^{\text{x}}$ = 100" + r"$\,\frac{\text{meV}^2}{Å^2}$" if kc_wert is not None else "kc unbekannt" # mit ^x
    # Farbe, Linienstil und -breite für bessere Unterscheidung
    color = colors[i % len(colors)]

    # Filter anwenden falls aktiviert
    if filter:
        # Nur Datenpunkte behalten, wo threshold_lower <= transmission <= threshold_upper
        mask = (transmission >= threshold_lower) & (transmission <= threshold_upper)
        frequenz_filtered = frequenz[mask]
        transmission_filtered = transmission[mask]
        
        # Anzahl der gefilterten Werte berechnen und ausgeben
        total_points = len(transmission)
        filtered_points = len(transmission_filtered)
        removed_points = total_points - filtered_points
        
        if removed_points > 0:
            print(f"Datei {kc_wert}: {removed_points} von {total_points} Werten außerhalb der Schwellenwerte gefiltert ({removed_points/total_points*100:.1f}%)")
    else:
        frequenz_filtered = frequenz
        transmission_filtered = transmission
    
    ax.plot(frequenz_filtered, transmission_filtered, 
           label=label, 
           color=color,
           linestyle=line_style,
           linewidth=line_width,
           alpha=1.0,  # Volle Deckkraft für kräftigere Farben
           zorder=10)  # Höhere zorder = vorne

# Achsen & Formatierungen
#ax.set_xlabel(r'Phonon Energy ($\mathrm{meV}$)', fontsize=12, fontproperties=prop)
#ax.set_ylabel(r'$\tau_{\mathrm{ph}}$', fontsize=12, fontproperties=prop)

ax.set_xlabel(r'$E_{\mathrm{ph}}\,(\mathrm{meV}$)', fontsize=20, fontproperties=prop)
if logscale:
    ax.set_ylabel(r'$\text{log}\left(\tau_{\mathrm{ph}}\right)$', fontsize=20, fontproperties=prop)
    ax.set_yscale('log')  # Logarithmische Y-Achse für bessere Sichtbarkeit
    #ax.set_xscale('log')
else:
    ax.set_ylabel(r'$\tau_{\mathrm{ph}}$', fontsize=20, fontproperties=prop)

#ax.axhline(1, ls="--", color="black")

# Beispielwerte für Begrenzungen (falls du die aus Konstanten berechnest, hier einsetzen)
ax.set_xlim(0.001, xlim_upper)  # X-Limit basierend auf Debye-Energie

if logscale:
    # Bei log scale: sehr niedrige untere Grenze um interessante Region bei 10^-15 zu zeigen
    ax.set_ylim(ylim_lower_log, ylim_upper)  # Von 1e-16 bis ylim_upper für log scale
else:
    ax.set_ylim(ylim_lower, ylim_upper)  # Y-Limit für linear scale

# Fonts auf Ticks anwenden
for label in ax.get_xticklabels():
    label.set_fontproperties(prop)
    label.set_fontsize(13)
for label in ax.get_yticklabels():
    label.set_fontproperties(prop)
    label.set_fontsize(13)

ax.xaxis.set_major_formatter(FormatStrFormatter('%.1f'))

ax.grid(zorder=1)  # Grid ganz hinten

# Hauptlegende für variable Parameter (ohne konstante Werte)
if legend_below:
    # Legende transparent - Graphen bleiben sichtbar
    legend = ax.legend(prop=prop, 
                      loc=legend_position,  # Variable Position
                      frameon=True, 
                      fancybox=True, 
                      shadow=True,
                      ncol=legend_ncol,  # Variable Spaltenanzahl
                      fontsize=legend_fontsize_main,
                      markerscale=legend_markerscale,
                      handlelength=legend_handlelength,
                      handletextpad=legend_handletextpad,
                      columnspacing=legend_columnspacing)
    legend.set_zorder(1)  # Hinter den Graphen (zorder=10)
    legend.get_frame().set_alpha(0.3)  # Sehr transparent
else:
    # Legende über den Graphen - verdeckt sie
    legend = ax.legend(prop=prop, 
                      loc=legend_position,  # Variable Position
                      frameon=True, 
                      fancybox=True, 
                      shadow=True,
                      ncol=legend_ncol,  # Variable Spaltenanzahl
                      fontsize=legend_fontsize_main,
                      markerscale=legend_markerscale,
                      handlelength=legend_handlelength,
                      handletextpad=legend_handletextpad,
                      columnspacing=legend_columnspacing)
    legend.set_zorder(15)  # Über den Graphen (zorder=10)
    legend.get_frame().set_alpha(0.9)  # Fast undurchsichtig

# Legende zwischen Grid und Graphen positionieren (wird oben überschrieben)
# legend.set_zorder(5)  # Wird durch legend_below Logic überschrieben
# legend.get_frame().set_alpha(0.9)  # Wird durch legend_below Logic überschrieben

# Konstanten-Legende wieder hinzufügen (da ax.legend() die vorherige überschreibt)
ax.add_artist(const_legend)
const_legend.set_zorder(5)
const_legend.get_frame().set_alpha(0.9)

# Kein zusätzlicher Platz mehr nötig, da Legende immer im Plot ist

if logscale:
        plt.savefig(dest_fld + "/transmission_comb(logscale).pdf", bbox_inches='tight')     
elif filter:
    plt.savefig(dest_fld + "/transmission_comb(filtered).pdf", bbox_inches='tight')
else:
    plt.savefig(dest_fld + "/transmission_comb.pdf", bbox_inches='tight')

plt.show()
plt.clf()