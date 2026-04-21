import matplotlib
import matplotlib.pyplot as plt 
import matplotlib.font_manager as fm
from matplotlib.ticker import FormatStrFormatter
import numpy as np
import glob
import re
import scienceplots

# Font und Stil
matplotlib.rcParams['font.family'] = r'C://Users//sevke//Desktop//Dev//fonts//fira_sans//FiraSans-Regular.ttf'
prop = fm.FontProperties(fname=r'C://Users//sevke//Desktop//Dev//fonts//fira_sans//FiraSans-Regular.ttf', size=16)
plt.style.use(['science', 'notebook', 'no-latex'])

# Plot-Einstellungen
legend_fontsize_main = 20
legend_markerscale = 1.5
legend_handlelength = 2.5
legend_handletextpad = 0.8
legend_columnspacing = 2.0

# Ordnerpfad anpassen - HIER DEN PFAD ÄNDERN
dest_fld = r"C:\Users\sevke\Desktop\Dev\MA\phonokit\plot\new_paper_results\periodic\3-1-3_N=2000_0-25_periodic_Nq=350\band_structure"

# Plot-Einstellungen
legend_position = 'upper right'
legend_ncol = 1

# Brillouin-Zone Optionen
# 'full': Ganze BZ [-π, π]
# 'positive': Nur positive Seite [0, π]
# 'negative': Nur negative Seite [-π, 0]
brillouin_zone = 'positive'  # Optionen: 'full', 'positive', 'negative'

ylim_upper = None  # None = automatisch
ylim_lower = None  # None = automatisch
xlim_upper = 29#80*0.4   # meV
xlim_lower = 0

# Ordner nach .npz Dateien durchsuchen
ordner = dest_fld + "/*.npz"

# Plot vorbereiten
fig, ax = plt.subplots(1, 1, figsize=(12, 8))
fig.tight_layout()

# Dateien sammeln
dateien = glob.glob(ordner)

# Ergebnisse speichern für Sortierung
plots = []

print(f"Gefundene Dateien: {len(dateien)}")

for datei in dateien:
    # Lade npz Datei
    try:
        data = np.load(datei, allow_pickle=True)
        
        # Erkenne Elektrodentyp
        electrode_type = str(data['electrode_type']) if 'electrode_type' in data else 'Unknown'
        
        # Extrahiere Parameter aus Dateiname
        filename = datei.split('\\')[-1]
        
        # Extrahiere kcoupl_xy falls vorhanden
        match_xy = re.search(r'kcoupl_xy=(\d+(?:\.\d+)?)', filename)
        k_xy = float(match_xy.group(1)) if match_xy else None
        
        # Extrahiere kcoupl_x falls vorhanden
        match_x = re.search(r'kcoupl_x=(\d+(?:\.\d+)?)', filename)
        k_x = float(match_x.group(1)) if match_x else None
        
        # Extrahiere ob L oder R Elektrode
        is_left = '_bandstruct_L.npz' in filename
        is_right = '_bandstruct_R.npz' in filename
        electrode_side = 'L' if is_left else ('R' if is_right else 'Unknown')
        
        # Prüfe Format
        if 'k_x' in data.keys() and 'freqs' in data.keys():
            # Ribbon2D Format
            k_x_data = data['k_x']
            freqs = data['freqs']
            
            # Konvertiere Frequenzen zu Energie (meV)
            # Annahme: freqs sind in THz oder rad/s, conversion zu meV nötig
            # Für jetzt: direkt verwenden, ggf. anpassen
            energies = freqs  # Falls Konversion nötig: freqs * conversion_factor
            
            plots.append({
                'filename': filename,
                'electrode_type': electrode_type,
                'electrode_side': electrode_side,
                'k_xy': k_xy,
                'k_x_param': k_x,
                'k_x_data': k_x_data,
                'energies': energies,
                'format': 'Ribbon2D'
            })
            
            print(f"✓ Geladen: {filename} ({electrode_type}, {energies.shape[1]} Bänder)")
            
        elif 'q_y_values' in data.keys():
            # DecimationFourier Format
            q_y_values = data['q_y_values']
            k_x_arrays = data['k_x_arrays']
            freqs_arrays = data['freqs_arrays']
            
            # Nimm den mittleren q_y Wert
            q_index = len(q_y_values) // 2
            print(q_y_values)
            q_index = -1
            k_x_data = k_x_arrays[q_index]
            freqs = freqs_arrays[q_index]
            
            energies = freqs
            
            plots.append({
                'filename': filename,
                'electrode_type': electrode_type,
                'electrode_side': electrode_side,
                'k_xy': k_xy,
                'k_x_param': k_x,
                'k_x_data': k_x_data,
                'energies': energies,
                'format': 'DecimationFourier',
                'q_y': q_y_values[q_index]
            })
            
            print(f"✓ Geladen: {filename} ({electrode_type}, q_y={q_y_values[q_index]:.4f}, {energies.shape[1]} Bänder)")
        
        data.close()
        
    except Exception as e:
        print(f"✗ Fehler beim Laden von {datei}: {e}")
        continue

# Sortiere nach k_xy, dann k_x
plots.sort(key=lambda x: (x['k_xy'] if x['k_xy'] is not None else 999999, 
                          x['k_x_param'] if x['k_x_param'] is not None else 999999,
                          x['electrode_side']))

print(f"\nVerarbeite {len(plots)} Bandstrukturen...")

# Farbpalette
n_plots = min(len(plots), 15)
if n_plots <= 9:
    colors = plt.cm.Set1(np.linspace(0, 1, n_plots))
elif n_plots <= 15:
    colors1 = plt.cm.Set1(np.linspace(0, 1, 9))
    colors2 = plt.cm.Dark2(np.linspace(0, 1, n_plots - 9))
    colors = np.vstack([colors1, colors2])
else:
    colors = plt.cm.tab20(np.linspace(0, 1, min(20, n_plots)))
    if len(plots) > 20:
        print(f"Warnung: Mehr als 20 Plots ({len(plots)}). Nur die ersten 20 werden geplottet.")
        plots = plots[:20]

# Linienstil
line_style = '-'
line_width = 1.5

# Plotte alle Bandstrukturen
for i, plot_data in enumerate(plots):
    k_x_data = plot_data['k_x_data']
    energies = plot_data['energies']
    
    # Filtere k-Werte basierend auf Brillouin-Zone Einstellung
    if brillouin_zone == 'positive':
        # Nur positive k-Werte
        mask = k_x_data >= 0
    elif brillouin_zone == 'negative':
        # Nur negative k-Werte
        mask = k_x_data <= 0
    else:  # 'full'
        # Alle k-Werte
        mask = np.ones(len(k_x_data), dtype=bool)
    
    k_x_filtered = k_x_data[mask]
    energies_filtered = energies[mask, :]
    
    # Label erstellen
    label_parts = []
    if plot_data['k_xy'] is not None:
        label_parts.append(f"k_xy={int(plot_data['k_xy'])}")
    if plot_data['k_x_param'] is not None:
        label_parts.append(f"k_x={int(plot_data['k_x_param'])}")
    label_parts.append(f"({plot_data['electrode_side']})")
    
    if plot_data['format'] == 'DecimationFourier':
        label_parts.append(f"q_y={plot_data['q_y']:.3f}")
    
    label = " ".join(label_parts)
    
    color = colors[i % len(colors)]
    
    # Plotte alle Bänder dieser Bandstruktur
    n_bands = energies_filtered.shape[1]
    for band_idx in range(n_bands):
        # Nur erstes Band bekommt Label
        band_label = label if band_idx == 0 else None
        
        ax.plot(energies_filtered[:, band_idx], k_x_filtered, 
               label=band_label, 
               color=color,
               linestyle=line_style,
               linewidth=line_width,
               alpha=0.8,
               zorder=10)

# Achsenbeschriftungen
ax.set_xlabel(r'$E\,(\mathrm{meV})$', fontsize=28, fontproperties=prop)
ax.set_ylabel(r'$q_{\text{x}}\,(\mathrm{1/a})$', fontsize=28, fontproperties=prop)

# Achsenlimits
if xlim_lower is not None and xlim_upper is not None:
    ax.set_xlim(xlim_lower, xlim_upper)
if ylim_lower is not None and ylim_upper is not None:
    ax.set_ylim(ylim_lower, ylim_upper)

# Fonts auf Ticks
for label in ax.get_xticklabels():
    label.set_fontproperties(prop)
    label.set_fontsize(20)
for label in ax.get_yticklabels():
    label.set_fontproperties(prop)
    label.set_fontsize(20)

ax.xaxis.set_major_formatter(FormatStrFormatter('%.1f'))
ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))

ax.grid(zorder=1, alpha=0.3)

# k=0 Linie
ax.axhline(y=0, color='k', linestyle='--', linewidth=1.0, alpha=0.5, zorder=5)

# Legende
legend = ax.legend(prop=prop, 
                  loc=legend_position,
                  frameon=True, 
                  fancybox=True, 
                  shadow=True,
                  ncol=legend_ncol,
                  fontsize=legend_fontsize_main,
                  markerscale=legend_markerscale,
                  handlelength=legend_handlelength,
                  handletextpad=legend_handletextpad,
                  columnspacing=legend_columnspacing)
legend.set_zorder(15)
legend.get_frame().set_alpha(0.9)

# Speichern mit BZ-Info im Dateinamen
bz_suffix = f"_{brillouin_zone}" if brillouin_zone != 'full' else ""
#plt.savefig(dest_fld + f"/bandstructure_comb{bz_suffix}.pdf", bbox_inches='tight')
plt.savefig(dest_fld + f"/bandstructure_comb{bz_suffix}.svg", bbox_inches='tight')

print(f"\n✓ Plots gespeichert in: {dest_fld}")
print(f"  Brillouin-Zone: {brillouin_zone}")

plt.show()
plt.clf()
