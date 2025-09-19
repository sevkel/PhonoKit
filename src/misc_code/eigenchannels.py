import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import scienceplots

#matplotlib.rcParams['font.family'] = r'C://Users//sevke//Desktop//Dev//fonts//fira_sans//FiraSans-Regular.ttf'
prop = fm.FontProperties(fname=r'C://Users//sevke//Desktop//Dev//fonts//fira_sans//FiraSans-Regular.ttf', size=14)
plt.style.use(['science', 'notebook', 'no-latex'])

#datapath = r"C:\Users\sevke\Desktop\Dev\MA\phonokit\plot\2y_2y2x_2y\trans_prob_matrices\2y_2y2x_2y___PT_elL=Ribbon2D_elR=Ribbon2D_CC=FiniteLattice2D_intrange=1_kc=900_kc_xy=0_trans_prob_matrix.npz"
#datapath = r"C:\Users\sevke\Desktop\Dev\MA\phonokit\plot\2y_2y2x_2y\trans_prob_matrices\2y_2y2x_2y___PT_elL=Ribbon2D_elR=Ribbon2D_CC=FiniteLattice2D_intrange=1_kc=900_kc_xy=1350_trans_prob_matrix.npz"
#datapath = r'C:\Users\sevke\Desktop\Dev\MA\phonokit\plot\3y_3y2x_3y\trans_prob_matrices\3_3y2x_3___PT_elL=Ribbon2D_elR=Ribbon2D_CC=FiniteLattice2D_intrange=1_kc=900_kc_xy=1890_trans_prob_matrix.npz'
#datapath = r'C:\Users\sevke\Desktop\Dev\MA\phonokit\plot\3y_3y2x_3y\trans_prob_matrices\3_3y2x_3___PT_elL=Ribbon2D_elR=Ribbon2D_CC=FiniteLattice2D_intrange=1_kc=900_kc_xy=1350_trans_prob_matrix.npz'
#datapath = r'C:\Users\sevke\Desktop\Dev\MA\phonokit\plot\3y_3y2x_3y\trans_prob_matrices\3_3y2x_3___PT_elL=Ribbon2D_elR=Ribbon2D_CC=FiniteLattice2D_intrange=1_kc=900_kc_xy=0_trans_prob_matrix.npz'
#datapath = r'C:\Users\sevke\Desktop\Dev\MA\phonokit\plot\tests\trans_prob_matrices\MEETING_TEST___PT_elL=Ribbon2D_elR=Ribbon2D_CC=FiniteLattice2D_intrange=1_kc=900_kc_xy=0_trans_prob_matrix.npz'
#datapath = r'C:\Users\sevke\Desktop\Dev\MA\phonokit\plot\tests\trans_prob_matrices\MEETING_TEST_xy___PT_elL=Ribbon2D_elR=Ribbon2D_CC=FiniteLattice2D_intrange=1_kc=900_kc_xy=450_trans_prob_matrix.npz'
#datapath = r'C:\Users\sevke\Desktop\Dev\MA\phonokit\plot\6y_2y2x_2y\trans_prob_matrices\6y_2y2x_2y___PT_elL=Ribbon2D_elR=Ribbon2D_CC=FiniteLattice2D_intrange=1_kc=900_kc_xy=450_trans_prob_matrix.npz'
#datapath = r'C:\Users\sevke\Desktop\Dev\MA\phonokit\plot\3y_1y2x_3y\trans_prob_matrices\3y_1y2x_3y___PT_elL=Ribbon2D_elR=Ribbon2D_CC=FiniteLattice2D_intrange=1_kc=900_kc_xy=0_trans_prob_matrix.npz'
#datapath = r'C:\Users\sevke\Desktop\Dev\MA\phonokit\plot\3y_1y2x_1y\trans_prob_matrices\3y_1y2x_1y___PT_elL=Ribbon2D_elR=Ribbon2D_CC=FiniteLattice2D_intrange=1_kc=900_kc_xy=90_trans_prob_matrix.npz'
datapath = r'C:\Users\sevke\Desktop\Dev\MA\phonokit\plot\4y_2y2x_4y\trans_prob_matrices\4y_2y2x_4y___PT_elL=Ribbon2D_elR=Ribbon2D_CC=FiniteLattice2D_intrange=1_kc=900_kc_xy=0_trans_prob_matrix.npz'
loaded_data = np.load(datapath)

w = loaded_data['w']
trans_prob = loaded_data['trans_prob_matrix']
trans_prob.shape



#----------------------------------------------------------------------------------------------------------------------------------



# Extract eigenvalues and eigenvectors for all 10000 matrices
print(f"Shape of trans_prob matrix: {trans_prob.shape}")
print(f"Number of frequencies: {len(w)}")

# Arrays for all eigenvalues and eigenvectors
all_eigenvals = np.zeros((trans_prob.shape[0], trans_prob.shape[1]), dtype=complex)
all_eigenvecs = np.zeros((trans_prob.shape[0], trans_prob.shape[1], trans_prob.shape[2]), dtype=complex)

# Calculate eigenvalues and eigenvectors for each matrix
print("Calculating eigenvalues and eigenvectors...")
for i in range(trans_prob.shape[0]):
    '''if i % 1000 == 0:
        print(f"Progress: {i}/{trans_prob.shape[0]} ({100*i/trans_prob.shape[0]:.1f}%)")
    '''
    # Calculate eigenvalues and eigenvectors
    eigvals, eigvecs = np.linalg.eig(trans_prob[i])
    
    # Sort in descending order by magnitude of eigenvalues
    sort_indices = np.argsort(np.abs(eigvals))[::-1]
    
    # Store sorted values
    all_eigenvals[i] = eigvals[sort_indices]
    all_eigenvecs[i] = eigvecs[:, sort_indices]

print("Finished!")




#----------------------------------------------------------------------------------------------------------------------------------




# Plot of the first few eigenvalues over all frequencies
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

# Check if eigenvalues are real (they should be for transmission matrices)
max_imag_part = np.max(np.abs(np.imag(all_eigenvals)))
print(f"Maximum imaginary part of eigenvalues: {max_imag_part:.2e}")

if max_imag_part < 1e-10:
    print("Eigenvalues are practically real - converting to real")
    all_eigenvals_real = np.real(all_eigenvals)
else:
    print("Eigenvalues have significant imaginary parts")
    all_eigenvals_real = all_eigenvals

# Plot 1: The first 5 eigenvalues over all frequencies
for i in range(min(5, all_eigenvals_real.shape[1])):
    ax1.plot(w, all_eigenvals_real[:, i], label=rf'$\tau^{{{i+1}}}_{{\mathrm{{ph}}}}$', alpha=0.8)

ax1.set_xlabel(r'$E_{\text{ph}}\,\text{(meV)}$', fontsize=16, fontproperties=prop)
ax1.set_ylabel(r'$\tau_{\text{ph}}^i$', fontsize=16, fontproperties=prop)
#ax1.set_title(r'Largest eigenvalues over all frequencies', fontsize=16)
ax1.legend(prop=prop)
ax1.grid(True, alpha=0.3)


for label in ax1.get_xticklabels():
    label.set_fontproperties(prop)
    label.set_fontsize(13)
for label in ax1.get_yticklabels():
    label.set_fontproperties(prop)
    label.set_fontsize(13)

# Plot 2: Total transmission as sum of all eigenvalues (eigenchannels)
total_transmission = np.sum(all_eigenvals_real, axis=1)
ax2.plot(w, total_transmission, 'b-', linewidth=2)

ax2.set_xlabel(r'$E_{\text{ph}}\,\text{(meV)}$', fontsize=16, fontproperties=prop)
ax2.set_ylabel(r'$\tau_{\text{ph}}$', fontsize=16, fontproperties=prop)
#ax2.legend(prop=prop)
ax2.grid(True, alpha=0.3)

for label in ax2.get_xticklabels():
    label.set_fontproperties(prop)
    label.set_fontsize(13)
for label in ax2.get_yticklabels():
    label.set_fontproperties(prop)
    label.set_fontsize(13)

# Format y-axis ticks to show only 1 decimal place on ax2
from matplotlib.ticker import FormatStrFormatter
ax2.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))

plt.tight_layout()
plt.show()

# Show statistics about eigenvalue distribution and total transmission
'''print(f"Minimum eigenvalue: {np.min(all_eigenvals_real):.6f}")
print(f"Maximum eigenvalue: {np.max(all_eigenvals_real):.6f}")
print(f"Average largest eigenvalue: {np.mean(all_eigenvals_real[:, 0]):.6f}")
print(f"Average smallest eigenvalue: {np.mean(all_eigenvals_real[:, -1]):.6f}")
print(f"Maximum total transmission: {np.max(total_transmission):.6f}")
print(f"Average total transmission: {np.mean(total_transmission):.6f}")'''