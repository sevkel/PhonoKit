import numpy as np
from ase import Atoms
from ase.calculators.emt import EMT
from utils import constants as const
import tmoutproc as top



def get_force_constant(distance):
    # Erzeuge zwei Goldatome im Abstand distance (in Angstrom)
    atoms = Atoms('Au2', positions=[[0, 0, 0], [distance, 0, 0]])
    atoms.calc = EMT()
    # Energie für kleine Verschiebungen um den Abstand
    d = 0.01  # kleine Verschiebung in Angstrom
    atoms.positions[1, 0] = distance - d
    e1 = atoms.get_potential_energy()
    atoms.positions[1, 0] = distance
    e2 = atoms.get_potential_energy()
    atoms.positions[1, 0] = distance + d
    e3 = atoms.get_potential_energy()
    # Zweite Ableitung (numerisch)
    k = (e1 - 2 * e2 + e3) / (d ** 2)
    return k

def test1():
    for dist in [3, 6, 9]:
        k = get_force_constant(dist)
        print(f"Kraftkonstante für Abstand {dist} Å: {k:.4f} eV/Å²")
        
def dchain():
    
    
    N = 4000
    
    E_D = 70
    #w_D = (E_D * const.meV2J / const.h_bar) / const.unit2SI
    #w = np.linspace(0, w_D * 1.1, N)
    
    w_D = (E_D * const.meV2J / const.h_bar) / const.meV2J
    w = np.linspace(1E-10, E_D * 1.1, N)
    
    k_c = 20 * 1#(const.eV2hartree / const.ang2bohr ** 2)
    k_l = 100 * 1# (const.eV2hartree / const.ang2bohr ** 2)
    #E = w * const.unit2SI * const.h_bar * const.J2meV
    
    k_cc = np.array([[k_c + k_l, -k_c], 
                     [-k_c, k_c + k_l]])
    
    sigmaL = np.zeros((N, 2, 2), dtype=complex)
    sigmaR = np.zeros((N, 2, 2), dtype=complex)
    
    lambdaL = np.zeros((N, 2, 2), dtype=complex)
    lambdaR = np.zeros((N, 2, 2), dtype=complex)
    
    
    def calc_fE(w, k_l=k_l):
        return 0.5 * (w**2 - 2*k_l - w * np.sqrt(w**2 - 4*k_l, dtype=complex))
    
    def calc_gE(w, k_l=k_l):
        
        if w**2 < 4*k_l:
            return w * np.sqrt(4*k_l - w**2)
        else:
            return 0
        
    def calc_T1(w, k_c=k_c, k_l=k_l):
        return (k_c**2 * (4*k_l-w**2)) / (k_l*(4*k_c**2+w**2 * (k_l-2*k_c)))
    
    D_cc = np.zeros((N, k_cc.shape[0], k_cc.shape[1]), dtype=complex)
    D_cc_dagger = np.zeros((N, k_cc.shape[0], k_cc.shape[1]), dtype=complex)
    T = np.zeros(N)
    Tprob = np.zeros((N, 2, 2), dtype=complex)
    T1 = np.zeros(N)
    
    for i in range(N):
        sigmaL[i][0, 0] = 1
        sigmaR[i][1, 1] = 1
        
        lambdaL[i][0, 0] = 1
        lambdaR[i][1, 1] = 1
        
        sigmaL[i][0, 0] = calc_fE(w[i]) 
        sigmaR[i][1, 1] = calc_fE(w[i]) 
        
        lambdaL[i][0, 0] = calc_gE(w[i]) 
        lambdaR[i][1, 1] = calc_gE(w[i])

        D_cc[i] = np.linalg.inv((w[i] + 1E-12j) * np.eye(2) - k_cc - sigmaL[i] - sigmaR[i]) 
        D_cc_dagger[i] = np.transpose(np.conj(D_cc[i]))
        
        Tprob[i] = np.dot(np.dot(D_cc[i], lambdaL[i]), np.dot(D_cc_dagger[i], lambdaR[i]))
        eigvals, eigvecs = np.linalg.eig(Tprob[i])
        diag = np.diag(eigvals)
        T[i] = np.real(np.trace(diag))
        #T[i] = np.real(np.trace(np.dot(np.dot(D_cc[i], lambdaL[i]), np.dot(D_cc_dagger[i], lambdaR[i]))))
        T1[i] = calc_T1(w[i])
        
    
    #plot T
    import matplotlib.pyplot as plt
    plt.plot(w, T)
    plt.plot(w, T1, linestyle='--', color='red', label='T1')
    plt.xlabel('Energy (meV)')
    plt.ylabel('Transmission (T)')
    plt.title('Transmission in 1D-Kette')
    plt.xlim(0, 30)
    plt.ylim(0, 1.1)
    plt.grid()
    #plt.show()
    print(np.real(T[0:5]))
    plt.savefig("/hpc/gpfs2/scratch/u/kellerse/Masterarbeit/Dev/phonon_transport/object_oriented/src/transmission_1d_chain.pdf")
    
if __name__ == "__main__":
    print("Test 1: Kraftkonstanten für verschiedene Abstände")
    test1()
    
    print("\n1D-Kette Transmission")
    dchain()
    
    print("Fertig!")