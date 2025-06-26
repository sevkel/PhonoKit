import numpy as np

# h_bar in Js
h_bar = 1.0545718 * 10 ** (-34)
eV2hartree = 0.0367493
ang2bohr = 1.88973
har2J = 4.35974E-18
bohr2m = 5.29177E-11
u2kg = 1.66054E-27
har2pJ = 4.35974e-6
J2meV = 6.24150934190e+21
meV2J = 1.60217656535E-22
#This unit is calulated from the greens function: w**2 and D have to have the same unit
unit2SI = np.sqrt(9.375821464623672e+29)

if __name__ == '__main__':
    unit = np.sqrt((1/har2J)/((1/bohr2m)**2*(1/u2kg)))
    print(1/unit)
    print(unit2SI)
    print(2*np.sqrt(0.1 * (eV2hartree / ang2bohr ** 2))*unit2SI * h_bar * J2meV)