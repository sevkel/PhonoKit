import os
import numpy as np
import calculate_kappa as ck
from utils import constants as const
import tmoutproc as top

def calc_kappa(T, w, temperature, data_path):
    kappa = list()
    
    # w to SI
    w_kappa = w * const.unit2SI
    E = const.h_bar * w_kappa

    # joule to hartree
    E = E / const.har2J

    valid_indices = ~np.isnan(T)

    if False in valid_indices:
        notespath = os.path.join(data_path, "notes.txt")
        if not os.path.exists(notespath):
            with open(notespath, "w") as f:
                f.write("Invalid data points found in T or E!")
                f.close()

    valid_T = T[valid_indices]
    valid_E = E[valid_indices]

    for j in range(0, len(temperature)):
        #kappa.append(ck.calculate_kappa(self.T[1:len(self.T)], E[1:len(E)], self.temperature[j]) * const.har2pJ)
        kappa.append(ck.calculate_kappa(valid_T[1:len(valid_T)], valid_E[1:len(valid_E)], temperature[j]) * const.har2pJ)

    return kappa


if __name__ == "__main__":

    main_path = r"C:\Users\sevke\Desktop\Dev\MA\phonokit\plot"
    temperature = np.linspace(0.001, 300, 1000)
    systems = ['2y_2y4x_2y', '3y_1y4x_1y', '3y_1y4x_3y', '3y_3y4x_3y']
    #systems = ['2y_2y4x_2y']

    for folder in os.listdir(main_path):
        if folder in systems:
            transpath = os.path.join(main_path, folder, "trans")
            #print(transpath)
            for dat in os.listdir(transpath):

                filename_kappa = dat.split('.')[0] + "_KAPPA.dat"

                datapath = os.path.join(transpath, dat)
                data = np.loadtxt(datapath, skiprows=1)
                w = data[:, 0]
                tau_ph = data[:, 1]

                kappa = calc_kappa(tau_ph, w, temperature, os.path.join(main_path, folder))

                kappa_path = os.path.join(main_path, folder, "kappa", filename_kappa)

                top.write_plot_data(kappa_path, (temperature, kappa), "T (K), kappa (pW/K)")