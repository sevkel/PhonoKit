import numpy as np
import scipy.signal
import matplotlib
import matplotlib.pyplot as plt
import numpy as np


class DebeyeModel:
    def __init__(self, E, k_c, E_D):
        self.k_c = k_c
        self.E_D = E_D
        self.E = E

        self.dimension = 1
        self.g0 = self.calculate_g0(E, E_D)
        self.g = self.calculate_g(self.g0)

    def calculate_g0(self, w, w_D):
        """Calculates surface greens function according to Markussen, T. (2013). Phonon interference effects in molecular junctions. The Journal of chemical physics, 139(24), 244101 (https://doi.org/10.1063/1.4849178).

        Args:
        w (array_like): Frequency where g0 is calculated
        w_D (float): Debeye frequency

        Returns:
        g0	(array_like) Surface greens function g0
        """

        def im_g(w):
            if (w <= w_D):
                Im_g = -np.pi * 3.0 * w / (2 * w_D ** 3)
            else:
                Im_g = 0
            return Im_g

        Im_g = map(im_g, w)
        Im_g = np.asarray(list(Im_g))
        Re_g = -np.asarray(np.imag(scipy.signal.hilbert(Im_g)))
        g0 = np.asarray((Re_g + 1.j * Im_g), complex)

        return g0

    def calculate_g(self, g_0):
        """

        Args:
        g_0 (array_like): Uncoupled surface greens function

        Returns:
        g	(array_like) Surface greens function coupled by dyson equation
        """

        gamma_hb = -self.k_c
        gamma_prime = gamma_hb
        # g = np.dot(g_0, np.linalg.inv(np.identity(g_0.shape[0]) + np.dot(gamma_prime,g_0)))
        g = g_0 / (1 + gamma_prime * g_0)


        return g

if __name__ == '__main__':
    
    E_D = 20
    N = 1000
    E = np.linspace(0,1.5*E_D, N)
    k_c = 200
    k_x = 100

    def calc_tau(d_s):
        tau = 4 * k_c ** 4 * np.imag(d_s) ** 2 / (
                    (E ** 2 - 2 * k_c - 2 * k_c ** 2 * np.real(d_s)) ** 2 + (4 * k_c ** 4 * np.imag(d_s) ** 2))

        return tau

    #electrode = DebeyeModel(E, k_c, E_D)
    #d_s = electrode.g
    #plt.plot(E,np.real(d_s)*k_c, color="red", label=r"$\mathrm{Re(d_s)}$")
    #plt.plot(E, np.imag(d_s)*k_c, color="red", ls="dashed", label=r"$\mathrm{Im(d_s)}$")
    #tau = calc_tau(d_s)
    #plt.plot(E, tau, color="red", label=r"$\tau$")

    #1d chain

    d_s = 0.5*(E**2-2*k_c-E*np.sqrt(E**2-4*k_x, dtype=complex))/(E**2*(k_x-k_c)+k_c**2)
    tau = calc_tau(d_s)
    #plt.plot(E, np.real(d_s)*k_c, color="green", label=r"$\mathrm{Re(d_s)}$")
    #plt.plot(E, np.imag(d_s)*k_c, color="green", ls="dashed", label=r"$\mathrm{Im(d_s)}$")
    plt.plot(E, tau, color="green", label=r"$\tau$")
    plt.xlim(0,np.max(E))
    #plt.yscale('log')
    plt.axvline(x=E_D, color='blue', linestyle='--', label=r"$E_D$")
    plt.legend()
    plt.show()
    plt.savefig("tau_debeye1D.pdf")