import numpy as np

import matplotlib.pyplot as plt


def calculate_coupled_sfg(E, k_x, k_y, k_xy, eps = 1E-9, eta = 1E-9, model = "1Dchain"):

    def is_symmetric(M):
        #checks if a matrix is symmetric
        return np.allclose(M, np.transpose(M))

    def check_sum_rule(M):
        return np.sum(M) == 0

    def decimation(e):
        w = np.identity(H_NN.shape[0]) * (e ** 2 + (1.j * eta))
        g = np.linalg.inv(w - H_NN)
        alpha_i = np.dot(np.dot(H_01, g), H_01)
        beta_i = np.dot(np.dot(H_01_dagger, g), H_01_dagger)
        epsilon_is = H_00 + np.dot(np.dot(H_01, g), H_01_dagger)
        epsilon_i = H_NN + np.dot(np.dot(H_01, g), H_01_dagger) + np.dot(np.dot(H_01_dagger, g), H_01)
        delta = np.abs(2 * np.trace(alpha_i))

        counter = 0
        terminated = False
        while (delta > eps):
            counter += 1
            if (counter > 10000):
                terminated = True
                break
            g = np.linalg.inv(w - epsilon_i)
            epsilon_i = epsilon_i + np.dot(np.dot(alpha_i, g), beta_i) + np.dot(np.dot(beta_i, g), alpha_i)
            epsilon_is = epsilon_is + np.dot(np.dot(alpha_i, g), beta_i)
            alpha_i = np.dot(np.dot(alpha_i, g), alpha_i)
            beta_i = np.dot(np.dot(beta_i, g), beta_i)
            delta = np.abs(2 * np.trace(alpha_i))
            #deltas.append(delta)
        if (delta >= eps or terminated):
            print("warning")

        g_0 = np.linalg.inv(w - epsilon_is)

        return g_0



    if model != "3_2_config":
        H_00 = np.array([[k_x]], dtype=complex)
        H_NN = np.array([[2*k_x]], dtype=complex)
        H_01 = np.array([[-k_x]], dtype=complex)

    else:
        H_00 = np.array([[k_x, k_xy, 0, 0, 0, 0],
                         [k_xy, k_y, 0, -k_y, 0, 0],
                         [0, 0, k_x, 2*k_xy, 0, 0],
                         [0, -k_y, 2*k_xy, 2 * k_y, 0, -k_y],
                         [0, 0, 0, 0, k_x, k_xy],
                         [0, 0, 0, -k_y, k_xy, k_y]])

        H_NN = np.array([[2 * k_x, 2*k_xy, 0, 0, 0, 0],
                         [2*k_xy, k_y, 0, -k_y, 0, 0],
                         [0, 0, 2 * k_x, 4 * k_xy, 0, 0],
                         [0, -k_y, 4 * k_xy, 2 * k_y, 0, -k_y],
                         [0, 0, 0, 0, 2 * k_x, 2*k_xy],
                         [0, 0, 0, -k_y, 2*k_xy, k_y]])

        H_01 = np.array([[-k_x, 0, 0, -k_xy, 0, 0],
                         [0, 0, -k_xy, 0, 0, 0],
                         [0, -k_xy, -k_x, 0, 0, -k_xy],
                         [-k_xy, 0, 0, 0, -k_xy, 0],
                         [0, 0, 0, -k_xy, -k_x, 0],
                         [0, 0, -k_xy, 0, 0, 0]])

    H_01_dagger = np.transpose(np.conj(H_01))

    assert is_symmetric(H_00)
    assert check_sum_rule(H_00+H_01)
    assert is_symmetric(H_NN)
    assert check_sum_rule(H_NN+2*H_01)

    g_0 = map(decimation, E)

    #convert g_0 to array
    g_0 = np.array(list(g_0))



    if model != "3_2_config":
        delta = np.array([[-k_x]])
    else:
        delta = np.array([[0, -k_xy, 0, 0, 0, 0],
                          [-k_xy, 0, 0, 0, 0, 0],
                          [0, 0, -k_x, 0, 0, 0],
                          [0, 0, 0, 0, 0, 0],
                          [0, 0, 0, 0, 0, -k_xy],
                          [0, 0, 0, 0, -k_xy, 0]])

    #be aware of jans definition in equation 2.82 -> sign
    g_0 = np.linalg.inv(np.identity(H_00.shape[0])+g_0@delta)@g_0





    return g_0

def calculate_Sigma(d, k_x, model = "1Dchain"):
    #shape (3+2)*2 x (3*2) -> 3: electrode y dimension, 2: center part dimension

    if model == "1Dchain_single_site":
        K_LC = np.array([[-k_x]])
    elif model == "1Dchain_2_sites":
        print("Using 1Dchain2 model")
        K_LC = np.array([[-k_x, 0]])
    elif model == "3_2_config":
        K_LC = np.array([[0, -k_xy, 0, 0],  # 0 x
                         [-k_xy, 0, 0, 0],  # 0 y
                         [-k_x, 0, 0, 0],  # 1 x
                         [0, 0, 0, 0],  # 1 y
                         [0, -k_xy, 0, 0],  # 2 x
                         [-k_xy, 0, 0, 0]])  # 2 y
        

    K_CL = np.transpose(np.conj(K_LC))
    Pi_l =  np.matmul(K_CL[None, :, :], d)
    Pi_l = np.matmul(Pi_l, np.transpose(np.conj(K_CL))[None, :, :])

    if model == "1Dchain_single_site":
        K_RC = np.array([[-k_x]])
    elif model == "1Dchain_2_sites":
        K_RC = np.array([[0, -k_x]])
    elif model == "3_2_config":
        K_RC = np.array([[0, 0, 0, -k_xy],  # 0 x
                         [0, 0, -k_xy, 0],  # 0 y
                         [0, 0, -k_x, 0],  # 1 x
                         [0, 0, 0, 0],  # 1 y
                         [0, 0, 0, -k_xy],  # 2 x
                         [0, 0, -k_xy, 0]])  # 2 y
        

    K_CR = np.transpose(np.conj(K_RC))
    Pi_r =  np.matmul(K_CR[None, :, :], d)
    Pi_r = np.matmul(Pi_r, np.transpose(np.conj(K_CR))[None, :, :])

    return Pi_l, Pi_r

def calculate_G_CC(E, Sigma_l, Sigma_r, k_x, k_y, eta=1E-9, model = "1Dchain"):

    if model == "1Dchain_single_site":
        K_CC = np.array([[2*k_x]])
    elif model == "1Dchain_2_sites":
        print("Using 1Dchain2 model")
        K_CC = np.array([[2*k_x, -k_x],
                         [-k_x, 2*k_x]])
    elif model == "3_2_config":
        K_CC = np.array([[2 * k_x, 2*k_xy, -k_x, 0],
                         [2*k_xy, 0, 0, 0],
                         [-k_x, 0, 2 * k_x, 2*k_xy],
                         [0, 0, 2*k_xy, 0]])

    G_CC = np.linalg.inv(np.identity(K_CC.shape[0])[None, :, :] * (E[:, None, None]+ (1.j * eta))**2 - K_CC[None, :, :] - Sigma_l - Sigma_r)
    return G_CC

def calculate_1d_coupled_sfg(E, k_x):
    """
    analytic solution for the 1D coupled SFG
    :param E:
    :param k_x:
    :return:
    """
    return 0.5*(E**2-2*k_x-E*np.sqrt(E**2-4*k_x, dtype=complex))/(k_x**2)

def calculate_1d_uncoupled_sfg(E, k_x):
    """
    analytic solution for the 1D uncoupled SFG
    """
    return 1/(2*k_x*E) * (E - np.sqrt(E**2 - 4*k_x, dtype=complex))




if __name__ == '__main__':
    k_x = 100
    k_y = 100
    k_xy = 20
    E = np.linspace(1E-3, 25, 500)
    #model = "1Dchain_single_site"
    #model = "1Dchain_2_sites"
    model = "3_2_config"

    d = calculate_coupled_sfg(E, k_x, k_y, k_xy, model = model)

    dos = (-1/np.pi)*np.imag(np.trace(d, axis1=1, axis2=2))
    dos_real = np.real(np.trace(d, axis1=1, axis2=2))

    plt.plot(E, dos)
    plt.plot(E, dos_real)
    plt.plot(E, dos_real, label="real part")
    plt.plot(E, dos, label="dos")
    plt.legend()
    #plt.savefig(r'C:\Users\sevke\Desktop\Dev\MA\phonokit\src\plot\phonon_model_check_DOS.pdf', bbox_inches='tight')
    #plt.show()
    #plt.clf()

    #"""
    Sigma_l, Sigma_r = calculate_Sigma(d, k_x, model = model)
    Pi_l = -2 * np.imag(Sigma_l)
    Pi_r = -2 * np.imag(Sigma_r)

    G_CC = calculate_G_CC(E, Sigma_l, Sigma_r, k_x, k_y, model = model)

    tau = np.trace(np.matmul(np.matmul(G_CC, Pi_l), np.matmul(np.conj(np.transpose(G_CC, axes=(0,2,1))), Pi_r)), axis1=1, axis2=2)

    plt.plot(E, np.real(tau))
    plt.ylim(0,1.5)
    #plt.savefig(r'C:\Users\sevke\Desktop\Dev\MA\phonokit\src\plot\phonon_model_check.pdf', bbox_inches='tight')
    plt.show()

    print('debug')




    #"""