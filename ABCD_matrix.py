import numpy as np


def cosd(arg):
    """
    Return the cosine of an argument

        Args:
            arg (float): angle in radian.

        Returns:
            float: cosine of an arg in degree.
    """
    return np.cos(arg * np.pi / 180.)


def sind(arg):
    """
    Return the sine of an argument

        Args:
            arg (float): angle in radian.

        Returns:
            float: sine of an arg in degree.
    """
    return np.sin(arg * np.pi / 180.)


def secd(arg):
    """
    Return the secant of an argument

        Args:
            arg (float): angle in radian.

        Returns:
            float: secant of an arg in degree.
    """
    return 1 / np.cos(arg * np.pi / 180.)


def transfer_matrix_method(stack, a0, b0, lam, theta, loss=False):
    """
    Calculate the transfer function through a Distributed Bragg reflector stack.

        Args:
            stack (object): bla.
            a0 (object): bla.
            b0 (object): bla.
            lam (object): bla.
            theta (object): bla.
            loss (bool): bla.

        Returns:
            electric_tot_te (array): Total energy of the Transverse Electric polarized light.
            electric_tot_tm (array): Total energy of the Transverse Magnetic polarized light.
            reflectivity_tot_te (array): Reflected energy of the Transverse Electric polarized light.
            reflectivity_tot_tm (array): Reflected energy of the Transverse Magnetic polarized light.
            transmission_tot_te (array): Transmitted energy of the Transverse Electric polarized light.
            transmission_tot_tm (array): Transmitted energy of the Transverse Magnetic polarized light.
            index_tot (array): Indexes of the different layers.
            length (array): Length of the stack.
            theta_tot (array): Accumulate angle thought the stack.

    """

    # Get the stack parameters
    index = stack.index
    length = stack.thickness

    # Preparation of the different arrays before calculation
    amp_te = np.array([a0, b0], dtype=complex)
    amp_tm = np.array([a0, b0], dtype=complex)
    electric_tot_te = complex(0)
    electric_tot_tm = complex(0)
    index_tot = 0
    theta_tot = 0
    theta = complex(theta)
    matrix_te = np.zeros([2, 2, len(index)], dtype=float)
    matrix_tm = np.zeros([2, 2, len(index)], dtype=complex)
    matrix_prop = np.zeros([2, 2, len(index)], dtype=complex)
    step_size = length.sum() / 2**17

    for i, re_index in enumerate(index):
        z0 = np.linspace(0, length[i], round(length[i] / step_size))
        # Calculation of the input angle
        if i > 0:
            # second and next layers
            n2 = index[i - 1] * cosd(theta)
            n2_tm = index[i - 1] * secd(theta)
            a2 = cosd(theta)
            # The angle of the beam is modified by refraction (Descartes law)
            theta = np.arcsin(np.real(re_index) / np.real(index[i - 1]) * sind(theta)) * 180. / np.pi
        else:
            # first layer
            n2 = index[i] * cosd(theta)
            n2_tm = index[i] * cosd(theta)
            a2 = cosd(theta)

        n1 = index[i] * cosd(theta)
        n1_tm = index[i] * secd(theta)
        a1 = cosd(theta)

        if loss:
            # Calculation of Fresnel coefficients (losses taken into account)
            sigma = np.real(stack.sigma)
            t12 = 2 * np.real(n1) / (np.real(n1) + np.real(n2)) * np.exp(
                -1 / 2 * (2 * np.pi * sigma[i] * (np.real(n2) - np.real(n1)) / lam) ** 2)
            t21 = 2 * np.real(n2) / (np.real(n1) + np.real(n2)) * np.exp(
                -1 / 2 * (2 * np.pi * sigma[i] * (np.real(n1) - np.real(n2)) / lam) ** 2)
            rho12 = (np.real(n1) - np.real(n2)) / (np.real(n1) + np.real(n2)) * np.exp(
                -2 * (2 * np.pi * sigma[i] * np.real(n1) / lam) ** 2)
            rho21 = (np.real(n2) - np.real(n1)) / (np.real(n1) + np.real(n2)) * np.exp(
                -2 * (2 * np.pi * sigma[i] * np.real(n2) / lam) ** 2)
            t_tm = 2 * a1 / a2 * np.real(n1_tm) / (np.real(n1_tm) + np.real(n2_tm))
            rho_tm = (np.real(n1_tm) - np.real(n2_tm)) / (np.real(n1_tm) + np.real(n2_tm))
        else:
            # Calculation of Fresnel coefficients (no losses)
            t12 = 2 * np.real(n1) / (np.real(n1) + np.real(n2))
            t21 = 2 * np.real(n2) / (np.real(n1) + np.real(n2))
            rho12 = (np.real(n1) - np.real(n2)) / (np.real(n1) + np.real(n2))
            rho21 = (np.real(n2) - np.real(n1)) / (np.real(n1) + np.real(n2))
            t_tm = 2 * a1 / a2 * np.real(n1_tm) / (np.real(n1_tm) + np.real(n2_tm))
            rho_tm = (np.real(n1_tm) - np.real(n2_tm)) / (np.real(n1_tm) + np.real(n2_tm))

        # Matrices corresponding to the propagation through the interface i for TE and TM polarized light
        matrix_te[:, :, i] = 1 / (-t12) * np.array([[1, -rho21],
                                                    [rho12, t12 * t21 - rho12 * rho21]], dtype=float)
        matrix_tm[:, :, i] = 1 / t_tm * np.array([[1, rho_tm],
                                                  [rho_tm, 1]], dtype=float)

        # Matrix corresponding to the propagation within the layer i
        phi = (2 * np.pi / lam) * re_index * length[i] * cosd(theta)
        matrix_prop[:, :, i] = np.array([[np.exp(1j * phi), 0],
                                         [0, np.exp(-1j * phi)]], dtype=complex)

        # ABCD matrices to calculate the amplitudes of TE and TM polarized light in layer i
        if i == 0:
            amp_te = amp_te
            amp_tm = amp_tm
        else:
            amp_te = np.dot(np.dot(matrix_te[:, :, i], matrix_prop[:, :, i - 1]), amp_te)
            amp_tm = np.dot(np.dot(matrix_tm[:, :, i], matrix_prop[:, :, i - 1]), amp_tm)

        # Calculation of the electric field of TE and TM polarized light (standing waves)
        k = (2 * np.pi / lam) * re_index * cosd(theta)
        electric_te = amp_te[0] * np.exp(1j * k * z0) + amp_te[1] * np.exp(-1j * k * z0)
        electric_tm = amp_tm[0] * np.exp(1j * k * z0) + amp_tm[1] * np.exp(-1j * k * z0)
        # values of the electric fields added to the previous ones
        electric_tot_te = np.hstack((electric_tot_te, electric_te))
        electric_tot_tm = np.hstack((electric_tot_tm, electric_tm))
        # value of the index added to the previous ones
        index_tot = np.hstack((index_tot, re_index))
        # value of the theta added to the previous ones
        theta_tot = np.hstack((theta_tot, theta))

    # to removed coherency, we calculate the intensity instead of the electric field
    reflectivity_tot_te = np.abs(amp_te[1] / amp_te[0]) ** 2
    reflectivity_tot_tm = np.abs(amp_tm[1] / amp_tm[0]) ** 2
    transmission_tot_te = np.abs(a0 * np.sqrt(index[0]) / (amp_te[0] * np.sqrt(index[-1]))) ** 2
    transmission_tot_tm = np.abs(a0 * np.sqrt(index[0]) / (amp_tm[0] * np.sqrt(index[-1]))) ** 2

    return electric_tot_te, electric_tot_tm, reflectivity_tot_te, reflectivity_tot_tm, transmission_tot_te, \
           transmission_tot_tm, index_tot, length, theta_tot
