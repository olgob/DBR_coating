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


def transfer_matrix_method(stack, a0, b0, wavelength, theta, loss=False):
    """
    Calculate the transfer function through a Distributed Bragg reflector stack.

        Args:
            stack (object): Stack that defines the index and the thickness of each layer of the coating.
            a0 (object): Initial amplitude of the incident electric field.
            b0 (object): Initial amplitude of the reflected electric field.
            wavelength (float): Wavelength of the light propagating through the stack.
            theta (float): Initial angle of the electric field with respect to the normal of the stack.
            loss (bool): Include losses in the calculation if True.

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
    index = stack.get_index()
    length = stack.get_thickness()

    # Preparation of the different arrays before calculation
    amp_te = np.array([a0, b0], dtype=complex)
    amp_tm = np.array([a0, b0], dtype=complex)
    electric_tot_te = complex()
    electric_tot_tm = complex()
    theta_tot = 0
    theta = complex(theta)
    matrix_te = np.zeros([2, 2, len(index)], dtype=float)
    matrix_tm = np.zeros([2, 2, len(index)], dtype=complex)
    matrix_prop = np.zeros([2, 2, len(index)], dtype=complex)
    step_size = length.sum() / 2 ** 17
    length_tot = np.array([0])
    index_tot = np.array([1.4812])

    for i, re_index in enumerate(index):
        length_arr = np.linspace(0, length[i], round(length[i] / step_size))  # length array of the layer i

        # Calculation of the input angle and indexes
        if i == 0:
            # first layer
            n_j = index[i] * cosd(theta)
            n_j_tm = index[i] * cosd(theta)
            angle_j = cosd(theta)
        else:
            # second and next layers
            n_j = index[i - 1] * cosd(theta)
            n_j_tm = index[i - 1] * secd(theta)
            angle_j = cosd(theta)
            # The angle of the beam is modified by refraction (Descartes law)
            theta = np.arcsin(np.real(re_index) / np.real(index[i - 1]) * sind(theta)) * 180. / np.pi

        n_i= index[i] * cosd(theta)
        n_i_tm = index[i] * secd(theta)
        angle_i = cosd(theta)

        if loss:
            # Calculation of Fresnel coefficients (losses taken into account)
            sigma = np.real(stack.sigma)
            t_ij = 2 * np.real(n_i) / (np.real(n_i) + np.real(n_j)) * np.exp(
                -1 / 2 * (2 * np.pi * sigma[i] * (np.real(n_j) - np.real(n_i)) / wavelength) ** 2)
            t_ji = 2 * np.real(n_j) / (np.real(n_i) + np.real(n_j)) * np.exp(
                -1 / 2 * (2 * np.pi * sigma[i] * (np.real(n_i) - np.real(n_j)) / wavelength) ** 2)
            r_ij = (np.real(n_i) - np.real(n_j)) / (np.real(n_i) + np.real(n_j)) * np.exp(
                -2 * (2 * np.pi * sigma[i] * np.real(n_i) / wavelength) ** 2)
            r_ji = (np.real(n_j) - np.real(n_i)) / (np.real(n_i) + np.real(n_j)) * np.exp(
                -2 * (2 * np.pi * sigma[i] * np.real(n_j) / wavelength) ** 2)
            t_tm = 2 * angle_i / angle_j * np.real(n_i_tm) / (np.real(n_i_tm) + np.real(n_j_tm))
            rho_tm = (np.real(n_i_tm) - np.real(n_j_tm)) / (np.real(n_i_tm) + np.real(n_j_tm))
        else:
            # Calculation of Fresnel coefficients (no losses)
            t_ij = 2 * np.real(n_i) / (np.real(n_i) + np.real(n_j))
            t_ji = 2 * np.real(n_j) / (np.real(n_i) + np.real(n_j))
            r_ij = (np.real(n_i) - np.real(n_j)) / (np.real(n_i) + np.real(n_j))
            r_ji = (np.real(n_j) - np.real(n_i)) / (np.real(n_i) + np.real(n_j))
            t_tm = 2 * angle_i / angle_j * np.real(n_i_tm) / (np.real(n_i_tm) + np.real(n_j_tm))
            rho_tm = (np.real(n_i_tm) - np.real(n_j_tm)) / (np.real(n_i_tm) + np.real(n_j_tm))

        # Matrices corresponding to the propagation through the interface i for TE and TM polarized light
        matrix_te[:, :, i] = 1 / (-t_ij) * np.array([[1, -r_ji],
                                                     [r_ij, t_ij * t_ji - r_ij * r_ji]], dtype=float)
        matrix_tm[:, :, i] = 1 / t_tm * np.array([[1, rho_tm],
                                                  [rho_tm, 1]], dtype=float)

        # Matrix corresponding to the propagation within the layer i
        phi = (2 * np.pi / wavelength) * re_index * length[i] * cosd(theta)
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
        k = (2 * np.pi / wavelength) * re_index * cosd(theta)
        electric_te = amp_te[0] * np.exp(1j * k * length_arr) + amp_te[1] * np.exp(-1j * k * length_arr)
        electric_tm = amp_tm[0] * np.exp(1j * k * length_arr) + amp_tm[1] * np.exp(-1j * k * length_arr)
        # values of the electric fields added to the previous ones
        electric_tot_te = np.hstack((electric_tot_te, electric_te))
        electric_tot_tm = np.hstack((electric_tot_tm, electric_tm))
        # value of the index added to the previous ones
        index_tot = np.hstack((index_tot, re_index))
        # value of the theta added to the previous ones
        theta_tot = np.hstack((theta_tot, theta))
        # value of the length of the layer added to the total length

        length_tot = np.append(length_tot, length_tot[-1] + length_arr)
        index_tot = np.append(index_tot, [n_i] * (len(length_arr) - 1))

    # we calculate the intensity instead of the electric field (phase information removed)
    reflectivity_tot_te = np.abs(amp_te[1] / amp_te[0]) ** 2
    reflectivity_tot_tm = np.abs(amp_tm[1] / amp_tm[0]) ** 2
    transmission_tot_te = np.abs(a0 * np.sqrt(index[0]) / (amp_te[0] * np.sqrt(index[-1]))) ** 2
    transmission_tot_tm = np.abs(a0 * np.sqrt(index[0]) / (amp_tm[0] * np.sqrt(index[-1]))) ** 2
    return electric_tot_te, electric_tot_tm, reflectivity_tot_te, reflectivity_tot_tm, transmission_tot_te, \
           transmission_tot_tm, index_tot, length_tot, theta_tot
