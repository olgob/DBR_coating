import numpy as np
import matplotlib.pyplot as plt
from ABCD_matrix import transfer_matrix_method
from stack import Stack


def get_spectral_response(wavelengths, stack):
    """
        Realize ABCD matrices calculation to get the spectral response of a coating stack

            Args:
                wavelengths (array): wavelengths for which the spectral response is calculated.
                stack (object) : Stack that defines the index and the thickness of each layer of the coating.

            Returns:
                reflectivity_te (array) : reflectivity for TE polarized light for each wavelengths specified
                transmission_te (array) : transmission for TE polarized light for each wavelengths specified
        """

    resolution = 1
    for i, re_index in enumerate(stack.index):
        step_size = stack.thickness.sum() / 2**17
        z0 = np.linspace(0, stack.thickness[i], round(stack.thickness[i] / step_size))
        resolution += len(z0)

    electric_tot_te = np.zeros([resolution, len(wavelengths)], dtype=complex)
    electric_tot_tm = np.zeros([resolution, len(wavelengths)], dtype=complex)
    reflectivity_te = np.zeros(len(wavelengths), dtype=complex)
    reflectivity_tm = np.zeros(len(wavelengths), dtype=complex)
    transmission_te = np.zeros(len(wavelengths), dtype=complex)
    transmission_tm = np.zeros(len(wavelengths), dtype=complex)
    index_tot = np.zeros([resolution, len(wavelengths)], dtype=complex)
    theta_tot = np.zeros([len(stack.index) + 1, wavelengths.size], dtype=complex)

    a0 = 1  # Initial amplitude of electric field going toward the coating
    b0 = 0  # Initial amplitude of electric field going back the coating (if 0, no counter propagating light)
    theta = 0  # angle of the beam with respect to the coating

    for i, lam in enumerate(wavelengths):
        electric_tot_te[:, i], electric_tot_tm[:, i], reflectivity_te[i], reflectivity_tm[i], transmission_te[i], \
        transmission_tm[i], index_tot, L, theta_tot = transfer_matrix_method(stack, a0, b0, lam, theta)
    return reflectivity_te, transmission_te, 1 - (reflectivity_te + transmission_te)


def load_stack(filename):
    data = np.genfromtxt(filename, skip_header=1)
    index_arr = data[:, 2]
    thickness_arr = data[:, 3] / 1e9
    stack = Stack(index_arr, thickness_arr)
    return stack


coating = load_stack('stack_example.txt')
wl = np.linspace(500e-9, 800e-9, 401)
reflectivity_te, transmission_te, loss_te = get_spectral_response(wl, coating)

fig, ax = plt.subplots()
ax.plot(wl * 1e9, np.real(reflectivity_te), label='reflectivity')
ax.plot(wl * 1e9, np.real(transmission_te), label='transmission')
ax.plot(wl * 1e9, np.real(loss_te), label='losses')
ax.set_title('Spectral response')
ax.set_xlabel('Wavelength [nm]')
ax.set_ylabel('Amplitude')
ax.grid(color='black', linestyle='--', linewidth=0.5)
ax.legend(loc='upper right')
plt.show()
