import numpy as np
import matplotlib.pyplot as plt
from dbr_coating.ABCD_matrix import transfer_matrix_method
from dbr_coating.stack import Stack


def get_spectral_response(wavelengths_arr, stack):
    """
        Realize ABCD matrices calculations on a range of wavelengths to get the spectral response of a coating stack

            Args:
                wavelengths_arr (array): wavelengths for which the spectral response is calculated.
                stack (object) : Stack that defines the index and the thickness of each layer of the coating.

            Returns:
                reflectivity_te (array) : reflectivity for TE polarized light for each wavelengths specified
                transmission_te (array) : transmission for TE polarized light for each wavelengths specified
        """

    resolution = 1
    for i, re_index in enumerate(stack.index):
        step_size = stack.thickness.sum() / 2 ** 17
        z0 = np.linspace(0, stack.thickness[i], round(stack.thickness[i] / step_size))
        resolution += len(z0)

    electric_tot_te = np.zeros([resolution, len(wavelengths_arr)], dtype=complex)
    electric_tot_tm = np.zeros([resolution, len(wavelengths_arr)], dtype=complex)
    reflectivity_te = np.zeros(len(wavelengths_arr), dtype=complex)
    reflectivity_tm = np.zeros(len(wavelengths_arr), dtype=complex)
    transmission_te = np.zeros(len(wavelengths_arr), dtype=complex)
    transmission_tm = np.zeros(len(wavelengths_arr), dtype=complex)
    index_tot = np.zeros([resolution, len(wavelengths_arr)], dtype=complex)
    theta_tot = np.zeros([len(stack.index) + 1, wavelengths_arr.size], dtype=complex)

    a0 = 1  # Initial amplitude of electric field going toward the coating
    b0 = 0  # Initial amplitude of electric field going back the coating (if 0, no counter propagating light)
    theta = 0  # angle of the beam with respect to the coating

    for i, lam in enumerate(wavelengths_arr):
        # print a progressbar in the console
        print_progressbar(i, len(wavelengths_arr), suffix = '%')
        electric_tot_te[:, i], electric_tot_tm[:, i], reflectivity_te[i], reflectivity_tm[i], transmission_te[i], \
        transmission_tm[i], index_tot, L, theta_tot = transfer_matrix_method(stack, a0, b0, lam, theta)
    return reflectivity_te, transmission_te, 1 - (reflectivity_te + transmission_te)



def load_stack(filename):
    """
    Load a stack configuration from a .txt file.

        Args:
            filename (string): filename of the tabular where the indexes and thicknesses of the layers are contained.

        Returns:
            stack (object) : Stack that defines the index and the thickness of each layer of the coating.
    """
    data = np.genfromtxt(filename, skip_header=1)
    index_arr = data[:, 2]
    thickness_arr = data[:, 3] / 1e9
    stack = Stack(index_arr, thickness_arr)
    return stack


def print_progressbar (iteration, total, prefix = '', suffix = '', decimals = 1, length = 100, fill = '█', printEnd = "\r"):
    """
    Call in a loop to create terminal progress bar
        Args:
            iteration   - Required  : current iteration (Int)
            total       - Required  : total iterations (Int)
            prefix      - Optional  : prefix string (Str)
            suffix      - Optional  : suffix string (Str)
            decimals    - Optional  : positive number of decimals in percent complete (Int)
            length      - Optional  : character length of bar (Int)
            fill        - Optional  : bar fill character (Str)
            printEnd    - Optional  : end character (e.g. "\r", "\r\n") (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print(prefix, bar, percent, suffix)
    print('{} |{}| {} {}'.format(prefix, bar, percent, suffix), end=printEnd)
    # Print New Line on Complete
    if iteration == total:
        print()

def plot_intensity_prop(stack, wavelengths_arr, colors_arr):
    """ Get the intensity profile inside the stack

        Args:
            stack (object): Stack that defines the index and the thickness of each layer of the coating.
            wavelengths_arr (list): List of the wavelengths to consider.
            colors_arr (list): List of color names to represent each wavelengths propagation on the plot

        Returns:
            None

    """

    # for the electric fields profile
    for i, wl in enumerate(wavelengths_arr):
        electric_tot_te, electric_tot_tm, reflectivity_te, reflectivity_tm, transmission_te, transmission_tm, index_tot, L_tot, theta_tot = transfer_matrix_method(
            stack, 1, 0, wl, 0)
        intensity = np.abs(electric_tot_te[::-1]) ** 2
        plt.plot(L_tot * 1e6, intensity / max(intensity) * 2, color=colors_arr[i])
    # for the indexes profile
    ax.plot(L_tot * 1e6, index_tot[::-1], color='black')
    ax.fill_between(L_tot * 1e6, index_tot[::-1], color='azure')


""" Start of the simulations """

# loading of the different stacks to study
filename = 'stack_example.txt'
coating_stack = load_stack(filename)

# Intensity propagation inside the stacks
fig, ax = plt.subplots()
ax.set_title('Intensity inside stack')
ax.set_xlabel('L [mm]')
ax.set_ylabel('Refractive index')
ax.grid(color='black', linestyle='--', linewidth=0.5)
plot_intensity_prop(stack=coating_stack, wavelengths_arr=[637e-9, 602e-9, 532e-9], colors_arr=['red', 'orange', 'green'])
plt.savefig('intensity_propagation.pdf', orientation='portrait', format='pdf')
plt.show()


# spectral response of the stacks
wl = np.linspace(450e-9, 800e-9, 401)
reflectivity_te_diamond, transmission_te_diamond, loss_te_diamond = get_spectral_response(wl, coating_stack)

fig, ax = plt.subplots()
ax.plot(wl * 1e9, np.real(transmission_te_diamond), label=filename, color='blue')
ax.set_title('Spectral response')
ax.set_xlabel('Wavelength [nm]')
ax.set_ylabel('Amplitude')
ax.grid(color='black', linestyle='--', linewidth=0.5)
ax.legend(loc='upper right')
plt.savefig('spectral_response.pdf', orientation='portrait', format='pdf')
plt.show()
