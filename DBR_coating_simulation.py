import numpy as np
import matplotlib.pyplot as plt
import ABCD_matrix
import stack


def loadCoatingData(filename):
    SiO2Index = 1.457
    Ta2O5Index = 2.071
    airIndex = 1
    substrateIndex = 1.46

    layers_data = np.loadtxt(filename, dtype='str')
    layerNumber = layers_data[:, 0].astype(int)
    layerMaterial = layers_data[:, 1]
    layerThickness = layers_data[:, 2].astype(float)
    layerThickness = layerThickness * 1e-9
    layerIndex = np.zeros_like(layerMaterial, dtype=float)
    n = 0
    while n < len(layerIndex):
        if layerMaterial[n] == 'SiO2':
            layerIndex[n] = SiO2Index
        elif layerMaterial[n] == 'Ta2O5':
            layerIndex[n] = Ta2O5Index
        elif layerMaterial[n] == 'Air':
            layerIndex[n] = airIndex
        elif layerMaterial[n] == 'Substrate':
            layerIndex[n] = substrateIndex
        else:
            layerIndex[n] = 0
        n += 1
    return layerIndex, layerThickness


def get_coating_result(wavelengths, stack):
    resolution = 1
    for i, re_index in enumerate(stack.index):
        step_size = stack.thickness.sum() / 2**17
        z0 = np.linspace(0, stack.thickness[i], round(stack.thickness[i] / step_size))
        resolution += len(z0)
    EtotTE = np.zeros([resolution, wavelengths.size], dtype=complex)
    EtotTM = np.zeros([resolution, wavelengths.size], dtype=complex)
    RefTE = np.zeros(wavelengths.size, dtype=complex)
    RefTM = np.zeros(wavelengths.size, dtype=complex)
    TraTE = np.zeros(wavelengths.size, dtype=complex)
    TraTM = np.zeros(wavelengths.size, dtype=complex)
    ntot = np.zeros([resolution, wavelengths.size], dtype=complex)
    thetatot = np.zeros([len(stack.index) + 1, wavelengths.size], dtype=complex)

    A0 = 1
    B0 = 0
    theta = 0
    for i, lam in enumerate(wavelengths):
        print(i / 1601 * 100, '%')
        EtotTE[:, i], EtotTM[:, i], RefTE[i], RefTM[i], TraTE[i], TraTM[i], ntot, L, thetatot \
            = ABCD_matrix.transfer_matrix_method(stack, A0, B0, lam, theta)
    return RefTE, TraTE, 1 - (RefTE + TraTE)


n_DTUflat_mirror, L_DTUflat_mirror = loadCoatingData('DTU_flat_coating.txt')
DTU_DBR_stack_flat = stack.Stack(index_arr=n_DTUflat_mirror, thickness_arr=L_DTUflat_mirror)
DTU_DBR_stack_flat_reversed = stack.Stack(index_arr=n_DTUflat_mirror[::-1], thickness_arr=L_DTUflat_mirror[::-1])
wavelengths = np.linspace(550e-9, 750e-9, 1601)
RefTE, TraTE, LossTE = get_coating_result(wavelengths, DTU_DBR_stack_flat)
# RefTE_reversed, TraTE_reversed, LossTE_reversed = get_coating_result(wavelengths, DTU_DBR_stack_flat_reversed)

plt.plot(wavelengths * 1e9, RefTE)
# plt.plot(wavelengths*1e9, RefTE_reversed, 'r--')
plt.show()
