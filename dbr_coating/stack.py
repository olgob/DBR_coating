import numpy as np

class Stack:
    """
    Class of a DBR stack.

    Attributes:
        index_arr (array): list of the consecutive layers index of the stack
        thickness_arr (array): list of the consecutive layers thickness of the stack
        sigma (float) : losses induces by surface roughness of the stack
    """
    def __init__(self, index_arr, thickness_arr, sigma=None):
        """ Initialization of the class. """
        self.index = index_arr
        self.thickness = thickness_arr
        if sigma is None:
            self.sigma = np.zeros_like(index_arr)
        else:
            self.sigma = np.array(np.real(sigma), dtype=float)

    def __add__(self, other):
        """ Add stacks."""
        # the stacks are added one after the other
        new_index = np.hstack([self.index, other.n])
        new_thickness = np.hstack([self.thickness, other.L])
        new_sigma = np.hstack([self.sigma, other.sigma])
        return Stack(new_index, new_thickness, new_sigma)

    def __invert__(self):
        """ Invert the order of the stack."""
        new_index = self.index[::-1]
        new_thickness = self.thickness[::-1]
        new_sigma = self.sigma[::-1]
        return Stack(new_index, new_thickness, new_sigma)

    def set_sigma(self, i, sigma):
        """ Set the value of losses sigma of the layer i of the stack."""
        self.sigma[i] = sigma

    def get_sigma(self):
        """ Get the value of losses sigma of the layer i of the stack."""
        return self.sigma

    def set_index(self, index_arr):
        """ Set the values of the indexes of the different layers of the stack."""
        self.index = index_arr
        return

    def get_index(self):
        """ Get the values of the indexes of the different layers of the stack."""
        return self.index

    def set_thickness(self, thickness_arr):
        """ Set the values of the indexes of the different layers of the stack."""
        self.index = thickness_arr
        return

    def get_thickness(self):
        """ Get the values of the thicknesses of the different layers of the stack."""
        return self.thickness



