## Description

**What is this repo or project?**  
This project is a software used to calculate the spectral response of a [Distributed_Bragg_reflector](https://en.wikipedia.org/wiki/Distributed_Bragg_reflector), but can be used as well to simulate the behaviour of optical cavities compozed by several mirrors.

* **How does it work?**  
The repository is compozed by 2 folders. The core of the program is located in distribution folder. A test can be run using the files int the test folder.  
A DBR stack object needs first to be defined where the index and the thickness of each layer of the DBR coating are described.  
The ABCD_matrix_method.py provides the calculation to get the reflectivity, tranmission, and losses due to surface roughness if specified.

* **Who will use this repo or project?**  
The project is adressed to people who want to get the coating performance of a specific design. The example is using a dummy stack where each layer thickness is equal to lambda / (4n) where lambda is the wavelength of the light and n is the real part of the refractive index of the layer.

* **What is the goal of this project?**   
The goal of this project is to provide a python package that ease the design of a desired coating and be able to find the best parameters to get a desired reflectivity at a certain wavelength. 

## How to install the library

To be continued