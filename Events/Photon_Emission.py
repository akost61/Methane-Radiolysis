import numpy as np
'''
Parameters used for Photoionization Inelastic Event cross section calculations.

PHOTOIONIZATION REACTIONS:
ax_pho parameters correspond to the following photoionization channels:
[0]  :  'Ly-α',
[1]  : 'Ly-β',
[2]  :  'Ly-γ',
[3]  :  'H-α',
[4]  : 'H-β',
[5]  :  'H-γ',
[6]  :  'H-δ',
[7]  :  'CH G-band',
[8]  :  'C III',
[9]  :  'C I',
[10] :  'C IV'



COEFFICIENT STRUCTURE:
ax_pho: Parameters for cross section calculations using empirical fit functions
  - x: parameter number (1-8 for the photoionization fit formula)


ENERGY THRESHOLDS AND RANGES:
E_th_pho: Photoionization threshold energies (in eV) for each reaction channel
  - Represents the minimum photon energy required to initiate each photoionization process
  - Used for graphical representation of cross section onset
E_max_pho: Upper energy bound for empirical fit region (in eV)
  - Varies by channel due to different physical behavior at high photon energies

CROSS SECTION CALCULATION:
Within the fit range (E_th to E_max):
  - Uses empirical formula with parameters a1-a8
  - Functional form captures photoionization probability vs photon energy
  - Extended formula (with a7, a8) used for channels requiring more complex energy dependence
  
FOR E>E_max:
use behavior of Bethe-Born theory for inelastic processes:
  - ln(σ) = slope_pho * ln(E) + offset_pho

'''


E_th_pho = np.array([
    1.47E-02, 1.66E-02, 1.73E-02, 1.66E-02, 1.73E-02,1.76E-02,1.76E-02,1.220e-02,
    2.350e-02, 1.570e-02,2.500e-2
])

E_max_pho = np.array(
    [0.986, 0.988, 0.984, 6.0, 6.0, 6.0, 6.0, 5.0, 0.4, 0.98, 1.0]
)

a1_pho = np.array(
    [8.54e-3, 5.21e-2, 2.24e-2, 
     1.33e-2, 1.55e-2, 1.33e-3, 4.05e-4, 1.07, 1.08e-2, 6.88e-4, 1.82e-3]
)

a2_pho = np.array(
    [2.82, 4.65, 4.21, 2.41, 3.54, 1.98, 2.31, 16.0, 4.35, 6.55, 2.2]
)

a3_pho = np.array(
    [0.0272, 0.00591, 0.0058, 0.0104, 0.0064, 0.023, 0.0249, 0.0084, 0.00808, 0.0172, 0.0116]
)

a4_pho = np.array(
    [-0.19, -1.24, -1.12, -1.19, -1.06, -0.4, -0.32, -1.191, -0.23, 0.21, 0.95]
)

a5_pho = np.array(
    [0.0595, 0.0166, 0.0178, 0.0335, 0.019, 0.0408, 0.0406, 0.01184, 0.0137, 0.025, 0.0187]
)

a6_pho = np.array(
    [1.6, 1.1, 1.16, 1.04, 1.05, 1.08, 1.11, 0.909, 1.81, 1.8, 0.95]
)


a7_pho = np.array(
    [0,0,0,0,0,0,0,3.270,1.745, 1.22, 1.82]
)

a8_pho = np.array(
    [0,0,0,0,0,0,0,2.77e-1,4.99, 3.16, 5.85]
)


slope_pho = np.array(
    [-1.5044, -1.1107, -1.1703, -1.0436, -1.0557, -1.0848, -1.1058, -0.91092, -1.3889, -1.4876, -0.97797]
)

offset_pho = np.array(
    [-31.591, -35.694, -36.075, -35.305, -36.695, -37.261, -38.033, -36.036, -35.749, -34.29, -38.783]
)


params_pho = np.array([
    a1_pho,a2_pho,a3_pho,a4_pho,a5_pho,a6_pho,a7_pho,a8_pho, E_th_pho, E_max_pho
])
