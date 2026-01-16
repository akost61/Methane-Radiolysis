import numpy as np

'''
- Parameters used for Molecular Excitation cross section calculations.
- ax_nu: vibrational inelastic events coefficients.
- ax_j: rotational inelastic events coefficients.
- The first index associated with the type of molecular excitation inelastic event
- The second index is associated with what range the energy falls between.

Any energy above the highest range value will calculate a power tail with the slope and offset values.

VIBRATIONAL EXCITATION (nu):
- Four vibrational modes: Nu1, Nu2, Nu3, Nu4
- Energy ranges defined by range_nu: [0-5 eV], [5-10 eV], [10-17 eV]
- Energy above 17 eV will use the power law tail.

ROTATIONAL EXCITATION (j):
- Two rotational transitions: J0→3, J0→4 (ground state J=0 to excited states J=3,4)
- Energy ranges defined by range_j: [0-7.5 eV], [7.5-30 eV]
- Energy above 30 eV will use the power law tail.


'''


range_nu = np.array([0, 5, 10, 17])

a0_nu = np.array([
    [9.5775E-18, 1.2176E-16, 1.9796E-16],   # Nu1
    [1.1232E-17, 2.5438E-16, 1.5228E-16],   # Nu2
    [2.3026E-17, 2.8946E-16, 1.3383E-16],   # Nu3
    [2.7555E-17, 2.33E-16, 3.5303E-17]      # Nu4
])

a1_nu = np.array([
    [-8.9135E-18, -8.6348E-17, -3.4049E-17],
    [-6.0423E-18, -1.618E-16, -1.9841E-17],
    [-1.2559E-17, -1.8694E-16, -1.149E-17],
    [-1.004E-17, -1.3779E-16, 3.2174E-18]
])

a2_nu = np.array([
    [3.3126E-18, 2.2772E-17, 2.0965E-18],
    [2.2599E-18, 3.8308E-17, 1.0086E-18],
    [3.3263E-18, 4.4288E-17, 3.2806E-19],
    [2.1595E-18, 3.0494E-17, -3.7628E-19]
])

a3_nu = np.array([
    [-2.1225E-19, -2.2987E-18, -4.3665E-20],
    [-1.008E-19, -3.622E-18, -1.8137E-20],
    [-1.2141E-19, -4.1437E-18, -2.3848E-21],
    [-9.2766E-20, -2.7257E-18, 9.5309E-21]
])

a4_nu = np.array([
    [0, 7.8662E-20, 0.0],
    [0, 1.1914E-19, 0.0],
    [0, 1.344E-19, 0.0],
    [0, 8.6051E-20, 0.0]
])

# slope arrays (for ranges >= 17)
slope_nu = np.array([-0.76715, -1.3092, -1.5041, -0.76174])
offset_nu = np.array([-36.926, -34.883, -34.149, -35.94])


params_nu = np.array([
    a0_nu,a1_nu,a2_nu,a3_nu, a4_nu
])

# J0→ ranges (polynomial ranges)
range_j = np.array([0, 7.5, 30])

# a0 arrays
a0_j = np.array([
    [4.0187E-17, 3.8926E-17],   # J0→3
    [-1.4292E-18, -5.9277E-16]   # J0→4
])

# a1 arrays
a1_j = np.array([
    [-3.4332E-17, 3.3496E-18],
    [4.8739E-18, 1.577E-16]
])

# a2 arrays
a2_j = np.array([
    [9.5048E-18, 1.8872E-19],
    [-3.8881E-18, -1.1631E-17]
])

# a3 arrays
a3_j = np.array([
    [-5.8352E-19, -6.3879E-21],
    [1.276E-18, 3.5567E-19]
])

# a4 arrays
a4_j = np.array([
    [0.0, 0.0],
    [-8.8977E-20, -3.9501E-21]
])

slope_j = np.array([-0.75386, -1.2547])
offset_j = np.array([-33.963, -32.873])

params_j = np.array([
    a0_j, a1_j, a2_j, a3_j, a4_j]
    )

