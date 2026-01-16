import numpy as np

'''
Parameters used for Electron Impact Excitation Inelastic Event cross section calculations

EXCITATION REACTIONS:
eie_1 : CH₄ + e⁻ -> CH₃* + H* + e⁻   
eie_2 : CH₄ + e⁻ -> CH₂* + H₂ + e⁻   
eie_3 : CH₄ + e⁻ -> CH* + H₂ + H* + e⁻  


COEFFICIENT STRUCTURE:
ax_eie_y: Polynomial coefficients for cross section calculations
  - x: coefficient order (0-4 for polynomial terms)
  - y: event type (1, 2, or 3)

EVENT-SPECIFIC DETAILS:
eie_1: Threshold 7.5 eV, ranges [7.5-17 eV], [17-40 eV], [40-100 eV], power tail for E > 100 eV
eie_2: Threshold 7.5 eV, ranges [7.5-22 eV], [22-40 eV], power tail for E > 40 eV
eie_3: Threshold 13 eV, range [13-90 eV], power tail for E > 90 eV

'''


range_eie_1 = np.array([7.5,17,40,100])

a0_eie_1 = np.array(
    [[1.7536E-16, -1.1553E-15, 2.0729E-16]]
)

a1_eie_1 = np.array([[
    -5.3092E-17, 1.6685E-16, -2.1828E-18]
])

a2_eie_1 = np.array([
    [5.0307E-18, -7.8001E-18, -7.6710E-22]
])

a3_eie_1 = np.array(
    [[-1.511E-19, 1.5689E-19, 7.2059E-23]]
)

a4_eie_1 = np.array(
    [[1.402E-21, -1.156E-21, 0]]
)

offset_eie_1 = np.array([-32.335])
slope_eie_1 = np.array([-1.1196])

params_eie_1 = np.array([a0_eie_1, a1_eie_1, a2_eie_1, a3_eie_1, a4_eie_1])


range_eie_2 = np.array([7.5,22,40])

a0_eie_2 = np.array(
    [[2.1296E-16, 1.2817E-16]]
)

a1_eie_2 = np.array([
    [-7.9147E-17, -3.555E-18,]
])

a2_eie_2 = np.array([[
   9.7273E-18, 1.1875E-20]
])

a3_eie_2 = np.array(
    [[-4.4331E-19, 0]]
)

a4_eie_2 = np.array(
    [[6.8159E-21, 0]]
)
offset_eie_2 = np.array([-7.9828])
slope_eie_2 = np.array([-8.636])

params_eie_2 = np.array([a0_eie_2, a1_eie_2, a2_eie_2, a3_eie_2, a4_eie_2])


range_eie_3 = np.array([13,90])
a0_eie_3 = np.array(
    [[-4.4972E-18]]
)

a1_eie_3 = np.array([
    [4.5944E-19]
])

a2_eie_3 = np.array([
   [-8.8557E-21]
])

a3_eie_3 = np.array(
    [[8.2949E-23]]
)

a4_eie_3 = np.array(
    [[-3.2217E-25]]
)
offset_eie_3 = np.array([-36.077])
slope_eie_3 = np.array([-0.8511])

params_eie_3 = np.array([a0_eie_3, a1_eie_3, a2_eie_3, a3_eie_3, a4_eie_3])