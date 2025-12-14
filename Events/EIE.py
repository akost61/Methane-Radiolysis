import numpy as np
from numba import jit, njit
import math

eie_1_range = np.array([7.5,17,40,100])

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

eie_1_offset = np.array([-32.335])
eie_1_slope = np.array([-1.1196])

params_eie_1 = np.array([a0_eie_1, a1_eie_1, a2_eie_1, a3_eie_1, a4_eie_1])


eie_2_range = np.array([7.5,22,40])

a0_eie_2 = np.array(
    [[2.1296E-16, 1.2817E-16]]
)

a1_eie_2 = np.array([[
    -7.9147E-17, -3.555E-18,]
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
eie_2_offset = np.array([-7.9828])
eie_2_slope = np.array([-8.636])

params_eie_2 = np.array([a0_eie_2, a1_eie_2, a2_eie_2, a3_eie_2, a4_eie_2])


eie_3_range = np.array([13,80])
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
eie_3_offset = np.array([-36.077])
eie_3_slope = np.array([-0.8511])

params_eie_3 = np.array([a0_eie_3, a1_eie_3, a2_eie_3, a3_eie_3, a4_eie_3])





