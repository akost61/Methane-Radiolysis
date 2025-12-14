import numpy as np
from numba import jit, njit
import math


ea_range = np.array([6,8,12.5])

a0_ea = np.array(
    [[2.7270E-17, 6.7363E-17]]
)

a1_ea = np.array(
    [[-2.3656E-17, -3.5794E-17]]
)

a2_ea = np.array(
    [[6.9593E-18, 6.5238E-18]]
)

a3_ea = np.array(
    [[-8.5858E-19, -4.9237E-19]]
)

a4_ea= np.array(
    [[3.8263E-20, 1.3212E-20]]
)

ea_slope = np.array( [-15.944])
ea_offset = np.array([-2.7595])

params_ea = np.array([a0_ea,a1_ea,a2_ea,a3_ea,a4_ea])
