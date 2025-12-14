import math
from numba import jit, njit, prange
import numpy as np
import random

from Events.Ionization import params_ion, slope_ion, offset_ion
from Events.Molecular_Excitation import Nu_ranges, params_nu, J_ranges, params_j, slope_nu, slope_j, offset_j, offset_nu
from Events.EIE import eie_1_range, eie_2_range, eie_3_range, params_eie_1,params_eie_2,params_eie_3, eie_1_offset, eie_1_slope, eie_2_offset, eie_2_slope, eie_3_offset, eie_3_slope
from Events.Electron_Attachment import params_ea, ea_range, ea_offset, ea_slope
from Events.Photon_Emission import params_pho, slope_pho, offset_pho

events = np.array(['Ion_1', 'Ion_2', 'Ion_3', 'Ion_4', 'Ion_5', 'Ion_6','Ion_7','EIE_1', 'EIE_2', 'EIE_3', 'EA',
          'Nu1', 'Nu2', 'Nu3', 'Nu4', 'Jto3', 'Jto4', 'Ly_a', 'Ly_b', 'Ly_g', 'H_a', 'H_b',
          'H_g', 'H_d', 'CH G-band', 'C3', 'C1', 'C4'])

delta_k = np.array([12.60, 14.52, 15.30, 18.28, 20.10, 20.42, 19.67 , 4.68, 4.95, 9.46, 3.93,
                    3.62e-1, 1.90e-1, 3.74e-1, 1.62e-1, 7.8e-3, 1.3e-2, 10.20, 12.08, 12.75, 1.89,
                    2.55, 2.86, 3.03, 2.88, 6.48, 7.49, 8.00                    
                    ])

event_names = ['CH₄ + e⁻ -> CH₄⁺ + 2e⁻',
 'CH₄ + e⁻ -> CH₃⁺ + H* +  2e⁻',
 'CH₄ + e⁻ -> CH₂⁺ + H₂ +  2e⁻',
 'CH₄ + e⁻ -> CH₃* + H⁺ +  2e⁻',
 'CH₄ + e⁻ -> CH⁺ + H₂ + H* +  2e⁻',
 'CH₄ + e⁻ -> CH₂* + H₂⁺ + 2e⁻',
 'CH₄ + e⁻ -> C⁺ + 2H₂ + 2e⁻',
 'CH₄ + e⁻ -> CH₃* + H* + e⁻',
 'CH₄ + e⁻ -> CH₂* + H₂ + e⁻',
 'CH₄ + e⁻ -> CH* + H₂ + H* + e⁻',
 'CH₄ + e⁻ -> CH₃* + H⁻',
 'mode v₁',
 'mode v₂',
 'mode v₃',
 'mode v₄',
 'J = 0 to J = 3',
 'J = 0 to J = 4',
 'Ly-α',
 'Ly-β',
 'Ly-γ',
 'H-α',
 'H-β',
 'H-γ',
 'H-δ',
 'CH G-band',
 'C III',
 'C I',
 'C IV']


# Shared constants
E_R = 1.36E-02  # Rydberg constant
sigma_0 = 1.00E-16
min_energy = 1.0

@jit(nopython=True)
def find_range(E, r):
    if E >= r[-1]:
        return len(r) -1
    for i in range(len(r) - 1):
        if r[i] <= E < r[i+1]:
            return i
    return -1

@jit(nopython=True)
def ME_cs(E, index, params, range, offset, slope):
    range_index = find_range(E, range)
    if range_index == -1:
        return 0
    if range_index == (len(range) - 1):
        slope = slope[index]
        offset = offset[index]
        return math.exp(slope * math.log(E) + offset)
    else:
        a0 = params[0][index][range_index]
        a1 = params[1][index][range_index]
        a2 = params[2][index][range_index]
        a3 = params[3][index][range_index]
        a4 = params[4][index][range_index]
        return a4 * (E**4) + a3 * (E**3) + a2 * (E**2) + a1 * (E) + a0
    
@jit(nopython=True)
def photon_cs(E, index, params, offset, slope):
    E_th = params[8][index]
    E_max = params[9][index]
    a1 = params[0][index]
    a2 = params[1][index]
    a3 = params[2][index]
    a4 = params[3][index]
    a5 = params[4][index]
    a6 = params[5][index]
    a7 = params[6][index]
    a8 = params[7][index]
    offset = offset[index]
    slope = slope[index]
    E_k = E/1000
    if E_k < E_th:
        return 0
    elif E_th <= E_k < E_max:
        E1 = E_k - E_th
        num_1 = sigma_0 * a1 * (E1 / E_R) ** a2
        den_1 = 1 + (E1 / a3) ** (a2 + a4) + (E1 / a5) ** (a2 + a6)
        term1 = num_1 / den_1
        if a7 > 0:
            num_2 = sigma_0 * a7 * a1 * (((E1 / a8) / E_R) ** a2)
            den_2 = 1 + ((E1 / a8)/a3) ** (a2 + a4) + (((E1/a8) / a5) ** (a2 +a6))
            term2 = num_2 / den_2
            return term1 + term2
        else:
            return term1
    else:
        return math.exp(slope * math.log(E) + offset)


@jit(nopython=True)
def ion_cs(E, index, params, offset, slope):
    E_th = params[6][index]
    E_max = params[7][index]
    E_k = E/1000
    a1 = params[0][index]
    a2 = params[1][index]
    a3 = params[2][index]
    a4 = params[3][index]
    a5 = params[4][index]
    a6 = params[5][index]
    slope = slope[index]
    offset = offset[index]
    E_physical_th = max(2 * min_energy + delta_k[index], E_th*1000)
    if E < E_physical_th:
        return 0.0
    else:
        if E_k < E_max:
            E1 = E_k - E_th
            num = sigma_0 * a1 * (E1 / E_R) ** a2
            den = 1 + (E1 / a3) ** (a2 + a4) + (E1 / a5) ** (a2 + a6)
            return num / den
        else:
            return math.exp(slope * math.log(E) + offset)


@jit(nopython=True)
def cross_section_calc(E):
    cross_sections = np.empty(28, dtype=np.float64)
    for i in prange(7):
        cross_sections[i] = ion_cs(E, i, params_ion, offset_ion, slope_ion)
    cross_sections[7] = ME_cs(E, 0, params_eie_1, eie_1_range, eie_1_offset, eie_1_slope)
    cross_sections[8] = ME_cs(E, 0, params_eie_2, eie_2_range,eie_2_offset, eie_2_slope)
    cross_sections[9] = ME_cs(E, 0, params_eie_3, eie_3_range,eie_3_offset, eie_3_slope)
    cross_sections[10] = ME_cs(E, 0, params_ea, ea_range, ea_offset, ea_slope)
    for n in prange(4):
        cross_sections[n+11] = ME_cs(E, n, params_nu ,Nu_ranges, offset_nu, slope_nu)
    for j in prange(2):
        cross_sections[15+j] = ME_cs(E,j, params_j, J_ranges, offset_j, slope_j)
    for p in prange(11):
        cross_sections[p+17] = photon_cs(E, p, params_pho, offset_pho, slope_pho)
    total =cross_sections.sum()
    return cross_sections/total

@jit(nopython=True)
def select_event(E):
    probs = cross_section_calc(E)
    limits = np.cumsum(probs)
    r = np.random.rand()
    for i, limit in enumerate(limits):
        if r < limit:
            return i
    return -1


    