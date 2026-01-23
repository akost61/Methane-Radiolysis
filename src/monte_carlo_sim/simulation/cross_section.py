import math
from numba import njit, prange
import numpy as np
from monte_carlo_sim.events.ionization import params_ion, slope_ion, offset_ion
from monte_carlo_sim.events.molecular_excitation import range_nu, params_nu, range_j, params_j, slope_nu, slope_j, offset_j, offset_nu
from monte_carlo_sim.events.eie import range_eie_1, range_eie_2, range_eie_3, params_eie_1,params_eie_2,params_eie_3, offset_eie_1, offset_eie_2, offset_eie_3, slope_eie_1, slope_eie_2, slope_eie_3
from monte_carlo_sim.events.electron_attachment import params_ea, range_ea, offset_ea, slope_ea
from monte_carlo_sim.events.photon_emission import params_pho, slope_pho, offset_pho
from monte_carlo_sim.simulation.constants import E_R, sigma_0, min_energy_ion, delta_k

"""
Cross Section Calculation and Event Selection for Methane Electron-Impact Processes

This module computes collision cross sections and performs Monte Carlo event selection
for all 28 inelastic processes in CH₄ electron-impact collisions.

CORE FUNCTIONS:

find_range(eV, r):
  - Determines which energy range bracket contains the given electron energy
  - Returns range index for piecewise polynomial evaluation
  - Returns -1 for energies above highest range (power law tail region)

ME_cs(eV, index, params, range, offset, slope):
  - Molecular Excitation and Electron Impact Excitation cross section calculator
  - Used for: vibrational modes, rotational transitions, EIE, and electron attachment


photon_cs(eV, index, params, offset, slope):
  - Photoionization cross section calculator
  - Uses empirical formula with up to 8 parameters (a1-a8)


ion_cs(eV, index, params, offset, slope):
  - Ionization cross section calculator  
  - Enforces physical threshold: E_physical_th = max(2*min_energy + delta_k, E_th*1000)


cross_section_calc(eV, manipulated=-1):
  - Computes all 28 normalized cross sections for given electron energy
  - Cross sections indexed as: [0-6] ionization, [7-9] EIE, [10] attachment, 
    [11-14] vibrational, [15-16] rotational, [17-27] photon emission
  - Optional manipulation: increases specified event cross section by 10% for sensitivity analysis
  - Returns probability distribution (normalized cross sections summing to 1)

select_event(eV, manipulated=-1):
  - Monte Carlo event selector using cross section probabilities
  - Generates random number and performs cumulative probability lookup
  - Returns event index (0-27) for the selected collision process
  - Returns -1 if no event selected (should not occur with proper normalization)

All functions JIT-compiled with Numba for performance. Energies in eV, cross sections in cm².
"""

@njit
def find_range(eV, r):
    if eV >= r[-1]:
        return len(r) -1
    for i in range(len(r) - 1):
        if r[i] <= eV < r[i+1]:
            return i
    return -1

@njit
def ME_cs(eV, index, params, range, offset, slope):
    range_index = find_range(eV, range)
    if range_index == -1:
        return 0
    if range_index == (len(range) - 1):
        slope = slope[index]
        offset = offset[index]
        return math.exp(slope * math.log(eV) + offset)
    else:
        a0 = params[0][index][range_index]
        a1 = params[1][index][range_index]
        a2 = params[2][index][range_index]
        a3 = params[3][index][range_index]
        a4 = params[4][index][range_index]
        return a4 * (eV**4) + a3 * (eV**3) + a2 * (eV**2) + a1 * (eV) + a0

@njit
def photon_cs(eV, index, params, offset, slope):
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
    eV_k = eV/1000
    if eV_k < E_th:
        return 0
    elif E_th <= eV_k < E_max:
        E1 = eV_k - E_th
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
        return math.exp(slope * math.log(eV) + offset)


@njit
def ion_cs(eV, index, params, offset, slope):
    E_th = params[6][index]
    E_max = params[7][index]
    eV_k = eV/1000
    a1 = params[0][index]
    a2 = params[1][index]
    a3 = params[2][index]
    a4 = params[3][index]
    a5 = params[4][index]
    a6 = params[5][index]
    slope = slope[index]
    offset = offset[index]
    E_physical_th = max(2 * min_energy_ion + delta_k[index], E_th*1000)
    if eV < E_physical_th:
        return 0.0
    else:
        if eV_k < E_max:
            E1 = eV_k - E_th
            num = sigma_0 * a1 * (E1 / E_R) ** a2
            den = 1 + (E1 / a3) ** (a2 + a4) + (E1 / a5) ** (a2 + a6)
            return num / den
        else:
            return math.exp(slope * math.log(eV) + offset)


@njit
def cross_section_calc(eV, manipulated=-1):
    cross_sections = np.empty(28, dtype=np.float64)
    for i in prange(7):
        cross_sections[i] = ion_cs(eV, i, params_ion, offset_ion, slope_ion)
    cross_sections[7] = ME_cs(eV, 0, params_eie_1, range_eie_1, offset_eie_1, slope_eie_1)
    cross_sections[8] = ME_cs(eV, 0, params_eie_2, range_eie_2, offset_eie_2, slope_eie_2)
    cross_sections[9] = ME_cs(eV, 0, params_eie_3, range_eie_3, offset_eie_3, slope_eie_3)
    cross_sections[10] = ME_cs(eV, 0, params_ea, range_ea, offset_ea, slope_ea)
    for n in prange(4):
        cross_sections[n+11] = ME_cs(eV, n, params_nu ,range_nu, offset_nu, slope_nu)
    for j in prange(2):
        cross_sections[15+j] = ME_cs(eV, j, params_j, range_j, offset_j, slope_j)
    for p in prange(11):
        cross_sections[p+17] = photon_cs(eV, p, params_pho, offset_pho, slope_pho)
    if manipulated != -1:
        cross_sections[manipulated] = cross_sections[manipulated] * 1.10
    total =cross_sections.sum()
    return cross_sections/total

@njit
def select_event(eV, manipulated=-1):
    probs = cross_section_calc(eV, manipulated)
    limits = np.cumsum(probs)
    r = np.random.rand()
    for i, limit in enumerate(limits):
        if r < limit:
            return i
    return -1


