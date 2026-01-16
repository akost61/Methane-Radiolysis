import numpy as np

'''
Parameters used for Ionization Inelastic Event cross section calculations.

IONIZATION REACTIONS:
ax_ion parameters correspond to the following ionization channels:
[0] : CH₄ + e⁻ -> CH₄⁺ + 2e⁻        
[1] : CH₄ + e⁻ -> CH₃⁺ + H* + 2e⁻   
[2] : CH₄ + e⁻ -> CH₂⁺ + H₂ + 2e⁻   
[3] : CH₄ + e⁻ -> CH₃* + H⁺ + 2e⁻   
[4] : CH₄ + e⁻ -> CH⁺ + H₂ + H* + 2e⁻  
[5] : CH₄ + e⁻ -> CH₂* + H₂⁺ + 2e⁻   
[6] : CH₄ + e⁻ -> C⁺ + 2H₂ + 2e⁻     

COEFFICIENT STRUCTURE:
ax_ion: Parameters for cross section calculations using empirical fit functions
  - x: parameter number (1-6 for the ionization fit formula)
  - Each ionization channel has its own set of parameters

ENERGY THRESHOLDS AND RANGES:
E_th_ion: Ionization threshold energies (in keV) for each reaction channel
  - Represents the minimum electron energy required to apply the cross section calculation.
E_max_ion: Upper energy bound for polynomial fit region (1.0 keV normalized scale for all channels)

CROSS SECTION CALCULATION:
Within the fit range (E_th to E_max):
  - Uses empirical formula with parameters a1-a6
  - Functional form captures ionization probability vs electron energy
  
POWER LAW TAIL:
For energies above E_max (high energy asymptotic behavior):
  - log10(σ) = slope_ion * log10(E) + offset_ion

'''


E_th_ion = np.array([
    1.26E-02,  
    1.26E-02,  
    1.26E-02,  
    2.22E-02,  
    0.0282,    
    2.23E-02,  
    2.11E-02  
])

E_max_ion = np.full(7, 1.0)  

a1_ion = np.array([9.66, 7.1323, 2.06E-01, 4.54E-01, 0.064, 6.30E-03, 4.97E-02])
a2_ion = np.array([2.32, 2.2621, 3.29E+00, 2.80E+00, 1.43, 5.20E+00, 3.51E+00])
a3_ion = np.array([5.60E-03, 5.6E-03, 1.50E-02, 2.00E-02, 0.0133, 1.74E-02, 2.24E-02])
a4_ion = np.array([-2.08E-01, -2.34E-01, -3.13E-01, 1.0E+00, -0.33, -7.78E-01, -8.22E-01])
a5_ion = np.array([1.87E-02, 1.92E-02, 2.87E-02, 7.90E-03, 0.0424, 2.72E-02, 3.90E-02])
a6_ion = np.array([9.10E-01, 9.08E-01, 1.01E+00, -5.22E-01, 1.181, 1.04E+00, 1.04E+00])


slope_ion = np.array([-0.74434, -0.7539, -0.91907, -0.9829, -1.1594, -1.0363, -1.0325])
offset_ion = np.array([-32.444, -32.592, -33.237, -33.6774, -33.686, -34.873, -32.365])

params_ion = np.array([a1_ion, a2_ion, a3_ion, a4_ion, a5_ion, a6_ion, E_th_ion, E_max_ion])


