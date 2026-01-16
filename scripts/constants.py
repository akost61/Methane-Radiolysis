import numpy as np

'''
Global Constants and Event Definitions for Methane (CH₄) Electron-Impact Cross Sections

PHYSICAL CONSTANTS:
E_R: Rydberg constant
sigma_0: Reference cross section 
min_energy: Minimum electron energy threshold (1.0 eV)

EVENT CLASSIFICATION (28 total processes):
Ionization [0-6]: Produce positive ions and secondary electrons
Radical formation [7-9]: Neutral excited fragments (electron impact excitation)
Attachment [10]: Negative hydrogen ion formation
Molecular excitation [11-16]: Vibrational (v₁-v₄) and rotational (J=0→3, J=0→4) modes
Photon emission [17-27]: Lyman series, Balmer series, and carbon emission lines

REACTION PRODUCTS:
reaction_produced: Maps reactive events to stoichiometric products {species: count}
  - Notation: ⁺ (cation), ⁻ (anion), * (excited/radical state)

ENERGY LOSS:
delta_k: Energy transferred per collision event (in eV) for each process
  - Includes ionization potentials, excitation energies, and photon emission energies

'''


E_R = 1.36E-02  
sigma_0 = 1.00E-16
min_energy = 1.0



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


reaction_produced = {
    'CH₄ + e⁻ -> CH₄⁺ + 2e⁻': {'CH₄⁺':1},
    'CH₄ + e⁻ -> CH₃⁺ + H* +  2e⁻': {'CH₃⁺':1, 'H*':1},
    'CH₄ + e⁻ -> CH₂⁺ + H₂ +  2e⁻': {'CH₂⁺':1, 'H₂':1},
    'CH₄ + e⁻ -> CH₃* + H⁺ +  2e⁻': {'CH₃*':1, 'H⁺':1},
    'CH₄ + e⁻ -> CH⁺ + H₂ + H* +  2e⁻': {'CH⁺':1, 'H₂':1, 'H*':1},
    'CH₄ + e⁻ -> CH₂* + H₂⁺ + 2e⁻': {'CH₂*':1, 'H₂⁺':1},
    'CH₄ + e⁻ -> C⁺ + 2H₂ + 2e⁻': {'C⁺':1, 'H₂':2},
    'CH₄ + e⁻ -> CH₃* + H* + e⁻': {'CH₃*':1, 'H*':1},
    'CH₄ + e⁻ -> CH₂* + H₂ + e⁻': {'CH₂*':1, 'H₂':1},
    'CH₄ + e⁻ -> CH* + H₂ + H* + e⁻': {'CH*':1, 'H₂':1, 'H*':1},
    'CH₄ + e⁻ -> CH₃* + H⁻': {'CH₃*':1, 'H⁻':1}
}

ionization = event_names[0:7]
radicals = event_names[7:10]
attachment = event_names[10]
reactives = ionization + radicals + [attachment]
molecular_excitation = event_names[11:17]
photon_producing = event_names[17:28]


delta_k = np.array([12.60, 14.52, 15.30, 18.28, 20.10, 20.42, 19.67, 4.68, 4.95, 9.46, 3.93,
                    3.62e-1, 1.90e-1, 3.74e-1, 1.62e-1, 7.8e-3, 1.3e-2, 10.20, 12.08, 12.75, 1.89,
                    2.55, 2.86, 3.03, 2.88, 6.48, 7.49, 8.00                    
                    ])

