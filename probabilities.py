from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
from cross_section import cross_section_calc
import numpy as np




event_code = ['Ion_1', 'Ion_2', 'Ion_3', 'Ion_4', 'Ion_5', 'Ion_6','Ion_7','EIE_1', 'EIE_2', 'EIE_3', 'EA',
          'Nu1', 'Nu2', 'Nu3', 'Nu4', 'Jto3', 'Jto4', 'Ly_a', 'Ly_b', 'Ly_g', 'H_a', 'H_b',
          'H_g', 'H_d', 'CH G-band', 'C3', 'C1', 'C4']

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

event_dict = dict(zip(event_code, event_names))

energies = np.logspace(0,5,400)

cross_sections_total = []
for e in energies:
    cross_sections = cross_section_calc(e)
    total = np.sum(cross_sections)
    cross_sections = cross_sections / total
    cross_sections_total.append(cross_sections)



with PdfPages(f"/Users/emmajia/Desktop/MachineCode Radiolysis/Probabilities.pdf") as pdf:
    for i in range(len(event_names)):
        y_values = [cross_sections_total[E][i] for E in range(400)]
        plt.figure()
        plt.plot(energies, y_values)
        plt.xscale('log')
        plt.xlabel('eV')
        plt.ylabel('Probabilities')
        plt.title(f'Probability of {event_names[i]}')
        pdf.savefig()
        plt.close()
