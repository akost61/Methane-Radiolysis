import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from constants import event_names
from scripts.run_simulation import run_simulations, combine_data
import numpy as np

'''
Generate convergence graphs showing mean occurrence of events vs number of simulations run.
'''


def create_convergence_graphs(batch_data, initial_eV, total_simulations):
    x_values = np.arange(1, total_simulations + 1, 1)
    mean_data = batch_data[-1]
    with PdfPages(f"./{int(initial_eV/1000)}keV_mean_sim.pdf") as pdf:

        for i in range(len(event_names)):
            ax = plt.gca()
            mean_value = mean_data[i]
            plus_2 = mean_value * 1.02
            minus_2 = mean_value * 0.98
            
            plt.figure(figsize=(8, 6))
            plt.title(f'Initial Electron at {int(initial_eV/1000)}keV For {event_names[i]} Mean Occurance vs Simulations Ran')
            plt.xlabel('Simulation Runs')
            plt.ylabel('Mean Occurance')
            plt.axhline(y=plus_2, color='green', linestyle='--', label='+2%')
            plt.axhline(y=minus_2, color='red', linestyle='--', label='-2%')
            plt.plot(x_values, batch_data[:, i])
            ax.set_xticks(np.linspace(0, total_simulations, 6))
            plt.legend()
            pdf.savefig()
            plt.close()
