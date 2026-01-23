import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from monte_carlo_sim.simulation.constants import event_names
from monte_carlo_sim.simulation.run_simulation import run_simulations, combine_data
import numpy as np

"""
Generate convergence graphs showing mean occurrence of events vs number of simulations run.
"""

def cumulative_combine(batch_data, combine_funct=np.add):
    cum_sums = np.empty((len(batch_data), *batch_data[0].shape), dtype=batch_data[0].dtype)
    cum_sums[0] = batch_data[0]
    
    for i, sim in enumerate(batch_data[1:], start=1):
        cum_sums[i] = combine_funct(cum_sums[i-1], sim)
    
    counts = np.arange(1, len(batch_data) + 1)[:, np.newaxis]
    cum_means = cum_sums / counts
    print(cum_means)    
    return cum_means



def create_convergence_graphs(batch_data, initial_eV, total_simulations):
    batch_data = cumulative_combine(batch_data)
    mean_data = batch_data[-1]
    x_values = np.arange(1, total_simulations + 1, 1)
    with PdfPages(f"./results/{int(initial_eV/1000)}keV_mean_sim.pdf") as pdf:

        for i in range(len(event_names)):
            plt.figure(figsize=(8, 6))
            ax = plt.gca()
            mean_value = mean_data[i]
            plus_2 = mean_value * 1.02
            minus_2 = mean_value * 0.98            
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
