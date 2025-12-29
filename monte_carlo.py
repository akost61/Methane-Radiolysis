import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from cross_section import event_names, delta_k
from run_simulation import get_batch_data
import numpy as np

def create_pdf(batch_data, initial_eV, total_simulations):
    x_values = np.arange(1000, total_simulations + 1, 1000)
    mean_data = batch_data[-1]
    with PdfPages(f"/Users/emmajia/Documents/GitHub/Methane-Radiolysis/MonteCarloSimulation/{int(initial_eV/1000)}keV_mean_sim.pdf") as pdf:

        for i in range(len(event_names)):
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
            plt.legend()
            pdf.savefig()
            plt.close()


data = get_batch_data(20000, 100000)
create_pdf(data, 20000, 100000)

data = get_batch_data(60000, 100000)
create_pdf(data, 60000, 100000)

data = get_batch_data(100000, 100000)
create_pdf(data, 100000, 100000)