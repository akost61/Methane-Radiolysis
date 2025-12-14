import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from cross_section import event_names, delta_k
from run_simulation import run_simulations
import numpy as np

def create_pdf(simulation, N_sim, initial_ev):
    print(simulation[0])
    num_simulations = np.arange(1, simulation.shape[0] + 1) * 1000
    x_values = np.arange(1, N_sim + 1, 1000)
    mean_data = np.zeros((len(simulation),28), np.float64)
    print(num_simulations)
    for i in range(len(simulation)):
        mean_data[i] = (simulation[i]/num_simulations[i])
    

    if len(x_values) < len(simulation):
        x_values = np.append(x_values, N_sim)

    with PdfPages(f"/Users/emmajia/Desktop/MachineCode Radiolysis/MonteCarloSimulation/{int(initial_ev/1000)}keV_mean_sim.pdf") as pdf:

        for i in range(len(event_names)):
            mean_value = mean_data[-1][i]
            plus_2 = mean_value * 1.02
            minus_2 = mean_value * 0.98
            
            plt.figure(figsize=(8, 6))
            plt.title(f'Initial Electron at {int(initial_ev/1000)}keV For {event_names[i]} Mean Occurance vs Simulations Ran')
            plt.xlabel('Simulation runs')
            plt.ylabel('Mean occurance')
            plt.axhline(y=plus_2, color='green', linestyle='--', label='+2%')
            plt.axhline(y=minus_2, color='red', linestyle='--', label='-2%')
            plt.plot(x_values, mean_data[:, i]) 
            plt.legend()
            
            pdf.savefig()
            plt.close()

# n_sims = 1000
# ev = 1000
# batch_1kev, t_e = run_simulations(ev, n_sims)
# arr = []
# for batch in batch_1kev:
#     batch = batch * delta_k
#     arr.append(batch.sum() - batch[10])

# print("Energy \t terminating energy")
# for i in range(len(arr)):
#     print(f"{arr[i]} + {t_e[i]} = {arr[i] + t_e[i]}")

# n_sims = 100000
# ev = 20000

# batch_20kev = run_simulations(ev, n_sims)

# n_sims = 100000
# ev = 60000

# batch_60kev = run_simulations(ev, n_sims)


# np.save("batch20kev.npy", batch_20kev)

# np.save('batch_60kev.npy', batch_60kev)

# batch_20kev = np.load('batch2kev.npy')
batch_1kev = np.load('batch1kev.npy')
batch_100kev = np.load('batch100kev.npy')

create_pdf(batch_100kev, 10000, 100000)
create_pdf(batch_1kev, 1000000, 1000)