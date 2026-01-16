import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from run_simulation import run_generation_simulations
from constants import event_names, reaction_produced, delta_k, reactives


def main():
    incident_energy = 100000 #eV
    total_simulations = 10000
    data_100kev, terminating_energy, electron_attachment_energy = run_generation_simulations(incident_energy, total_simulations)
    df_events = pd.DataFrame(data_100kev, columns=event_names)
    df_events['Total'] = df_events.sum(axis=1)
    df_events.to_excel(f"./{int(incident_energy/1000)}keV_{total_simulations}_simulations_results.xlsx", sheet_name='generation_data', index=True)

    return 0

if __name__ == "__main__":
    main()