import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from run_simulation import run_simulations, run_generation_simulations
from convergence import create_convergence_graphs
from constants import event_names, reaction_produced, delta_k

'''
Main Execution Script for Radiolysis Monte Carlo Simulations

This script orchestrates the complete simulation workflow: running Monte Carlo simulation,
processing results, generating convergence diagnostics, and exporting data to Excel.

WORKFLOW:
1. Run Monte Carlo simulations at specified incident energy
2. Generate convergence analysis plots
3. Process results into three data views: events, energy, and species
4. Perform energy conservation validation
5. Export results to Excel workbook

DATA PROCESSING:

create_dataframes(data, electron_attachment_energy):
  - Converts raw simulation data into three analysis perspectives:
    * df_events: Event frequency counts for all 28 event channels
    * df_energy: Energy transfer per event type (Count × delta_k), except electron attachment is set to the total electron attachment energy
    * df_species: Species production counts aggregated across all reactions
 - Note - electron attachment is different from other events because it includes both changes in the potential energy of the molecules, but also the removal of the electron 
        and any excess kinetic energy from the simulation

main():
  - Configurable parameters: incident_energy (eV), total_simulations
  - Runs simulation suite and generates diagnostic plots
  - Validates energy conservation: input energy = transferred energy + terminating energy (which is associated with each electron's remaining energy after it crosses the 1 eV threshold)
  - Exports three-sheet Excel workbook:
    * Event_data: Raw counts the 28 different events/channels
    * Energy_data: Energy distribution for each of 28 different events/channels
    * Species_data: Net species production with stoichiometry
  - File naming: {energy_in_keV}keV_{num_sims}_simulations_results.xlsx

ENERGY CONSERVATION CHECK:
Total input = incident_energy × total_simulations
Total output = Σ(event_count × delta_k) + terminating_energy + attachment_energy
These should match within numerical precision.

Default configuration: 100 keV incident electrons, 10,000  simulations.
'''


def create_dataframes(data, electron_attachment_energy):
    combined_data = data.sum(axis=0)
    df_events = pd.DataFrame({'Event': event_names, 'Count': combined_data})


    df_energy = df_events.copy()
    df_energy['delta k'] = delta_k
    df_energy['Energy Transferred'] = df_energy['Count'] * df_energy['delta k']

    ea_mask = df_energy['Event'] == 'CH₄ + e⁻ -> CH₃* + H⁻'
    df_energy.loc[ea_mask, 'Energy Transferred'] = electron_attachment_energy.sum()
    df_energy = df_energy[['Event','Energy Transferred']]
    df_species = df_events.copy()
    df_species['Product'] = df_species['Event'].map(reaction_produced)
    species_rows = []
    for i, event in enumerate(event_names):
        if event in reaction_produced:
            count = combined_data[i]
            for species, stoich in reaction_produced[event].items():
                species_rows.append({
                    'Species': species,
                    'Count': count * stoich
                })
    
    df_species = pd.DataFrame(species_rows)
    
    df_species = df_species.groupby('Species')['Count'].sum().to_frame()

    return df_events, df_energy, df_species


def main():
    incident_energy = 100000  # in eV
    total_simulations = 10000
    data_100kev, terminating_energy, electron_attachment_energy =run_simulations(incident_energy, total_simulations)
    create_convergence_graphs(data_100kev, incident_energy, total_simulations)
    df_events, df_energy, df_species = create_dataframes(data_100kev, electron_attachment_energy)
    total_energy = terminating_energy.sum() + df_energy['Energy Transferred'].sum()
    print("Total energy input:", incident_energy * total_simulations)
    print("Total energy used :", df_energy['Energy Transferred'].sum())
    print("Total energy lost at termination :", terminating_energy.sum())
    print("Total energy used in model: ", total_energy)
    file_name = f"./{int(incident_energy/100)}keV_{total_simulations}_simulations_results.xlsx"
    with pd.ExcelWriter(file_name, engine='openpyxl') as writer:
        df_events.to_excel(writer, sheet_name='Event_data', index=False)
        df_energy.to_excel(writer, sheet_name='Energy_data', index=False)
        df_species.to_excel(writer, sheet_name='Species_data', index=True)

    
    incident_energy = 100000 #eV
    total_simulations = 10000
    data_100kev, terminating_energy, electron_attachment_energy = run_generation_simulations(incident_energy, total_simulations)
    df_events = pd.DataFrame(data_100kev, columns=event_names)
    df_events['Total'] = df_events.sum(axis=1)
    df_events.to_excel(f"./{int(incident_energy/1000)}keV_{total_simulations}_simulations_results.xlsx", sheet_name='generation_data', index=True)



    return 0



if __name__ == "__main__":
    main()