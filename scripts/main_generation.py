import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from run_simulation import run_generation_simulations
from constants import event_names, reaction_produced, delta_k, reactives

'''
Generation-Tracking Script for Radiolysis Monte Carlo Simulations

This script tracks  events organized by electron generation (primary, secondary, 
tertiary, etc.) to analyze cascade propagation patterns in radiolysis simulations.

WORKFLOW:
1. Run Monte Carlo simulations with generation tracking at specified incident energy
2. Organize event data by generation number (primary = 0 ; secondary = 1; tertiary = 2; etc.)
3. Calculate total events per generation
4. Export generation-resolved data to Excel

DATA STRUCTURE:

run_generation_simulations():
  - Returns event counts organized by generation rather than by trajectory
  - Each row represents a generation level (0=primary, 1=secondary, etc.)
  - Each column represents one of the 28 collision event types
  - Shape: [10 × 28 events]

OUTPUT FORMAT:

Excel workbook structure:
  - Single sheet: 'generation_data'
  - Rows: Generation levels (indexed 0, 1, 2, ...)
  - Columns: All 28 event types from event_names
  - Additional 'Total' column: Sum of all events in each generation
  - File naming: {energy_in_keV}keV_{num_sims}_simulations_results.xlsx

USE CASES:
- Analyze depth and branching in electron-induced radiolysis
- Identify dominant events at each generation


KEY DIFFERENCES FROM MAIN SIMULATION SCRIPT:
- No energy conservation validation (tracks event counts only)
- No species production analysis
- Data organized by generation instead of aggregated totals
- Single-sheet output instead of three-sheet workbook

Default configuration: 100 keV incident electrons, 10,000 simulations.
'''

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