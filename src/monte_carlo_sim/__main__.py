from monte_carlo_sim.file_writing.file_writing import create_results_folder, write_s_csv, write_s_readme, write_g_csv, write_g_readme
from monte_carlo_sim.simulation.run_simulation import run_simulations, run_generation_simulations
from monte_carlo_sim.simulation.constants import code_names, delta_k, reaction_produced

"""
Methane Radiolysis Simulation - Main Execution Entrypoint

This script provides the Command Line Interface (CLI) for the MRTS simulation.
It handles user input validation, orchestrates the simulation engine, and 
triggers the data export workflow.

Inputs:
    - Incident Energy (eV): The starting kinetic energy of the primary electron.
    - Cut-off Energy (eV): The threshold below which tracking ceases.
    - Total Simulations: Number of independent Monte Carlo trials.
    - Data Type: Choice between Standard (per-simulation) or Generational (event-tiered) output.
"""


def get_valid_input(prompt, max_value=None, type_func=float):
    while True:
        entry = input(f"Enter {prompt}: ")
        try:
            value = type_func(entry)
            if value > 0 and (max_value is None or value < max_value):
                return value
            print("Error: Value must be greater than 0", end="")
            if max_value is not None:
                print(f" and less than {max_value} (incident energy).")
            else:
                print()  
        except ValueError:
            print(f"Invalid input. Please enter a {type_func.__name__}.")

def get_gen_input():
     while True:
        choice = input("Select Data Type (Standard / Generational): ").strip().lower() or "standard"
        if choice in ['standard', 's']:
            return 1
        if choice in ['generational', 'g']:
            return 2
        print("Invalid choice. Please enter 'Standard' or 'Generational'.")

def main():
    print("Monte Carlo Methane Radiolysis Simulation\n")
    incident_energy = get_valid_input("incident energy in eV", type_func=float)
    cut_off = get_valid_input("cut-off energy in eV", max_value=incident_energy, type_func=float)
    total_simulations = get_valid_input("total simulations", type_func=int)
    request_type = get_gen_input()

    if request_type == 1:
        data, t_e, e_a = run_simulations(incident_energy, total_simulations, cut_off)
        results = create_results_folder(incident_energy, total_simulations, cut_off)
        write_s_csv(results, data, code_names, t_e, e_a)
        write_s_readme(results, incident_energy, cut_off, total_simulations, t_e, e_a)
    else:
        data, t_e, e_a = run_generation_simulations(incident_energy, total_simulations, cut_off)
        results = create_results_folder(incident_energy, total_simulations, cut_off, generational=True)
        write_g_csv(results, data, code_names)
        write_g_readme(results, incident_energy, cut_off, total_simulations, t_e, e_a)

    return



if __name__ == "__main__":
    main()
