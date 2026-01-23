import pandas as pd
from pathlib import Path
from datetime import date

"""
Methane Radiolysis Simulation - Data Export Module

This module handles the creation of output directories and the persistence of 
simulation results. It provides functionality to save simulation data into 
CSV formats and generate descriptive README metadata for both standard 
and generational Monte Carlo simulation runs.

Functions:
    - create_results_folder: Generates a timestamped directory for outputs.
    - write_s_csv/readme: Handles standard (per-simulation) data export.
    - write_g_csv/readme: Handles generational (binned by event tier) data export.
"""


generation_names = [
    "Primary",
    "Secondary",
    "Tertiary",
    "Quaternary",
    "Quinary",
    "Senary",
    "Septenary",
    "Octonary",
    "Nonary",
    "Denary",
]

def create_results_folder(initial_energy, simulations, cut_off, generational=False):
    today = date.today().strftime("%Y-%m-%d")
    initial_energy_kev = initial_energy / 1000
    folder_name = f"results_{today}_{initial_energy_kev}keV_{simulations}_{cut_off}"
    if generational:
        folder_name += "_generational"
    results_dir = Path.cwd() / folder_name
    results_dir.mkdir(exist_ok=True)
    return results_dir

def write_s_csv(results_dir, data, event_names, terminating_energy, electron_attachment):
    today = date.today().strftime("%Y-%m-%d")
    filename = f"results.csv"
    df = pd.DataFrame(data, columns=event_names)
    df["Terminating Energy"] = terminating_energy
    df["Electron Energy Captured"] = electron_attachment
    df["Simulation"] = "#" + (df.index + 1).astype(str) + " Simulation"

    df = df[
        ["Simulation"] + event_names +
        ["Terminating Energy", "Electron Energy Captured"]
    ]

    df.to_csv(results_dir / filename, index=False)
    return


def write_s_readme(results_dir,initial_energy, cut_off, simulations,
                   terminating_energy, electron_attachment):
    try:
        terminating_energy = terminating_energy.sum()
        electron_attachment = electron_attachment.sum()
        filename = f"README.txt"

        with open(results_dir / filename, "w", encoding="utf-8") as f:
            f.write("# Standard Simulation Results\n\n")
            f.write("## Simulation Parameters\n")
            f.write(f"- Initial Energy: {initial_energy} eV\n")
            f.write(f"- Cut-off Energy: {cut_off} eV\n")
            f.write(f"- Total Simulations: {simulations}\n\n")

            f.write("## Results Summary\n")
            f.write(f"- Total Input Energy: {initial_energy * simulations} eV\n")
            f.write(f"- Total Terminating Energy: {terminating_energy} eV\n")
            f.write(
                "- Total Electron Energy Captured "
                "(bond energy + excess kinetic energy): "
                f"{electron_attachment} eV\n"
            )

        return

    except PermissionError as e:
        raise PermissionError("Insufficient permissions to write README file") from e

    except Exception as e:
        raise RuntimeError("Failed to write README file") from e

def write_g_csv(results_dir, data, event_names):
    filename = f"results.csv"
    df = pd.DataFrame(data, columns=event_names)

    df["Generation"] = generation_names

    df = df[
        ["Generation"] + event_names
    ]

    df.to_csv(results_dir / filename, index=False)
    return

def write_g_readme(results_dir,initial_energy, cut_off, simulations,
                   terminating_energy, electron_attachment):

    try:
        filename = f"README.txt"

        with open(results_dir / filename, "w", encoding="utf-8") as f:
            f.write("# Generational Simulation Results\n\n")
            f.write("## Simulation Parameters\n")
            f.write(f"- Initial Energy: {initial_energy} eV\n")
            f.write(f"- Cut-off Energy: {cut_off} eV\n")
            f.write(f"- Total Simulations: {simulations}\n\n")

            f.write("## Results Summary\n")
            f.write(f"- Total Input Energy: {initial_energy * simulations} eV\n")
            f.write(f"- Total Terminating Energy: {terminating_energy} eV\n")
            f.write(
                "- Total Electron Energy Captured "
                "(bond energy + excess kinetic energy): "
                f"{electron_attachment} eV\n"
            )

        return

    except PermissionError as e:
        raise PermissionError("Insufficient permissions to write README file") from e

    except Exception as e:
        raise RuntimeError("Failed to write README file") from e

