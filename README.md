# Methane Radiolysis Monte Carlo Simulation
## Overview

This repository contains a Python-based Monte Carlo simulation for methane radiolysis. The model tracks inelastic events, energy transfer, chemical species formation, and generational data of primary and daughter electrons until 1eV. Radiolysis of methane by electrons from a maximum of 100 keV, 28 different inelastic interaction events including ionization (with and without dissociation), electron impact excitation (neutral dissociation, molecular excitation and photon emission), and electron attachment dissociation. The code is optimized using numba and uses pandas for structured data analysis.

## Features

- Monte Carlo simulation of methane radiolysis events

- Energy (eV) transfer associated with each event

- Species creation and reaction accounting

- Daughter electron generation and species tracking

- Performance optimization with numba


## Requirements

- Python >=3.8,<3.13


## Installation

### Quick Install
```bash
pip install git+https://github.com/akost61/MethaneRadiolysis.git

```
### Troubleshooting Installation

If installation fails with an error about `llvmlite` or `numba`:

Install Miniconda(https://www.anaconda.com/download/success)

```bash
conda create -n methane python=3.11
```
```bash
conda activate methane
```
```bash
conda install numba
```
```bash
python -m pip install git+https://github.com/akost61/MethaneRadiolysis.git
```


## Usage
```bash
mrie
```

Follow the prompts to run your simulation.
- incident energy in eV : the initial energy of the electron
- cut-off energy in eV : the minimum energy of the electron that the program will no longer track
- total simulations: the total number of simulations the program will run for

## Output

The simulation produces a folder in your cwd. Inside will be README.txt, results.csv.

### Example Session
```
Monte Carlo Methane Radiolysis Simulation

Enter incident energy in eV: 100000
Enter cut-off energy in eV: 1
Enter total simulations: 10000
Select Data Type (Standard / Generational): s
```

## Output

Results are saved in a timestamped folder in your current directory:
```
results_2026-01-23_100.0keV_10000_1/
├── README.md                    # Simulation parameters and summary
└── simulation_results.csv       # Detailed output data
```

## Structure

```
METHANERADIOLYSIS/
│
├── src/
│   └── monte_carlo_sim/
│       ├── events/
│       │   ├── eie.py                   # Electron Impact Excitation event parameters
│       │   ├── electron_attachment.py   # Electron attachment event parameters
│       │   ├── ionization.py            # Ionization event parameters
│       │   ├── molecular_excitation.py  # Molecular excitation event parameters
│       │   └── photon_emission.py       # Photon emission event parameters
│       ├── file_writing/
│       │   ├── file_writing.py             # Write files for __main__.py
│       ├── simulation/
│       │   ├── constants.py             # Physical constants and event names
│       │   ├── cross_section.py         # Cross-section calculations for particle interactions
│       │   └── run_simulation.py        # Main simulation execution logic
│       └── __main__.py                      # Entry point for running simulations
│
├── results/                             # Sample results of incident=100_000ev; cut_off=1kev; simulations=10_000
│
├── Supplementary Information/                                
│
├── pyproject.toml                       # build for program
│
├── LICENSE                              # Project license information
└── README.md                            # Project documentation and usage guide
```

## Runtime Notes

Typical runtime depends on:

- Incident electron energy (eV)

- Number of Monte Carlo simulations performed

Higher energies and larger numbers of histories result in longer runtimes.




## Contact

- apkostiu@ualberta.ca
