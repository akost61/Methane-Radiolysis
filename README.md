# Methane Radiolysis Monte Carlo Simulation
## Overview

This repository contains a Python-based Monte Carlo simulation for methane radiolysis. The model tracks inelastic events, energy transfer, chemical species formation, and generational data of primary and daughter electrons until 1eV. Radiolysis of methane by electrons from a maximum of 100 keV, 28 different inelastic interaction events including ionization (with and without dissociation), electron impact excitation (neutral dissociation, molecular excitation and photon emission), and electron attachment dissociationThe code is optimized using numba and uses pandas for structured data analysis.

## Features

- Monte Carlo simulation of methane radiolysis events

- Energy (eV) transfer associated with each event

- Species creation and reaction accounting

- Daughter electron generation and species tracking

- Performance optimization with numba



## Requirements

- Python 3.13.2

- numpy

- pandas

- numba

- math

- tqdm

- openpyxl

- matplotlib.pyplot

## Usages

Two simulation types can be ran inside src/monte_carlo_sim  - 1. Recording events occurrances of the simulation; 2. Recording events occurrances based on electron generations of the simulation.

python3 -m monte_carlo_sim.main



## Structure

```
METHANERADIOLYSIS/
│
├── src/
│   └── monte_carlo_sim/
│       ├── events/
│       │   ├── __init__.py
│       │   ├── ele.py                   # Electron Impact Excitation event parameters
│       │   ├── electron_attachment.py   # Electron attachment event parameters
│       │   ├── ionization.py            # Ionization event parameters
│       │   ├── molecular_excitation.py  # Molecular excitation event parameters
│       │   └── photon_emission.py       # Photon emission event parameters
│       ├── plotting/
│       │   └── convergence.py
│       ├── simulation/
│       │   ├── __init__.py
│       │   ├── constants.py
│       │   ├── cross_section.py
│       │   └── run_simulation.py
│       ├── __init__.py
│       └── main.py
│
├── results/
│
├── LICENSE
└── README.md

```
## Runtime Notes

Typical runtime depends on:

- Incident electron energy (eV)

- Number of Monte Carlo simulations performed

Higher energies and larger numbers of histories result in longer runtimes.

## Output

The simulation produces excel sheets, a convergence graph and energy conservation (printed in Terminal) based on the number of simulations (set to 10,000) and energy (set to 100keV):

main.py

- Convergence mean values of each event (/MethaneRadiolysis/results/100keV_mean_sim.pdf)

- Energy conservation (Terminal)
  
- Event-level interaction data (/MethaneRadiolysis/results/100keV_10000_simulations_results.xlsx)

- Species yield (/MethaneRadiolysis/results/100keV_10000_simulations_results.xlsx)

- Energy transfer metrics (/MethaneRadiolysis/results/100keV_10000_simulations_results.xlsx)

- Daughter electron generation data (/MethaneRadiolysis/results/100keV_10000_simulations_results.xlsx)

