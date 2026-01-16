# Methane Radiolysis Monte Carlo Simulation
## Overview

This repository contains a Python-based Monte Carlo simulation for methane radiolysis. The model tracks inelastic events, energy transfer, chemical species formation, and generational data of primary and daughter electrons until 1eV. Radiolysis of methane by electrons from a maximum of 100 keV, 28 different inelastic interaction events including ionization (with and without dissociation), electron impact excitation (neutral dissociation, molecular excitation and photon emission), and electron attachment dissociationThe code is optimized using numba and uses pandas for structured data analysis.

## Features

- Monte Carlo simulation of methane radiolysis events

- Energy deposition and transfer tracking (eV)

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

Two simulation types can be ran inside the scripts directory - 1. Recording events occurrances of the simulation; 2. Recording events occurrances based on electron generations of the simulation.

1. python3 main.py

2. python3 main_generational.py

## Runtime Notes

Typical runtime depends on:

- Incident electron energy (eV)

- Number of Monte Carlo simulations performed

Higher energies and larger numbers of histories result in longer runtimes.

## Output

The simulation produces excel sheets:

1. main.py

- Event-level interaction data

- Species yield

- Energy transfer metrics

2. main_generation.py

- Daughter electron generation data


Outputs are stored as structured data objects (pandas DataFrames) for analysis.
