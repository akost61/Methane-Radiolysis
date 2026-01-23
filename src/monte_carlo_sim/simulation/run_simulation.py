import numpy as np
from tqdm import tqdm
from numba import njit, prange
from monte_carlo_sim.simulation.cross_section import select_event
from monte_carlo_sim.simulation.constants import event_names, delta_k, min_energy_ion

"""
Monte Carlo Simulation Engine

This module performs Monte Carlo simulations of Electron-CH₄ inelastic interactions,
tracking secondary electron generation, energy degradation, and event statistics.

STACK OPERATIONS:
stack_push/stack_pop: LIFO stack management for tracking active electrons
stack_push_gen/stack_pop_gen: Extended stack tracking both energy and generation number

ENERGY PARTITION:
ion_event(eV, index): 
  - Handles ionization energy partitioning between incident and ejected electrons
  - Uses random sampling from physically-motivated distribution
  - Returns (eV_old, eV_new) for incident and secondary electrons

ion_gen_event(generation, energy, index):
  - Extended version tracking generation number for cascade analysis
  - Increments generation for secondary electron
  - Returns (eV_new, gen_new, eV_old)

SIMULATION FUNCTIONS:

run_sim(eV, manipulated=-1):
  - Single simulation starting from incident initial electron energy (eV)
  - Tracks all 28 event types until all electrons fall below min_energy threshold
  - Handles ionization (produces 2 electrons), excitation (produces 1 electron), and attachment (terminates electron)
  - Returns: event_count array, terminating_energy (sub-threshold energy below 1.0eV), electron_attachment_energy (energy absorbed with electron attachment)

run_batch_simulations(eV, storage, manipulated=-1):
  - Parallelized batch execution using Numba prange
  - Runs multiple independent cascade simulations
  - Returns: event counts, terminating energies, attachment energies for entire batch

run_simulations(eV, total_sims, manipulated=-1, chunk_size=500):
  - Interface on terminal with progress tracking
  - Processes simulations in chunks to manage memory
  - Aggregates results across all simulations
  - Optional manipulation parameter for sensitivity analysis (10% cross section increase)

GENERATION FUNCTIONS:

sim_generation(eV):
  - Tracks events by electron generation (primary, secondary, tertiary, etc.)
  - Records up to 10 generations in gen_data[generation][event_index]
  - Useful for understanding depth, energy transfer and events caused by generations

run_generation_simulations_batch(eV, total_sims):
  - Batch execution with generation tracking
  - Not parallelized to preserve generation statistics

run_generation_simulations(eV, total_sims, chunk_size=500):
  - Interface on terminal with progress tracking
  - Returns summed generation data across all simulations

UTILITIES:
combine_data(simulation_results):
  - Computes cumulative running average of simulation results
  - Useful for convergence analysis

Stack size set to 20 elements (sufficient for typical depths) as daughter electrons are handled first (dealing in lower energies).
All energies in eV, event counts are integers.
"""

@njit
def stack_push(stack, top, value):
    stack[top] = value
    return top + 1

@njit
def stack_pop(stack, top):
    if top==0:
        return 0, top
    top -= 1
    return stack[top], top


@njit
def ion_event(eV, index):
    u = np.random.rand()
    eV = eV - delta_k[index]
    x_max = (eV) / 2
    eV_new = (min_energy_ion * x_max) / (x_max - u * (x_max - min_energy_ion))
    eV_old = eV - eV_new
    return eV_old, eV_new


@njit
def run_sim(eV, min_energy=1, manipulated=-1):
    E_stack = np.empty(20, dtype=np.float64)
    event_count = np.zeros(28, dtype=np.float64)
    top = 0
    top = stack_push(E_stack, top, eV)
    terminating_energy = 0
    electron_attachment_energy = 0
    while top != 0:
        eV, top = stack_pop(E_stack, top)
        indx = select_event(eV, manipulated)
        event_count[indx] += 1
        if indx < 7:
            eV_old, eV_new = ion_event(eV, indx)

            if eV_old > min_energy:
                top = stack_push(E_stack, top, eV_old)
            else:
                terminating_energy += eV_old
            if eV_new > min_energy:
                top = stack_push(E_stack, top, eV_new)
            else:
                terminating_energy += eV_new

        else:
            if indx != 10:
                eV = eV - delta_k[indx]
                if eV > min_energy:
                    top = stack_push(E_stack, top, eV)
                else:
                    terminating_energy += eV
            else:
                electron_attachment_energy += eV
    return event_count, terminating_energy, electron_attachment_energy

@njit(parallel=True)
def run_batch_simulations(eV, storage, min_energy=1, manipulated=-1):
    EA_size = int(storage.shape[0])
    EA = np.empty(EA_size, dtype=np.float64)
    t_e_size = int(storage.shape[0])
    t_e = np.empty(t_e_size, dtype=np.float64)
    for i in prange(storage.shape[0]):
        storage[i], t_e[i], EA[i] = run_sim(eV, min_energy=min_energy, manipulated=manipulated)
    return storage, t_e, EA

def run_simulations(eV, total_sims, min_energy=1, manipulated=-1, chunk_size=500):
    result = np.zeros((int(total_sims), 28), dtype=np.int64)
    terminating_energy_total = np.zeros((int(total_sims)), dtype=np.float64)
    EA_total =  np.zeros((int(total_sims)), dtype=np.float64)
    print(f'Running {eV}eV electron simulations for {total_sims} iterations...')

    with tqdm(total=total_sims, unit="sim") as pbar:
        completed = 0
        while completed < total_sims:
            n = int(min(chunk_size, total_sims - completed))
            temp_storage = np.empty((n, 28), dtype=np.int64)
            chunk, terminating_energy, EA_chunk = run_batch_simulations(eV, temp_storage, min_energy=min_energy, manipulated=manipulated)
            result[completed:completed+n] = chunk
            terminating_energy_total[completed:completed+n] = terminating_energy
            EA_total[completed:completed+n] = EA_chunk
            completed += n
            pbar.update(n)

    return result, terminating_energy_total, EA_total


@njit
def stack_push_gen(gen_stack, energy_stack, top, energy, generation):
    gen_stack[top] = generation
    energy_stack[top] = energy
    return top + 1

@njit
def stack_pop_gen(gen_stack, energy_stack, top):
    if top == 0:
        return 0, 0.0, top
    top -= 1
    return gen_stack[top], energy_stack[top], top

@njit    
def ion_gen_event(generation, energy, index):
    gen_new = generation + 1
    eV = energy - delta_k[index]
    u = np.random.rand()
    x_max = eV / 2
    eV_new = (min_energy_ion * x_max) / (x_max - u * (x_max - min_energy_ion))
    eV_old = eV - eV_new

    return eV_new, gen_new, eV_old

@njit
def sim_generation(eV, min_energy=1):
    terminating_energy = 0.0
    electron_attachment_energy = 0.0
    
    gen_stack = np.empty(20, dtype=np.int32)
    energy_stack = np.empty(20, dtype=np.float64)
    top = 0
    
    generation = 0
    gen_data = np.zeros((10, 28), dtype=np.int64)
    
    top = stack_push_gen(gen_stack, energy_stack, top, eV, generation)
    
    while top != 0:
        generation, energy, top = stack_pop_gen(gen_stack, energy_stack, top)
        
        indx = select_event(energy)
        gen_data[generation][indx] += 1
        
        if indx < 7:
            eV_new, gen_new, eV_update = ion_gen_event(generation, energy, indx)

            if eV_update < min_energy:
                terminating_energy += eV_update
            else:
                top = stack_push_gen(gen_stack, energy_stack, top, eV_update, generation)

            if eV_new < min_energy:
                terminating_energy += eV_new
            else:
                top = stack_push_gen(gen_stack, energy_stack, top, eV_new, gen_new)
        else:
            if indx != 10:
                energy = energy - delta_k[indx]
                if energy < min_energy:
                    terminating_energy += energy
                else:
                    top = stack_push_gen(gen_stack, energy_stack, top, energy, generation)
            else:
                electron_attachment_energy += energy
                
    return gen_data, terminating_energy, electron_attachment_energy

@njit
def run_generation_simulations_batch(eV, total_sims, min_energy=1):
    storage = np.zeros((total_sims, 10, 28), dtype=np.int64)
    terminating_energy_total = 0.0
    electron_attachment_energy_total = 0.0
    for i in range(total_sims):
        simulation, terminating_energy, electron_attachment_energy = sim_generation(eV, min_energy=min_energy)
        storage[i] = simulation
        terminating_energy_total += terminating_energy
        electron_attachment_energy_total += electron_attachment_energy
    return storage.sum(axis=0), terminating_energy_total, electron_attachment_energy_total


def run_generation_simulations(eV, total_sims, min_energy=1, chunk_size=500):
    result = np.zeros((10, 28), dtype=np.int64)
    terminating_energy_total = 0.0
    electron_attachment_energy_total = 0.0
    print(f'Running {eV}eV electron simulations for {total_sims} iterations...')
    with tqdm(total=total_sims, unit="sim") as pbar:
        completed = 0
        while completed < total_sims:
            n = min(chunk_size, total_sims - completed)
            chunk, terminating_energy, electron_attachment_energy = run_generation_simulations_batch(eV, n, min_energy=min_energy)
            result += chunk
            terminating_energy_total += terminating_energy
            electron_attachment_energy_total += electron_attachment_energy
            completed += n
            pbar.update(n)

    return result, terminating_energy_total, electron_attachment_energy_total

def combine_data(simulation_results):
    cumsum = np.cumsum(simulation_results, axis=0)
    counts = np.arange(1, len(simulation_results) + 1).reshape(-1, 1)
    return cumsum / counts


