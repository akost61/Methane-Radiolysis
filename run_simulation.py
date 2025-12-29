import numpy as np
from numba import njit, prange
from cross_section import select_event, events, delta_k, event_names

min_energy = 1.0


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
def ion_event(E, index):
    u = np.random.rand()
    E = E - delta_k[index]
    x_max = (E) / 2
    E_new = (min_energy * x_max) / (x_max - u * (x_max - min_energy))
    E_old = E - E_new
    return E_old, E_new


@njit
def run_sim(E):
    E_stack = np.empty(20, dtype=np.float64)
    event_count = np.zeros(28, dtype=np.float64)
    top = 0
    top = stack_push(E_stack, top, E)
    terminating_energy = 0
    while top != 0:
        E, top = stack_pop(E_stack, top)
        indx = select_event(E)
        event_count[indx] += 1
        if indx < 7:
            E_old, E_new = ion_event(E, indx)

            if E_old > min_energy:
                top = stack_push(E_stack, top, E_old)
            else:
                terminating_energy += E_old
            if E_new > min_energy:
                top = stack_push(E_stack, top, E_new)
            else:
                terminating_energy +=E_new

        else:
            if indx != 10:
                E = E - delta_k[indx]
                if E > min_energy:
                    top = stack_push(E_stack, top, E)
                else:
                    terminating_energy += E
            else:
                terminating_energy += E
    return event_count, terminating_energy

@njit
def stack_gen_push(E_stack, top, E, gen):
    E_stack[top, 0] = E
    E_stack[top, 1] = gen
    return top + 1

@njit
def stack_gen_pop(E_stack, top):
    top -= 1
    E  = E_stack[top, 0]
    gen = E_stack[top, 1]
    return E, gen, top


@njit
def run_simulations(E, storage):
    t_e_size = int(storage.shape[0])
    t_e = np.empty(t_e_size, dtype=np.float64)
    for i in prange(storage.shape[0]):
        if (i+1) % 1000 == 0:
            print(i+1, ' sims completed')
        storage[i], t_e[i] = run_sim(E)
    return storage, t_e

@njit
def get_batch_data(eV, total_sims):
    storage = np.empty((total_sims, 28), dtype=np.float64)
    event_index, terminating_energy = run_simulations(eV, storage)
    number_of_batches = int(total_sims / 1000)
    x_values = np.arange(1000, 1001 * number_of_batches, 1000)
    batch_storage = np.empty((number_of_batches, 28), dtype=np.float64)
    for i in prange(number_of_batches):
        start = i * 1000
        end = (i+1) * 1000
        batch_storage[i] = event_index[start:end].sum(axis=0)
    for i in prange(batch_storage.shape[0]):
        if i !=0:
            batch_storage[i] += batch_storage[i-1]
    for i in prange(batch_storage.shape[0]):
        batch_storage[i] = batch_storage[i] / x_values[i]
        
    return batch_storage
