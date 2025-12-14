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
    x_max = (E - delta_k[index]) / 2
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
def run_gen_sim(E):
    E_stack = np.empty((20, 2), dtype=np.float64)

    event_count = np.zeros(28, dtype=np.float64)
    generation_event_count = np.zeros((100, 28), dtype=np.int64)

    top = 0
    top = stack_gen_push(E_stack, top, E, 0)

    terminating_energy = 0

    while top != 0:
        E, gen, top = stack_gen_pop(E_stack, top)

        indx = select_event(E)
        event_count[indx] += 1
        generation_event_count[gen, indx] += 1 

        if indx < 7:
            E_old, E_new = ion_event(E, indx)
            next_gen = gen + 1

            if E_old > min_energy:
                top = stack_gen_push(E_stack, top, E_old, next_gen)
            else:
                terminating_energy += E_old

            if E_new > min_energy:
                top = stack_gen_push(E_stack, top, E_new, next_gen)
            else:
                terminating_energy += E_new

        else:
            if indx != 10:
                E = E - delta_k[indx]
                if E > min_energy:
                    top = stack_gen_push(E_stack, top, E, gen)
                else:
                    terminating_energy += E
            else:
                terminating_energy += E

    return event_count, generation_event_count


@njit
def run_simulations(E, N_sims):
    next_free = 0
    batch_size =  (N_sims+999)//1000
    batch_simulations = np.zeros((batch_size, 28), dtype=np.float64)
    t_e_range = np.zeros(batch_size, dtype=np.float64)
    simulation_index = np.zeros(28, dtype=np.float64)
    terminating_energy = 0
    for r in range(N_sims):
        index_list, t_e = run_sim(E)
        terminating_energy += t_e
        if (r+1) % 1000==0:
            print(r+1, ' sims completed')
        for i in range(28):
            simulation_index[i] += index_list[i]
        if (r + 1) % 1000 == 0 or (r + 1) == N_sims:
            if next_free == 0:
                batch_simulations[next_free] = simulation_index.copy()
                t_e_range[next_free] = terminating_energy
                
            else:
                batch_simulations[next_free] = batch_simulations[next_free-1] + simulation_index
                t_e_range[next_free] = t_e_range[next_free-1] + terminating_energy

            next_free+=1
            simulation_index[:] = 0
            terminating_energy = 0

        
    return batch_simulations, t_e_range


# event_index,terminating_energy = run_simulations(100000, 10000)
# dict_event = dict(zip(events, event_index))

# for k in dict_event:
#      print(f'{k} = {dict_event[k]}')

# energy_total = event_index * delta_k
# print()
# dict_energy = dict(zip(events, energy_total))

# for k in dict_energy:
#      print(f'{k} = {dict_energy[k]}')

# print("Sum of All Events: ", energy_total.sum())
# print(f'terminating energy: {terminating_energy}')
# print("Total Energy: ", terminating_energy + energy_total.sum())

# print(f"Total Number of Events: {event_index.sum()}") 

