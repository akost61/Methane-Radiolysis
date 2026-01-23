import pytest
import numpy as np
from scipy import stats


from src.monte_carlo_sim.simulation.run_simulation import run_simulations
from src.monte_carlo_sim.simulation.constants import delta_k

class TestBasicEventCounts:


    def test_all_counts_non_negative(self):
        event_counts, _, _ = run_simulations(eV=100_000, total_sims=1, manipulated=-1)

        assert np.all(event_counts >= 0), (
        f"Negative event counts found: {event_counts[event_counts < 0]}"
    )


    def test_all_counts_are_int(self):
        event_counts, _, _ = run_simulations(eV=100_000, total_sims=1, manipulated=-1)
        event_counts = event_counts[0]
        for i, count in enumerate(event_counts):
            assert float(count).is_integer(), f"Event count at index {i} has a decimal part: {count}"

class TestReproducibility:


    def test_statistical_reproducibility(self):
        total_sims = 5000 
        eV = 100_000
        manipulated = -1

        counts1, _, _ = run_simulations(eV=eV, total_sims=total_sims, manipulated=manipulated)

        counts2, _, _ = run_simulations(eV=eV, total_sims=total_sims, manipulated=manipulated)


        assert np.isclose(counts1.sum(), counts2.sum(), rtol=0.01), \
            "Total number of events differs more than 1%"
        counts1 = counts1.sum(axis=0)
        counts2 = counts2.sum(axis=0)
        freq1 = counts1 / counts1.sum()
        freq2 = counts2 / counts2.sum()

        np.testing.assert_allclose(freq1, freq2, rtol=0.05, 
            err_msg="Event distributions differ beyond statistical tolerance")

        print("Statistical reproducibility test passed.")

        
class TestEnergyScaling:

    def test_energy_transfer(self):
        energy = 100_000
        event_counts, elim_energy, EA_energy = run_simulations(eV=energy, total_sims=1, manipulated=-1)
        event_counts = event_counts.sum(axis=0)
        elim_energy = elim_energy.sum()
        EA_energy=EA_energy.sum()
        total_transferred = np.sum(event_counts * delta_k) + EA_energy - (event_counts[10] * delta_k[10]) + elim_energy
        np.testing.assert_allclose(total_transferred, 100_000, rtol=0.0005, err_msg=f"Energy transfer should be 100_000 but got {total_transferred}")


    def test_energy_scaling_is_monotonic(self):
        energies = [10000, 50000, 100000]
        total_events = []
        for energy in energies:
            event_counts, _, _ = run_simulations(eV=energy, total_sims=1, manipulated=-1)
            total_events.append(np.sum(event_counts))

        for i in range(len(total_events) - 1):
            assert total_events[i] < total_events[i + 1], f"Total events should increase with energy: {total_events}"

class TestEventDistribution:

    def test_event_counts_variance(self):
        n_runs = 50
        all_event_counts = []

        for _ in range(n_runs):
            event_counts, _, _ = run_simulations(eV=100_000, total_sims=1, manipulated=-1)
            all_event_counts.append(event_counts)

        all_event_counts = np.array(all_event_counts)
        assert all_event_counts.shape[0] == n_runs, "Number of runs does not match"
        assert all_event_counts.shape[1] > 0, "Event counts are empty"
        assert np.var(all_event_counts, axis=0).sum() > 0, "Event counts do not have variance"

        assert np.var(all_event_counts, axis=0).sum() < 1e6, "Event counts have too much variance"


class TestConvergence:

    def test_mean_stabilized_with_more_runs(self):
        n_runs = 2000

        event_counts, _, _ = run_simulations(eV=100_000, total_sims=2000, manipulated=-1)

        running_mean = np.cumsum(event_counts, axis=0) / np.arange(1, n_runs + 1)[:, np.newaxis]
        changes = np.abs(np.diff(running_mean, axis=0))

        avg_change_first = np.mean(changes[:n_runs//2, :])
        avg_change_second = np.mean(changes[n_runs//2:, :])

        assert avg_change_second < avg_change_first, \
            f"Mean did not stabilize: {avg_change_second} is not less than {avg_change_first}"
        
    def test_standard_error_convergence(self):
        n_tiers = [100, 400, 1600] 
        results = []

        for n in n_tiers:
            counts, _, _ = run_simulations(eV=100_000, total_sims=n)
            counts = counts.sum(axis=0)
            stds = np.std(counts, axis=0, ddof=1)
            se_per_event = stds / np.sqrt(n)
            
            results.append(np.mean(se_per_event))

        for i in range(len(results) - 1):
            assert results[i+1] < results[i], \
                f"Precision failed to improve at N={n_tiers[i+1]}"
