from tools.plotting import plot_pandemic
from simulations.exponential_meetings import run_simulation

VIRUS = 'SARS-CoV2'
MEAN_INTERACTIONS = 10  # mean number of random interactions among people
POPULATION_SIZE = 10000  # size of the simulated population
N_DAYS = 60
RESULTS_FILE = 'results/exponential.json'

if __name__ == '__main__':
    results = run_simulation(
        virus_type=VIRUS,
        population_size=POPULATION_SIZE,
        n_days=N_DAYS,
        mean_interactions=MEAN_INTERACTIONS
    )

    plot_pandemic(results)

    results.to_json(RESULTS_FILE)
