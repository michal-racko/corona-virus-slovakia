import logging

from tools.plotting import plot_pandemic
from simulations.exponential_meetings import run_simulation

VIRUS = 'SARS-CoV2'
MEAN_INTERACTIONS = 5  # mean number of random interactions among people
POPULATION_SIZE = 50000  # size of the simulated population
N_DAYS = 120
RESULTS_FILE = 'results/exponential.json'

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s: %(message)s'
)

if __name__ == '__main__':
    logging.info('Starting the simulation')

    results = run_simulation(
        virus_type=VIRUS,
        population_size=POPULATION_SIZE,
        n_days=N_DAYS,
        mean_interactions=MEAN_INTERACTIONS
    )

    logging.info('Plotting results')

    plot_pandemic(results)

    logging.info(f'Saving results to: {RESULTS_FILE}')

    results.to_json(RESULTS_FILE)
