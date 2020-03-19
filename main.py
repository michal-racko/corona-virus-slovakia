import logging

from tools.plotting import plot_pandemic
from simulations.single_city import run_simulation


VIRUS = 'SARS-CoV2'
POPULATION_SIZE = 300000  # size of the simulated population
N_DAYS = 120
RESULTS_FILE = 'results/exponential.json'
PLOTTING_FILE = 'plots/exponential.png'

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s: %(message)s'
)

if __name__ == '__main__':
    logging.info('Starting the simulation')

    results = run_simulation(
        virus_type=VIRUS,
        population_size=POPULATION_SIZE,
        n_days=N_DAYS
    )

    logging.info('Plotting results')

    plot_pandemic(results, filepath=PLOTTING_FILE)

    logging.info(f'Saving results to: {RESULTS_FILE}')

    results.to_json(RESULTS_FILE)
