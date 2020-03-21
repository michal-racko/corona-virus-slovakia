import logging
import argparse

from tools.config import Config
from tools.plotting.geographical import plot_final_mortality
from tools.plotting.timeseries import plot_pandemic
from examples.single_city import run_simulation

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s: %(message)s'
)

DEFAULT_CONFIG_PATH = 'config/monte_carlo.yml'

parser = argparse.ArgumentParser(
    description='A Monte Carlo simulation of virus spreading'
)

parser.add_argument(
    '-c',
    '--config-path',
    help='Path to the config file (.yml)',
    required=False
)

args = parser.parse_args()

config = Config()

try:
    config.read(args.config_path)

    logging.info(f'Using config: {args.config_path}')

except TypeError:
    config.read(DEFAULT_CONFIG_PATH)

    logging.info(f'Using config: {DEFAULT_CONFIG_PATH}')

if __name__ == '__main__':
    logging.info('Starting the simulation')

    results = run_simulation()

    logging.info('Plotting results')

    results_file = config.get('result_file')

    logging.info(f'Saving results to: {results_file}')

    results.to_json(results_file)

    plotting_file = config.get('plotting', 'pandemic_plot')

    plot_pandemic(results, plotting_file)

    # plot_final_mortality(results)
