import logging
import argparse

from multiprocessing import Pool

from tools.config import Config
from tools.data_structure import GeographicalResult
from tools.plotting.timeseries import plot_pandemic
from tools.plotting.geographical import plot_infected

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s: %(message)s'
)

DEFAULT_CONFIG_PATH = 'config/business_as_usual.yml'

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

except TypeError:
    config.read(DEFAULT_CONFIG_PATH)


def plot_geo_data(day_i):
    day_string = f'{day_i}'.zfill(3)

    plot_infected(
        data,
        day_i=day_i,
        filepath=f'{geo_timeseries_dir}/infected/{day_string}.png'
    )


if __name__ == '__main__':
    logging.info('Plotting simulation results')

    data = GeographicalResult.read_json(
        config.get('result_file')
    )

    geo_timeseries_dir = config.get('geographical', 'timeseries_dir')

    pool = Pool(config.get('n_processes'))

    pool.map(plot_geo_data, [i for i in range(config.get('simulation_days'))])

    individual_city_dir = config.get('individual_cities', 'dir')

    for city_name in config.get('individual_cities', 'cities'):
        try:
            city_data = data.get_timeseries(city_name)

        except KeyError:
            logging.error(f'No data found for: {city_name}')
            continue

        plot_pandemic(
            city_data,
            f'{individual_city_dir}/{city_name}.png'
        )

    plot_pandemic(
        data.get_total_timeseries(),
        f'{individual_city_dir}/total.png'
    )
