import logging

from tools.config import Config
from tools.simulation.virus import Virus
from tools.simulation.population import PopulationBase
from tools.data_structure import TimeSeriesResult
from tools.simulation.population_centre import PopulationCentreBase


def run_simulation() -> TimeSeriesResult:
    config = Config()

    population_size = 450000
    virus = Virus.from_string(config.get('virus', 'name'))
    n_days = config.get('simulation_days')

    population_centre = PopulationCentreBase(
        name='Mock city',
        longitude=17.1,
        latitude=48.15,
        populations=[PopulationBase(int(population_size / 10), virus) for i in range(10)],
        virus=virus
    )

    population_centre.infect(50)

    for day_i in range(n_days):
        if day_i % 10 == 0:
            logging.info(f'day: {day_i}')

        population_centre.next_day()

    return TimeSeriesResult(
        simulation_days=population_centre.simulation_days,
        infected=population_centre.infected,
        unaffected=population_centre.unaffected,
        immune=population_centre.immune,
        dead=population_centre.dead,
        new_cases=population_centre.new_cases
    )
