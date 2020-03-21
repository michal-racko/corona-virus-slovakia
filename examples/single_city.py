import logging

from tools.simulation.virus import Virus
from tools.simulation.population import PopulationBase
from tools.data_structure import SimulationResult
from tools.simulation.population_centre import PopulationCentreBase


def run_simulation(virus_type: str,
                   population_size: int,
                   n_days: int) -> SimulationResult:
    virus = Virus.from_string(virus_type)

    population_centre = PopulationCentreBase(
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

    return SimulationResult(
        days=population_centre.simulation_days,
        infected=population_centre.infected,
        unaffected=population_centre.unaffected,
        immune=population_centre.immune,
        dead=population_centre.dead,
        new_cases=population_centre.new_cases
    )
