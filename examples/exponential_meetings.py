import logging

import numpy as np

from tools.simulation.virus import Virus
from tools.simulation.population import PopulationBase
from tools.data_structure import TimeSeriesResult

"""
Simulates a single population whose members randomly meet each other each day.
Numbers of interactions are drawn from an exponential distribution.
"""


def _shuffle_people(current_population: PopulationBase,
                    mean_interactions) -> PopulationBase:
    """
    Alters health states of members of the given population by letting them meet randomly

    :returns:       Updated population
    """
    health_states = current_population.get_health_states()

    interaction_multiplicities = np.random.exponential(
        mean_interactions,
        len(health_states)
    ).astype(int)

    for interaction_i in range(interaction_multiplicities.max()):
        transmission_mask = (interaction_multiplicities > interaction_i)

        n_infected = health_states[transmission_mask].astype(int).sum()

        current_population.infect(n_infected)

    return current_population


def run_simulation(virus_type: str,
                   population_size: int,
                   n_days: int,
                   mean_interactions=25) -> TimeSeriesResult:
    virus = Virus.from_string(virus_type)
    population = PopulationBase(population_size, virus)

    population.infect(10)

    days = []

    infected_numbers = []
    unaffected_numbers = []
    immune_numbers = []
    dead_numbers = []
    new_cases = []

    for day_i in range(n_days):
        if day_i % 10 == 0:
            logging.info(f'day: {day_i}')

        population = _shuffle_people(
            population,
            mean_interactions
        )

        population.heal()

        population.kill()

        days.append(day_i)

        infected_numbers.append(population.get_n_infected())
        unaffected_numbers.append(population.get_n_unaffected())
        immune_numbers.append(population.get_n_immune())
        dead_numbers.append(population.get_n_dead())
        new_cases.append(population.get_n_new_cases())

        population.next_day()

    return TimeSeriesResult(
        simulation_days=days,
        infected=infected_numbers,
        unaffected=unaffected_numbers,
        immune=immune_numbers,
        dead=dead_numbers,
        new_cases=new_cases
    )
