import logging

import numpy as np

from tools.virus import Virus
from tools.population import PopulationBase
from tools.simulation_result import SimulationResult

"""
Simulates a single population whose members randomly meet each other each day.
Numbers of interactions are drawn from an exponential distribution.
"""


def _shuffle_people(current_population: PopulationBase,
                    virus: Virus,
                    population_size: int,
                    mean_interactions) -> PopulationBase:
    """
    Alters health states of members of the given population by letting them meet randomly

    :returns:       Updated population
    """
    health_states = current_population.get_healt_states(population_size)

    interaction_multiplicities = np.random.exponential(
        mean_interactions,
        len(health_states)
    ).astype(int)

    for interaction_i in range(interaction_multiplicities.max()):
        transmission_mask = (interaction_multiplicities > interaction_i) * \
                            (np.random.random(len(health_states)) <= virus.get_transmission_probability())

        n_infected = health_states[transmission_mask].astype(int).sum()

        current_population.infect(n_infected)

    return current_population


def run_simulation(virus_type: str,
                   population_size: int,
                   n_days: int,
                   mean_interactions=25) -> SimulationResult:
    virus = Virus.from_string(virus_type)
    population = PopulationBase(population_size)

    population.infect(10)

    days = []

    infected_numbers = []
    unaffected_numbers = []
    immune_numbers = []
    dead_numbers = []

    for day_i in range(n_days):
        if day_i % 10 == 0:
            logging.info(f'day: {day_i}')

        population = _shuffle_people(
            population,
            virus,
            population_size,
            mean_interactions
        )

        population.heal(
            virus.illness_days_mean,
            virus.illness_days_std
        )

        population.kill(
            virus.get_mortality(),
            virus.illness_days_mean
        )

        days.append(day_i)

        infected_numbers.append(population.get_n_infected())
        unaffected_numbers.append(population.get_n_unaffected())
        immune_numbers.append(population.get_n_immune())
        dead_numbers.append(population.get_n_dead())

        population.next_day()

    return SimulationResult(
        days=days,
        infected=infected_numbers,
        unaffected=unaffected_numbers,
        immune=immune_numbers,
        dead=dead_numbers
    )
