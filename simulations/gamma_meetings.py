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
                    virus: Virus) -> PopulationBase:
    """
    Alters health states of members of the given population by letting them meet randomly

    :returns:       Updated population
    """
    health_states = current_population.get_healt_states()

    interaction_multiplicities = current_population.get_stochastic_interaction_multiplicities()

    for interaction_i in range(interaction_multiplicities.max()):
        transmission_mask = (interaction_multiplicities > interaction_i) * \
                            (np.random.random(len(health_states)) <= virus.get_transmission_probability())

        n_infected = health_states[transmission_mask].astype(int).sum()

        current_population.infect(n_infected)

    interaction_multiplicities = current_population.get_periodic_interaction_multiplicities()

    health_states = current_population.get_healt_states(random_seed=42)

    for interaction_i in range(interaction_multiplicities.max()):
        transmission_mask = (interaction_multiplicities > interaction_i) * \
                            (np.random.random(len(health_states)) <= virus.get_transmission_probability())

        n_infected = health_states[transmission_mask].astype(int).sum()

        current_population.infect(n_infected)

    return current_population


def run_simulation(virus_type: str,
                   population_size: int,
                   n_days: int) -> SimulationResult:
    virus = Virus.from_string(virus_type)
    population = PopulationBase(population_size)

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
            virus
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
        new_cases.append(population.get_n_new_cases())

        population.next_day()

    return SimulationResult(
        days=days,
        infected=infected_numbers,
        unaffected=unaffected_numbers,
        immune=immune_numbers,
        dead=dead_numbers,
        new_cases=new_cases
    )
