from tools.simulation.virus import Virus
from tools.simulation.population import PopulationBase
from tools.data_structure import SimulationResult

"""
A basic example of how to use the PopulationBase class
"""


def run_simulation(virus_type: str,
                   population_size: int,
                   n_days: int):
    virus = Virus.from_string(virus_type)

    population = PopulationBase(population_size)

    population.infect(10)  # infect 10 randomly selected individuals

    days = []

    infected_numbers = []
    unaffected_numbers = []
    immune_numbers = []
    dead_numbers = []

    for day_i in range(n_days):
        health_states = population.get_healt_states(
            int(population_size * 0.5)
        )  # get health states of randomly selected half of the population

        n_infected = health_states.astype(int).sum()  # find the number of infected

        population.infect(
            n_infected
        )  # infect new people (each of the selected people meets another
        # random person from the population and infect them with 100% probability)

        population.heal(
            healing_days_mean=virus.illness_days_mean,
            healing_days_std=virus.illness_days_std
        )  # simulate recovery from the illness

        population.kill(
            mortality=virus.get_mortality(),
            healing_days=virus.illness_days_mean
        )  # let some of the ill people pass away

        # === log statistics ===

        days.append(day_i)

        infected_numbers.append(population.get_n_infected())
        unaffected_numbers.append(population.get_n_unaffected())
        immune_numbers.append(population.get_n_immune())
        dead_numbers.append(population.get_n_dead())

        # ======================

        population.next_day()  # see you all tomorrow

    return SimulationResult(
        days=days,
        infected=infected_numbers,
        unaffected=unaffected_numbers,
        immune=immune_numbers,
        dead=dead_numbers
    )
