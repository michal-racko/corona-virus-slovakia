from tools.population import Population

import matplotlib.pyplot as pl

if __name__ == '__main__':
    population = Population(10000)

    infected_numbers = []
    dead_numbers = []

    population.infect(15)

    for i in range(30):
        n_infected = population.get_members(8000).astype(int).sum()

        population.infect(n_infected)
        population.heal(14, 5)
        population.kill(0.05, 14)
        population.next_day()

        infected_numbers.append(
            population.get_n_infected()
        )

        dead_numbers.append(
            population.get_n_dead()
        )

    pl.plot(dead_numbers)
    pl.show()
