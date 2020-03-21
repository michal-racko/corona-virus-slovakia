import logging

import numpy as np

from tools.simulation.virus import Virus
from tools.simulation.population import PopulationBase
from tools.data_structure import SimulationResult
from tools.simulation.population_centre import PopulationCentreBase

CITIES = [
    {
        'name': 'Bratislava',
        'longitude': 17.1,
        'latitude': 48.15,
        'inhabitants': 300000
    },
    {
        'name': 'Martin',
        'longitude': 18.90,
        'latitude': 49.10,
        'inhabitants': 54000
    },
    {
        'name': 'Zilina',
        'longitude': 18.75,
        'latitude': 49.20,
        'inhabitants': 80000
    },
    {
        'name': 'Trencin',
        'longitude': 18.00,
        'latitude': 48.90,
        'inhabitants': 55000
    },
    {
        'name': 'Kosice',
        'longitude': 21.25,
        'latitude': 48.72,
        'inhabitants': 240000
    },
]


def run_simulation(virus_type: str,
                   population_size: int,
                   n_days: int) -> SimulationResult:
    virus = Virus.from_string(virus_type)

    cities = [
        PopulationCentreBase(
            longitude=city['longitude'],
            latitude=city['latitude'],
            populations=[PopulationBase(int(city['inhabitants'] / 10), virus) for i in range(10)],
            virus=virus
        ) for city in CITIES
    ]

    cities[0].infect(50)

    for day_i in range(n_days):
        if day_i % 10 == 0:
            logging.info(f'day: {day_i}')

        for city_i in cities:
            for city_j in cities:
                distance = np.sqrt(
                    (city_i.longitude - city_j.longitude) ** 2 + (city_i.latitude - city_j.latitude) ** 2
                )

                if distance == 0:
                    continue

                travelling = len(city_i) / (distance ** 3)

                health_states = city_i.get_health_states(travelling)

                n_infected = health_states.astype(int).sum()

                city_j.infect(n_infected)

        for city in cities:
            city.next_day()

    days = []
    infected = []
    unaffected = []
    immune = []
    dead = []
    new_cases = []

    for city in cities:
        days += city.simulation_days
        infected += city.infected
        unaffected += city.unaffected
        immune += city.immune
        dead += city.dead
        new_cases += city.new_cases

    return SimulationResult(
        days=days,
        infected=infected,
        unaffected=unaffected,
        immune=immune,
        dead=dead,
        new_cases=new_cases,
    )
