import pandas as pd

from typing import List

from tools.simulation.virus import Virus
from tools.simulation.population import PopulationBase
from tools.simulation.population_centre import PopulationCentreBase


def city_factory(city_filepath: str, virus: Virus) -> List[PopulationCentreBase]:
    """
    Prepares cities based on the given csv file.
    Expected csv structure:
        <city:str>,<population: int>, <longitude: float>, <latitude: float>, <infected: int>
        (column order does not matter)

    :param city_filepath:       path to the csv file with cities

    :param virus:               desired virus type

    :return:                    instances of PopulationCentreBase ready for a simulation
    """
    city_df = pd.read_csv(city_filepath)

    cities = []

    for _, _data in city_df.iterrows():
        city_data = _data.to_dict()

        populations = [
            PopulationBase(int(city_data['population'] / 10), virus) for i in range(10)
        ]

        current_city = PopulationCentreBase(
            name=city_data['city'],
            longitude=city_data['longitude'],
            latitude=city_data['latitude'],
            populations=populations,
            virus=virus
        )

        if city_data['infected'] != 0:
            current_city.infect(city_data['infected'])

        cities.append(current_city)

    return cities
