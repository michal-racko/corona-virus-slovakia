import json

import numpy as np

from tools.general import ensure_dir


class TimeSeriesResult:
    """
    Encapsulates results of a simulation as timeseries (per parameter)
    """

    def __init__(self,
                 simulation_days: list,
                 infected: list,
                 unaffected: list,
                 new_cases: list,
                 immune: list,
                 dead: list,
                 hospitalized: list):
        assert len(infected) == len(simulation_days)
        assert len(unaffected) == len(simulation_days)
        assert len(new_cases) == len(simulation_days)
        assert len(immune) == len(simulation_days)
        assert len(dead) == len(simulation_days)
        assert len(hospitalized) == len(simulation_days)

        self.days = simulation_days
        self.infected = infected
        self.unaffected = unaffected
        self.new_cases = new_cases
        self.immune = immune
        self.dead = dead
        self.hospitalized = hospitalized

    def to_json(self, filepath: str):
        """
        Saves results as a .json file to the given path
        """
        ensure_dir('/'.join(filepath.split('/')[:-1]))

        with open(filepath, 'w') as f:
            json.dump({
                'days': self.days,
                'infected': self.infected,
                'unaffected': self.unaffected,
                'new_cases': self.new_cases,
                'immune': self.immune,
                'dead': self.dead,
                'hospitalized': self.hospitalized
            }, f)


class GeographicalResult:
    def __init__(self):
        self._data = {}

    def add_result(self, data: dict):
        name = data['name']

        self._data[name] = {k: v for k, v in data.items() if k != 'name'}

    def _get_parameter(self,
                       parameter_key: str,
                       day_i: int,
                       asratio: bool) -> tuple:
        """
        :param parameter_key:

        :param day_i:               desired day of the simulation

        :param asratio:             return values as ratios of population sizes

        :returns:                   longitudes, latitudes, values at the given day
        """
        values = []
        longitudes = []
        latitudes = []

        try:
            for name, data in self._data.items():
                value = data[parameter_key][day_i]

                if asratio:
                    value /= data['size']

                values.append(value)
                longitudes.append(data['longitude'])
                latitudes.append(data['latitude'])

        except KeyError:
            raise KeyError(
                f'Parameter: {parameter_key} not found in the simulated data'
            )

        except IndexError:
            raise IndexError(
                f'Day {day_i} is out of the range of the simulation'
            )

        values = np.array(values)
        longitudes = np.array(longitudes)
        latitudes = np.array(latitudes)

        return longitudes, latitudes, values

    def get_timeseries(self, city_name: str) -> TimeSeriesResult:
        """
        :returns:           timeseries for the given city name
        """
        return TimeSeriesResult(
            simulation_days=self._data[city_name]['simulation_days'],
            infected=self._data[city_name]['infected'],
            unaffected=self._data[city_name]['unaffected'],
            new_cases=self._data[city_name]['new_cases'],
            immune=self._data[city_name]['immune'],
            dead=self._data[city_name]['dead'],
            hospitalized=self._data[city_name]['hospitalized']
        )

    def get_total_timeseries(self) -> TimeSeriesResult:
        first = True

        simulation_days = []
        infected = []
        unaffected = []
        new_cases = []
        immune = []
        dead = []
        hospitalized = []

        for city_name, city_data in self._data.items():
            if first:
                simulation_days = city_data['simulation_days']

                infected = np.array(city_data['infected'])
                unaffected = np.array(city_data['unaffected'])
                new_cases = np.array(city_data['new_cases'])
                immune = np.array(city_data['immune'])
                dead = np.array(city_data['dead'])
                hospitalized = np.array(city_data['hospitalized'])

                first = False

            else:
                infected += np.array(city_data['infected'])
                unaffected += np.array(city_data['unaffected'])
                new_cases += np.array(city_data['new_cases'])
                immune += np.array(city_data['immune'])
                dead += np.array(city_data['dead'])
                hospitalized += np.array(city_data['hospitalized'])

        return TimeSeriesResult(
            simulation_days=simulation_days,
            infected=infected.tolist(),
            unaffected=unaffected.tolist(),
            new_cases=new_cases.tolist(),
            immune=immune.tolist(),
            dead=dead.tolist(),
            hospitalized=hospitalized.tolist()
        )

    def get_mortalities(self, day_i=-1, asratio=False) -> tuple:
        return self._get_parameter('dead', day_i, asratio)

    def get_infected(self, day_i=-1, asratio=False) -> tuple:
        return self._get_parameter('infected', day_i, asratio)

    @classmethod
    def read_json(cls, filepath: str):
        """
        Loads data from a .json file
        """
        with open(filepath) as f:
            _data = json.load(f)

        instance = cls()

        for city_name, city_data in _data.items():
            city_data.update({
                'name': city_name
            })

            instance.add_result(city_data)

        return instance

    def to_json(self, filepath: str):
        """
        Saves results as a .json file to the given path
        """
        ensure_dir('/'.join(filepath.split('/')[:-1]))

        with open(filepath, 'w') as f:
            json.dump(self._data, f)
