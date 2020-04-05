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
                 susceptible: list,
                 immune: list,
                 dead: list,
                 hospitalized: list,
                 critical_care: list):
        assert len(infected) == len(simulation_days)
        assert len(susceptible) == len(simulation_days)
        assert len(immune) == len(simulation_days)
        assert len(dead) == len(simulation_days)
        assert len(hospitalized) == len(simulation_days)
        assert len(critical_care) == len(simulation_days)

        self.days = simulation_days
        self.infected = infected
        self.susceptible = susceptible
        self.new_cases = None
        self.immune = immune
        self.dead = dead
        self.hospitalized = hospitalized
        self.critical_care = critical_care

        self._prepare_new_cases()

    def _prepare_new_cases(self):
        infected = np.zeros(len(self.infected) + 1)

        self.new_cases = np.diff(infected)

    def to_json(self, filepath: str):
        """
        Saves results as a .json file to the given path
        """
        ensure_dir('/'.join(filepath.split('/')[:-1]))

        with open(filepath, 'w') as f:
            json.dump({
                'days': self.days,
                'infected': self.infected,
                'susceptible': self.susceptible,
                'new_cases': self.new_cases,
                'immune': self.immune,
                'dead': self.dead,
                'hospitalized': self.hospitalized,
                'critical_care': self.critical_care,
            }, f)


class GeographicalResult:
    def __init__(self):
        self._city_ids = None
        self._city_names = None
        self._city_sizes = None

        self._cid_to_name = None
        self._name_to_cid = None

        self._longitudes = None
        self._latitudes = None

        self._data = {
            'susceptible': [],
            'infected': [],
            'immune': [],
            'dead': [],
            'hospitalized': [],
            'critical_care': []
        }

    def to_json(self, filepath: str):
        """
        Saves results as a .json file to the given path
        """
        ensure_dir('/'.join(filepath.split('/')[:-1]))

        with open(filepath, 'w') as f:
            json.dump({
                'data': self._data,
                'city_ids': self._city_ids,
                'city_names': self._city_names,
                'city_sizes': self._city_sizes,
                'longitudes': self._longitudes,
                'latitudes': self._latitudes
            }, f)

    @classmethod
    def read_json(cls, filepath: str):
        """
        Loads data from a .json file
        """
        with open(filepath) as f:
            _data = json.load(f)

        instance = cls()

        instance.set_data(_data['data'])
        instance.set_city_coords(
            _data['longitudes'],
            _data['latitudes']
        )
        instance.set_city_ids(
            _data['city_ids'],
            _data['city_names']
        )
        instance.set_city_sizes(
            _data['city_sizes']
        )

        return instance

    def set_data(self, data: dict):
        self._data = data

    def set_city_ids(self, city_ids, city_names):
        self._city_ids = [int(cid) for cid in city_ids]
        self._city_names = [str(cn) for cn in city_names]

        self._cid_to_name = {cid: cn for cid, cn in zip(city_ids, city_names)}
        self._name_to_cid = {cn: cid for cid, cn in zip(city_ids, city_names)}

    def set_city_sizes(self, city_sizes):
        self._city_sizes = [int(cs) for cs in city_sizes]

    def set_city_coords(self, longitudes, latitudes):
        self._longitudes = [float(l) for l in longitudes]
        self._latitudes = [float(l) for l in latitudes]

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
        try:
            values = np.array(self._data[parameter_key][day_i])

        except KeyError:
            raise KeyError(
                f'Parameter: {parameter_key} not found in the simulated data'
            )

        except IndexError:
            raise IndexError(
                f'Day {day_i} is out of the range of the simulation'
            )

        if asratio:
            values = values / np.array(self._city_sizes)

        return np.array(self._longitudes), np.array(self._latitudes), values

    def get_timeseries(self, city_name: str) -> TimeSeriesResult:
        """
        :returns:           timeseries for the given city name
        """
        city_index = self._city_names.index(city_name)

        return TimeSeriesResult(
            simulation_days=[i for i in range(len(np.array(self._data['infected']).T[city_index]))],
            infected=np.array(self._data['infected']).T[city_index],
            susceptible=np.array(self._data['susceptible']).T[city_index],
            immune=np.array(self._data['immune']).T[city_index],
            dead=np.array(self._data['dead']).T[city_index],
            hospitalized=np.array(self._data['hospitalized']).T[city_index],
            critical_care=np.array(self._data['critical_care']).T[city_index]
        )

    def get_total_timeseries(self) -> TimeSeriesResult:
        return TimeSeriesResult(
            simulation_days=[i for i in range(len(np.array(self._data['infected']).T[0]))],
            infected=np.array(self._data['infected']).sum(axis=1),
            susceptible=np.array(self._data['susceptible']).sum(axis=1),
            immune=np.array(self._data['immune']).sum(axis=1),
            dead=np.array(self._data['dead']).sum(axis=1),
            hospitalized=np.array(self._data['hospitalized']).sum(axis=1),
            critical_care=np.array(self._data['critical_care']).sum(axis=1)
        )

    def get_mortalities(self, day_i=-1, asratio=False) -> tuple:
        return self._get_parameter('dead', day_i, asratio)

    def get_infected(self, day_i=-1, asratio=False) -> tuple:
        return self._get_parameter('infected', day_i, asratio)
