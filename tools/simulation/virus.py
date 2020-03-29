import re
import typing
import logging

from abc import ABCMeta, abstractmethod

from tools.config import Config
from tools.input_data import InputData

virus_type = typing.TypeVar('virus_type', bound='Virus')


class Virus:
    """
    Abstracts a virus which might spread among people
    """
    __metaclass__ = ABCMeta

    def __init__(self,
                 asymptomatic_ratio: float,
                 hospitalized_ratio: float):
        config = Config()

        self.illness_days_mean = config.get('virus', 'infectious_days_mean')
        self.illness_days_std = config.get('virus', 'infectious_days_std')

        self.transmission_probability = config.get('virus', 'transmission_probability')

        self.asymptomatic_ratio = asymptomatic_ratio
        self.hospitalized_ratio = hospitalized_ratio
        self.mild_symptoms_ratio = 1 - hospitalized_ratio - asymptomatic_ratio

        input_data = InputData()

        mean_periodic_interactions = config.get('population', 'mean_periodic_interactions')
        mean_stochastic_interactions = config.get('population', 'mean_stochastic_interactions')

        mean_interactions = mean_periodic_interactions + mean_stochastic_interactions

        self.R = (1 + input_data.mean_travel_ratio) * \
                 mean_interactions * self.illness_days_mean * self.transmission_probability

        logging.info(
            f'Initialized the {self.__class__.__name__} virus with R0={self.R:.4f}'
        )

    @abstractmethod
    def get_mortality(self, **kwargs) -> float:
        """
        :returns:       mortality ratio for the given parameters
        """
        pass

    @classmethod
    def from_string(cls,
                    string_option: str,
                    *args,
                    **kwargs) -> virus_type:
        """
        Creates an object of the subclass whose name matches string_option.

        :param string_option:           a string defining which subclass to be used

        :return:                        an object of the corresponding subclass

        :raises NotImplementedError:    if no matching subclass found
        """
        _option = ''.join(
            re.findall(r'[0-9a-zA-Z]', string_option)
        ).lower()

        for subclass in cls.__subclasses__():
            if subclass.__name__.lower() == _option:
                return subclass(*args, **kwargs)

        else:
            raise NotImplementedError(
                f'No corresponding class found for option "{string_option}"'
            )


class SARSCoV2(Virus):
    def __init__(self, *args, **kwargs):
        super().__init__(
            asymptomatic_ratio=0.4,
            hospitalized_ratio=0.1
        )

        self.mortality = 0.03

        input_data = InputData()

        self.age_mortalities = input_data.symptoms['fatal']
        self.age_critical_care = input_data.symptoms['critical_care']
        self.age_hospitalized = input_data.symptoms['hospitalized']

    def get_mortality(self, age, **kwargs) -> float:
        return self.age_mortalities[age]
