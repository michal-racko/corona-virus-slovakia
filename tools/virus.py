import re
import typing

from abc import ABCMeta, abstractmethod

virus_type = typing.TypeVar('virus_type', bound='Virus')


class Virus:
    """
    Abstracts a virus which might spread among people
    """
    __metaclass__ = ABCMeta

    def __init__(self,
                 illness_days_mean: int,
                 illness_days_std: float):
        """
        :param illness_days_mean:       Mean period for which a person is ill

        :param illness_days_std:        Standard deviation of illness days
        """
        self.illness_days_mean = illness_days_mean
        self.illness_days_std = illness_days_std

    @abstractmethod
    def get_mortality(self, **kwargs) -> float:
        """
        :returns:       mortality ratio for the given parameters
        """
        pass

    @abstractmethod
    def get_transmission_probability(self, **kwargs) -> float:
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
    def __init__(self):
        self._mortality = 0.05
        self._transmission_probability = 0.05

        super().__init__(
            illness_days_mean=10,
            illness_days_std=3
        )

    def get_mortality(self, **kwargs) -> float:
        return self._mortality

    def get_transmission_probability(self, **kwargs) -> float:
        return self._transmission_probability
