import re

from abc import ABCMeta, abstractmethod


class Virus:
    """
    Abstracts a virus which might spread among people
    """
    __metaclass__ = ABCMeta

    def __init__(self,
                 incubation_period: int,
                 illness_days: int):
        self.incubation_period = incubation_period
        self.illness_days = illness_days

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
                    **kwargs):
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
        self.mortality = 0.05
        self.transmission_probability = 0.05

        super().__init__(
            incubation_period=10,
            illness_days=7
        )

    def get_mortality(self, **kwargs) -> float:
        return self.mortality

    def get_transmission_probability(self, **kwargs) -> float:
        return self.transmission_probability
