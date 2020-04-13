import re

from abc import ABCMeta, abstractmethod


class Restrictions:
    """
    Abstracts restrictions on the simulated population
    """

    __metaclass__ = ABCMeta

    @abstractmethod
    def impose(self, population):
        """
        Imposes desired restrictions on the given population by mutating
        its attributes.

        :param population:          object of the Population class
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


class Interactions(Restrictions):
    """
    Abstracts restrictions on random interactions
    """

    def __init__(self,
                 day_start: int,
                 day_end: int,
                 ratio: float):
        self._day_start = day_start
        self._day_end = day_end

        self._ratio = ratio

    def impose(self, population):
        if population.day_i == self._day_start:
            population.stochastic_interactions = population.stochastic_interactions * self._ratio
            population.mean_stochastic_interactions = population.mean_stochastic_interactions * self._ratio

            if population.social_network is not None:
                population.social_network.multiply_daily_fraction(self._ratio)

            population.restrictions_start.append(
                population.day_i
            )

        elif population.day_i == self._day_end:
            population.stochastic_interactions = population.stochastic_interactions / self._ratio
            population.mean_stochastic_interactions = population.mean_stochastic_interactions / self._ratio

            if population.social_network is not None:
                population.social_network.multiply_daily_fraction(1 / self._ratio)

            population.restrictions_end.append(
                population.day_i
            )


class ICUBased(Restrictions):
    """
    Abstracts restrictions on random interactions
    based on the ratio of occupied ICUs
    """
    def __init__(self,
                 icu_ratio_start: float,
                 icu_ratio_end: float,
                 ratio: float):
        self._icu_ratio_start = icu_ratio_start
        self._icu_ratio_end = icu_ratio_end

        self._ratio = ratio

        self._imposed = False

    def impose(self, population):
        occupied_icu_ratio = population.occupied_icu_beds / population.n_icu_beds

        if not self._imposed and occupied_icu_ratio > self._icu_ratio_start:
            population.stochastic_interactions = population.stochastic_interactions * self._ratio
            population.mean_stochastic_interactions = population.mean_stochastic_interactions * self._ratio

            if population.social_network is not None:
                population.social_network.multiply_daily_fraction(self._ratio)

            population.restrictions_start.append(
                population.day_i
            )

            self._imposed = True

        if self._imposed and occupied_icu_ratio < self._icu_ratio_end:
            population.stochastic_interactions = population.stochastic_interactions / self._ratio
            population.mean_stochastic_interactions = population.mean_stochastic_interactions / self._ratio

            if population.social_network is not None:
                population.social_network.multiply_daily_fraction(1 / self._ratio)

            population.restrictions_end.append(
                population.day_i
            )

            self._imposed = False
