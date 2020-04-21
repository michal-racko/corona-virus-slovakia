import re

import cupy as cp

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

            if population.social_network is not None:
                population.social_network.multiply_daily_fraction(self._ratio)

            population.restrictions_start.append(
                population.day_i
            )

        elif population.day_i == self._day_end:
            population.stochastic_interactions = population.stochastic_interactions / self._ratio

            if population.social_network is not None:
                population.social_network.multiply_daily_fraction(1 / self._ratio)

            population.restrictions_end.append(
                population.day_i
            )


class InteractionsElderly(Restrictions):
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
            elderly_mask = population.age >= 60

            population.stochastic_interactions[elderly_mask] = population.stochastic_interactions[
                                                                   elderly_mask] * self._ratio

        elif population.day_i == self._day_end:
            elderly_mask = population.age >= 60

            population.stochastic_interactions[elderly_mask] = population.stochastic_interactions[
                                                                   elderly_mask] / self._ratio


class Travel(Restrictions):
    """
    Abstracts restrictions on travel among municipalities
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
            population.migration_matrix = population.migration_matrix * self._ratio

        elif population.day_i == self._day_end:
            population.migration_matrix = population.migration_matrix / self._ratio


class AdaptiveTravel(Restrictions):
    def __init__(self,
                 ratio: float,
                 threshold: float,
                 duration: int):
        self._ratio = ratio

        self._threshold = threshold
        self._duration = duration

        self._day_imposed = None

    def impose(self, population):
        if population.day_i == 0:
            self._day_imposed = cp.ones(len(population.migration_matrix)) * -1

        infected_ratios = population.city_infection_probs

        over_threshold = infected_ratios > self._threshold

        # impose quarantine

        update_mask = over_threshold * (self._day_imposed == -1)

        if update_mask.astype(int).sum() != 0:
            population.migration_matrix[update_mask] = population.migration_matrix[update_mask] * self._ratio

            self._day_imposed[update_mask] = population.day_i

        # wow quarantine

        update_mask = (population.day_i - self._day_imposed > self._duration) * (self._day_imposed != -1)

        if update_mask.astype(int).sum() != 0:
            population.migration_matrix[update_mask] = population.migration_matrix[update_mask] / self._ratio

            self._day_imposed[update_mask] = -1


class AdaptiveInteractions(Restrictions):
    def __init__(self,
                 ratio: float,
                 threshold: float,
                 duration: int):
        self._ratio = ratio

        self._threshold = threshold
        self._duration = duration

        self._day_imposed = None

    def impose(self, population):
        if population.day_i == 0:
            self._day_imposed = cp.ones(len(population.city_ids)) * -1

        infected_ratios = population.city_infection_probs

        over_threshold = infected_ratios > self._threshold

        # impose quarantine

        update_mask = over_threshold * (self._day_imposed == -1)

        if update_mask.astype(int).sum() != 0:
            affected_cities = cp.array(population.city_ids)[update_mask]

            for cid in affected_cities:
                population.city_lockdown[population.city_id == cid] = True
                population.stochastic_interactions[population.city_id == cid] = population.stochastic_interactions[
                                                                                    population.city_id == cid
                                                                                    ] * self._ratio

            self._day_imposed[update_mask] = population.day_i

        # wow quarantine

        update_mask = (population.day_i - self._day_imposed > self._duration) * (self._day_imposed != -1)

        if update_mask.astype(int).sum() != 0:
            affected_cities = cp.array(population.city_ids)[update_mask]

            for cid in affected_cities:
                population.city_lockdown[population.city_id == cid] = False
                population.stochastic_interactions[population.city_id == cid] = population.stochastic_interactions[
                                                                                    population.city_id == cid
                                                                                    ] / self._ratio

            self._day_imposed[update_mask] = -1


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

            if population.social_network is not None:
                population.social_network.multiply_daily_fraction(self._ratio)

            population.restrictions_start.append(
                population.day_i
            )

            self._imposed = True

        if self._imposed and occupied_icu_ratio < self._icu_ratio_end:
            population.stochastic_interactions = population.stochastic_interactions / self._ratio

            if population.social_network is not None:
                population.social_network.multiply_daily_fraction(1 / self._ratio)

            population.restrictions_end.append(
                population.day_i
            )

            self._imposed = False
