import time

try:
    import cupy as cp

    cuda = True

except ImportError:
    import numpy as cp

    cuda = False

from tools.config import Config
from tools.general import singleton, ensure_dir
from tools.input_data import InputData
from tools.simulation.virus import Virus
from tools.plotting.checks import household_age_distribution, age_distribution, time_ranges


@singleton
class Population:
    """
    Represents the entire population used in the simulation.
    """

    def __init__(self,
                 virus: Virus):
        self._size = 0

        self._virus = virus
        self.config = Config()

        input_data = InputData()

        # === Attributes of cities:
        self._city_ids = None
        self._city_population_sizes = None
        self._city_infection_probs = None
        self._city_infected_counts = None
        self._city_populations = {
            i: p for i, p in zip(
                input_data.get_city_ids(),
                input_data.get_population_sizes()
            )
        }

        # === Attributes of individuals:
        self._city_id = None
        self._indices = None

        self._init_cities()

        self._is_alive = cp.ones(self._size).astype(bool)
        self._is_immune = cp.zeros(self._size).astype(bool)
        self._is_susceptible = cp.ones(self._size).astype(bool)
        self._is_infected = cp.zeros(self._size).astype(bool)
        self._is_infectious = cp.zeros(self._size).astype(bool)

        self._illness_days_total = cp.ones(self._size) * -1
        self._day_contracted = cp.ones(self._size) * -1

        self._respects_quarantine = cp.random.random(self._size) <= self.config.get(
            'population',
            'respect_quarantine'
        )
        self._is_in_quarantine = cp.zeros(self._size).astype(bool)

        self._age = cp.zeros(self._size).astype(int)
        self._unique_ages = None

        self._init_ages(input_data.age_distribution)

        self._health_condition = cp.random.random(self._size).astype(float)

        # === Meeting patterns:
        self._mean_stochastic_interactions = self.config.get('population', 'mean_stochastic_interactions')

        self._stochastic_interactions = cp.random.poisson(
            self._mean_stochastic_interactions,
            self._size
        )

        # === Symptoms:
        sympt_dice = cp.random.random(self._size)

        self._is_symptomatic = cp.zeros(self._size).astype(bool)

        for age, percentage in input_data.symptoms['symptomatic'].items():
            self._is_symptomatic[self._age == age] = (sympt_dice[self._age == age] <= percentage)

        self._hospitalization_percentage = cp.zeros(self._size).astype(float)

        for age, percentage in input_data.symptoms['hospitalized'].items():
            self._hospitalization_percentage[self._age == age] = percentage * \
                                                                 input_data.symptoms['symptomatic'][age]

        self._critical_care_percentage = cp.zeros(self._size).astype(float)

        for age, percentage in input_data.symptoms['critical_care'].items():
            self._critical_care_percentage[self._age == age] = percentage * \
                                                               input_data.symptoms['symptomatic'][age]

        self._death_percentage = cp.zeros(self._size).astype(float)

        for age, percentage in input_data.symptoms['fatal'].items():
            self._death_percentage[self._age == age] = percentage

        # === Attributes of households:
        self._household_meetings = {}
        self._household_meetings_elderly = None
        self._household_sizes = None
        self._max_household_size = None

        self._init_households(input_data.household_data)

        # === Medical needs:
        self._need_hospitalization = cp.zeros(self._size).astype(bool)
        self._hospitalization_start = cp.random.normal(
            self.config.get('hospitalization_start_mean'),
            self.config.get('hospitalization_start_std'),
            self._size
        )

        self._hospitalization_finish = self._hospitalization_start + cp.random.normal(
            self.config.get('hospitalization_length_mean'),
            self.config.get('hospitalization_length_std'),
            self._size
        )

        self._need_critical_care = cp.zeros(self._size).astype(bool)
        self._critical_care_start = self._hospitalization_start + cp.random.normal(
            self.config.get('critical_care_start_mean'),
            self.config.get('critical_care_start_std'),
            self._size
        )

        self._critical_care_finish = self._critical_care_start + cp.random.normal(
            self.config.get('critical_care_length_mean'),
            self.config.get('critical_care_length_std'),
            self._size
        )

        self._infectious_start = cp.random.normal(
            self.config.get('infectious_start_mean'),
            self.config.get('infectious_start_std'),
            self._size
        ).astype(float)

        self._healing_days = cp.random.normal(
            self._virus.illness_days_mean,
            self._virus.illness_days_std,
            self._size
        )

        self._day_i = 0

        self._update_infection_probs()

        self._plot_time_ranges()

        # === Result placeholders:

        simulation_days = self.config.get('simulation_days')

        self._statistics = {
            'susceptible': cp.zeros((simulation_days, len(self._city_populations))),
            'infected': cp.zeros((simulation_days, len(self._city_populations))),
            'immune': cp.zeros((simulation_days, len(self._city_populations))),
            'dead': cp.zeros((simulation_days, len(self._city_populations))),
            'hospitalized': cp.zeros((simulation_days, len(self._city_populations))),
            'critical_care': cp.zeros((simulation_days, len(self._city_populations)))
        }

    def _plot_time_ranges(self):
        """
        Plots histograms of time ranges as used in the simulation.

        Figure will be saved to the directory specified in the config file (check_plots)
        """
        check_plot_dir = self.config.get(
            'check_plots'
        )

        ensure_dir(check_plot_dir)

        need_hospital = self._health_condition <= self._hospitalization_percentage
        need_critical_care = self._health_condition <= self._critical_care_percentage

        healing = self._infectious_start + self._healing_days

        healing[need_hospital] = self._hospitalization_finish[need_hospital]
        healing[need_critical_care] = self._critical_care_finish[need_critical_care]

        if cuda:
            infectious = cp.asnumpy(self._infectious_start)
            healing = cp.asnumpy(healing)
            hospitalization = cp.asnumpy(self._hospitalization_start[need_hospital])
            critical_care = cp.asnumpy(self._critical_care_start[need_critical_care])

        else:
            infectious = self._infectious_start
            healing = healing
            hospitalization = self._hospitalization_start[need_hospital]
            critical_care = self._critical_care_start[need_critical_care]

        time_ranges(
            infectious=infectious,
            healing=healing,
            hospitalization=hospitalization,
            critical_care=critical_care,
            filepath=f'{check_plot_dir}/time-ranges.png'

        )

    def _init_households(self, household_data: dict):
        """
        Initializes parameters for households.

        Seniors living in pairs are simulated separately.
        """
        dice = cp.random.random(self._size)

        elderly_indexes = self._indices[(self._age >= 60) * (dice <= household_data['elderly'][2])]

        current_city_ids = self._city_id[elderly_indexes]

        meetings = [[], []]

        for city_id in self._city_ids:
            city_indexes = elderly_indexes[current_city_ids == city_id]

            split_indexes = cp.split(
                city_indexes[:int(len(city_indexes) / 2) * 2],
                2
            )

            for i, s in enumerate(split_indexes):
                meetings[i].append(s)

        self._household_meetings_elderly = {
            'first': cp.hstack(meetings[0]),
            'second': cp.hstack(meetings[1])
        }

        # === split the rest of the population into households:

        self._household_sizes = cp.ones(self._size)

        splitting_dice = cp.random.random(self._size)

        current_threshold = 0
        self._max_household_size = max([s for s in household_data['young'].keys()])

        for s, ratio in household_data['young'].items():
            mask = (current_threshold <= splitting_dice) * (splitting_dice < current_threshold + ratio)

            self._household_sizes[mask] = s

            current_threshold += ratio

        self._household_sizes[
            cp.hstack([
                self._household_meetings_elderly['first'],
                self._household_meetings_elderly['second'],
            ])
        ] = -1

        for household_size in range(2, self._max_household_size + 1):
            current_indexes = self._indices[self._household_sizes == household_size]

            if len(current_indexes) == 0:
                continue

            current_city_ids = self._city_id[current_indexes]

            meetings = [[] for i in range(household_size)]

            for city_id in self._city_ids:
                city_indexes = current_indexes[current_city_ids == city_id]

                split_indexes = cp.split(
                    city_indexes[:int(len(city_indexes) / household_size) * household_size],
                    household_size
                )

                for i, s in enumerate(split_indexes):
                    meetings[i].append(s)

            self._household_meetings[household_size] = [
                cp.hstack(m) for m in meetings
            ]

        self._household_sizes[self._household_sizes == -1] = 2

        # === plot the distribution of households:

        check_plot_dir = self.config.get('check_plots')

        ensure_dir(check_plot_dir)

        if cuda:
            ages = cp.asnumpy(self._age)
            household_sizes = cp.asnumpy(self._household_sizes)

        else:
            ages = self._age
            household_sizes = self._household_sizes

        household_age_distribution(
            ages=ages,
            household_sizes=household_sizes,
            filepath=f'{check_plot_dir}/household-distributions.png'
        )

        age_distribution(
            ages,
            filepath=f'{check_plot_dir}/age-distributions.png'
        )

    def _init_cities(self):
        """
        Initializes city ids for simulated people
        based on population sizes from the input data
        """
        self._size = int(sum(self._city_populations.values()))

        self._indices = cp.arange(self._size).astype(int)

        self._city_id = cp.ones(self._size) * -1

        self._city_ids = []
        self._city_population_sizes = []

        current_offset = 0

        for city_id, city_population in self._city_populations.items():
            self._city_id[current_offset:current_offset + city_population] = city_id

            current_offset += city_population

            self._city_ids.append(int(city_id))
            self._city_population_sizes.append(int(city_population))

        self._city_id_array = cp.array(self._city_ids)

        self._city_population_sizes = cp.array(self._city_population_sizes)

    def _init_ages(self, age_data: dict):
        """
        Initializes ages for simulated people
        based on population sizes from the input data
        """
        age_dice = cp.random.random(self._size)

        age_total = sum(age_data.values())

        current_threshold = 0

        self._unique_ages = []

        for age, age_population in age_data.items():
            age_mask = (current_threshold <= age_dice) * (age_dice < current_threshold + age_population / age_total)

            self._age[age_mask] = age

            current_threshold += age_population / age_total

            self._unique_ages.append(age)

    def _update_statistics(self):
        """
        Writes current results to self._statistics
        """
        _, susceptible = self.get_susceptible_by_city()
        self._statistics['susceptible'][self._day_i][:] = susceptible

        _, infected = self.get_infected_by_city()
        self._statistics['infected'][self._day_i][:] = infected

        _, immune = self.get_immune_by_city()
        self._statistics['immune'][self._day_i][:] = immune

        _, dead = self.get_dead_by_city()
        self._statistics['dead'][self._day_i][:] = dead

        _, hospitalized = self.get_hospitalized_by_city()
        self._statistics['hospitalized'][self._day_i][:] = hospitalized

        _, critical_care = self.get_critical_care_by_city()
        self._statistics['critical_care'][self._day_i][:] = critical_care

    def get_results(self) -> dict:
        """
        :returns:               json-serializable dict of results in the format required by
                                tools/data_structure.GeographicalResult
        """
        return {
            'susceptible': self._statistics['susceptible'].tolist(),
            'infected': self._statistics['infected'].tolist(),
            'immune': self._statistics['immune'].tolist(),
            'dead': self._statistics['dead'].tolist(),
            'hospitalized': self._statistics['hospitalized'].tolist(),
            'critical_care': self._statistics['critical_care'].tolist()
        }

    def get_susceptible(self) -> cp.ndarray:
        """
        Updates the susceptible mask

        :returns:           <cp.ndarray<bool>>  mask defining whether individuals are susceptible
        """
        self._is_susceptible = ~self._is_immune * self._is_alive * ~self._is_infected

        return self._is_susceptible

    def get_susceptible_by_city(self, as_json=False) -> tuple:
        """
        :param as_json:     if True returns a json-serializable values, otherwise cp/np.ndarray

        :returns:           tuple of city_ids and susceptible people counts respectively
        """
        susceptible = self._city_id[self._is_susceptible]

        if len(susceptible) == 0:
            if as_json:
                return self._city_ids, [0 for i in self._city_ids]

            else:
                return self._city_ids, cp.zeros(len(self._city_ids))

        city_ids, values = cp.unique(susceptible, return_counts=True)

        return self._sort_by_city_ids(city_ids, values, as_json=as_json)

    def get_n_susceptible(self) -> int:
        return int(self._is_susceptible.astype(int).sum())

    def get_infected_by_city(self, as_json=False) -> tuple:
        """
        :param as_json:     if True returns a json-serializable values, otherwise cp/np.ndarray

        :returns:           tuple of city_ids and infected people counts respectively
        """
        infected = self._city_id[self._is_infected]

        if len(infected) == 0:
            if as_json:
                return self._city_ids, [0 for i in self._city_ids]

            else:
                return self._city_ids, cp.zeros(len(self._city_ids))

        city_ids, values = cp.unique(infected, return_counts=True)

        return self._sort_by_city_ids(city_ids, values, as_json=as_json)

    def get_n_infected(self) -> int:
        return int(self._is_infected.astype(int).sum())

    def get_immune_by_city(self, as_json=False) -> tuple:
        """
        :param as_json:     if True returns a json-serializable values, otherwise cp/np.ndarray

        :returns:           tuple of city_ids and infected people counts respectively
        """
        immune = self._city_id[self._is_immune]

        if len(immune) == 0:
            if as_json:
                return self._city_ids, [0 for i in self._city_ids]

            else:
                return self._city_ids, cp.zeros(len(self._city_ids))

        city_ids, values = cp.unique(immune, return_counts=True)

        return self._sort_by_city_ids(city_ids, values, as_json=as_json)

    def get_n_immune(self) -> int:
        return int(self._is_immune.astype(int).sum())

    def get_dead_by_city(self, as_json=False) -> tuple:
        """
        :param as_json:     if True returns a json-serializable values, otherwise cp/np.ndarray

        :returns:           tuple of city_ids and infected people counts respectively
        """
        dead = self._city_id[~self._is_alive]

        if len(dead) == 0:
            if as_json:
                return self._city_ids, [0 for i in self._city_ids]

            else:
                return self._city_ids, cp.zeros(len(self._city_ids))

        city_ids, values = cp.unique(dead, return_counts=True)

        return self._sort_by_city_ids(city_ids, values, as_json=as_json)

    def get_hospitalized_by_city(self, as_json=False) -> tuple:
        """
        :param as_json:     if True returns a json-serializable values, otherwise cp/np.ndarray

        :returns:           tuple of city_ids and hospitalized people counts respectively
        """
        hospitalized = self._city_id[self._need_hospitalization]

        if len(hospitalized) == 0:
            if as_json:
                return self._city_ids, [0 for i in self._city_ids]

            else:
                return self._city_ids, cp.zeros(len(self._city_ids))

        city_ids, values = cp.unique(hospitalized, return_counts=True)

        return self._sort_by_city_ids(city_ids, values, as_json=as_json)

    def get_critical_care_by_city(self, as_json=False) -> tuple:
        """
        :param as_json:     if True returns a json-serializable values, otherwise cp/np.ndarray

        :returns:           tuple of city_ids and people-at-critical-care counts respectively
        """
        critical_care = self._city_id[self._need_critical_care]

        if len(critical_care) == 0:
            if as_json:
                return self._city_ids, [0 for i in self._city_ids]

            else:
                return self._city_ids, cp.zeros(len(self._city_ids))

        city_ids, values = cp.unique(critical_care, return_counts=True)

        return self._sort_by_city_ids(city_ids, values, as_json=as_json)

    def get_dead_ages(self):
        """
        :returns:           numpy array with ages of those who passed away
        """
        if cuda:
            return cp.asnumpy(self._age[~self._is_alive])

        else:
            return self._age[~self._is_alive]

    def _sort_by_city_ids(self, city_ids, values, dtype=None, default=0, as_json=False) -> tuple:
        """
        Sorts the inputs such that they are ordered by the city_ids as defined in the constructor

        :param as_json:         if True returns a json-serializable values, otherwise cp/np.ndarray

        :param city_ids:        city ids as returned byt the unique functions

        :param values:          values as returned byt the unique functions

        :param default:         default value for city ids not included in the input city ids

        :return:                city_ids, corresponding_values
        """
        indexes = cp.searchsorted(self._city_id_array, city_ids)

        if default == 0:
            res = cp.zeros(len(self._city_ids))

        else:
            res = cp.ones(len(self._city_ids)) * default

        res[indexes] = values

        if dtype is not None:
            res = res.astype(dtype)

        if as_json:
            res = res.tolist()

        return self._city_ids, res

    def _update_infection_probs(self, random_seed=None):
        """
        Updates probability of infection based on how many inhabitants
        are infectious each city
        """
        if random_seed is None:
            self._reset_random_seed()

        else:
            cp.random.seed(random_seed)

        infecious_city_ids = self._city_id[self._is_infectious]

        if len(infecious_city_ids) == 0:
            self._city_infected_counts = cp.zeros(len(self._city_ids))

        else:
            city_ids, infected_counts = cp.unique(infecious_city_ids, return_counts=True)

            _, self._city_infected_counts = self._sort_by_city_ids(city_ids, infected_counts, as_json=False)

        self._city_infection_probs = self._city_infected_counts / self._city_population_sizes * \
                                     self._virus.transmission_probability

    def travel(self, migration_matrix: cp.ndarray):
        """
        Increases probability of infection based on how many infected people
        have come to the given cities

        :param migration_matrix:    matrix containing numbers of people travelling among cities
        """
        incoming_infected = cp.matmul(migration_matrix, self._city_infected_counts / self._city_population_sizes)
        incoming_total = migration_matrix.sum(axis=0)

        self._city_infection_probs += incoming_infected / (incoming_total + self._city_population_sizes) * \
                                      self._virus.transmission_probability

    def _spread_in_cities(self, random_seed=None):
        """
        Infects individuals in the given cities by letting them interact
        with current infection probabilities.

        :param random_seed:         random seed to be used
        """
        if random_seed is None:
            self._reset_random_seed()

        else:
            raise NotImplementedError(
                'Moved to households, shall be redefined later'
            )
            cp.random.seed(random_seed)

        city_probs = {cid: prob for cid, prob in zip(self._city_ids, self._city_infection_probs)}

        probabilities = cp.zeros(self._size).astype(float)

        current_offset = 0

        for city_id, city_population in self._city_populations.items():
            probabilities[current_offset:current_offset + city_population] = city_probs[city_id]

            current_offset += city_population

        susceptible_indexes = self._indices[self.get_susceptible()]

        poisson_mean = self._stochastic_interactions[susceptible_indexes]

        susceptible_probabilities = probabilities[self._is_susceptible]

        interaction_multiplicities = cp.random.poisson(
            poisson_mean,
            len(susceptible_indexes)
        )

        indexes = cp.arange(len(susceptible_indexes))

        for interaction_i in range(int(interaction_multiplicities.max())):
            current_indexes = indexes[interaction_multiplicities > interaction_i]

            susceptible = susceptible_indexes[current_indexes]
            current_probs = susceptible_probabilities[current_indexes]

            infected = susceptible[cp.random.random(len(susceptible)) <= current_probs]

            self._is_infected[infected] = True
            self._day_contracted[infected] = self._day_i

    def _spread_in_households(self):
        """
        Infects individuals in households by letting them interact
        with current infection probabilities.
        """
        first_infectious = self._is_infectious[self._household_meetings_elderly['first']]
        first_susceptible = self._is_susceptible[self._household_meetings_elderly['first']]

        second_infectious = self._is_infectious[self._household_meetings_elderly['second']]
        second_susceptible = self._is_susceptible[self._household_meetings_elderly['second']]

        transmissions12 = first_susceptible * second_infectious * (
                cp.random.random(
                    len(first_susceptible)
                ) <= self._virus.household_transmission_probability
        )

        transmissions21 = second_susceptible * first_infectious * (
                cp.random.random(
                    len(first_susceptible)
                ) <= self._virus.household_transmission_probability
        )

        infected_ids = []

        for household_size, household_ids in self._household_meetings.items():
            n_infectious = self._is_infectious[household_ids[0]].astype(int)

            for ids in household_ids[1:]:
                n_infectious += self._is_infectious[ids]

            infection_probs = n_infectious * self._virus.household_transmission_probability

            for ids in household_ids:
                susceptible_mask = self._is_susceptible[ids]

                susceptible_ids = ids[susceptible_mask]

                infected = susceptible_ids[cp.random.random(len(susceptible_ids)) <= infection_probs[susceptible_mask]]

                if len(infected) != 0:
                    infected_ids.append(infected)

        newly_infected = cp.hstack(
            [
                self._household_meetings_elderly['second'][transmissions12],
                self._household_meetings_elderly['first'][transmissions21]
            ] + infected_ids
        )

        self._is_infected[newly_infected] = True
        self._day_contracted[newly_infected] = self._day_i

    def infect_by_cities(self, city_ids, infection_counts, random_seed=None):
        """
        Infects random people in the given cities
        """
        if random_seed is None:
            self._reset_random_seed()

        else:
            cp.random.seed(random_seed)

        self.get_susceptible()

        for cid, count in zip(city_ids, infection_counts):
            susceptible = self._indices[self._is_susceptible * (self._city_id == cid)]

            if count < len(susceptible):
                newly_infected = cp.random.choice(susceptible, count, replace=False)

            else:
                newly_infected = susceptible

            self._is_infected[newly_infected] = True
            self._day_contracted[newly_infected] = self._day_i

    def _update_infectiousness(self):
        """
        Makes people infectious once they are infected long enough
        """
        infected_indexes = self._indices[self._is_infected]

        infectious = infected_indexes[
            self._infectious_start[infected_indexes] <= self._day_i - self._day_contracted[infected_indexes]
            ]

        self._is_infectious[infectious] = True

        infectious_indexes = self._indices[self._is_infectious]

        # Put into quarantine those who have symptoms for more than a day and respect the quarantine
        in_quarantine = (self._day_i - self._day_contracted[infectious_indexes] -
                         self._infectious_start[infectious_indexes] >= 1) * \
                        self._is_symptomatic[infectious_indexes] * self._respects_quarantine[infectious_indexes]

        self._is_infectious[infectious_indexes[in_quarantine]] = False

        try:
            prob_tested = self.config.get('n_tests_daily') / (
                    int(self._is_infectious.astype(int).sum()) * self._mean_stochastic_interactions
            )

        except ZeroDivisionError:
            prob_tested = 0

        in_quarantine = cp.random.random(len(infectious_indexes)) <= prob_tested

        self._is_infectious[infectious_indexes[in_quarantine]] = False

    def _hospitalize(self):
        """
        Manages hospitalization, critical care and people who passed away
        """
        infected_indexes = self._indices[self._is_infected]

        hospitalization_mask = (
                                       self._health_condition[infected_indexes] <=
                                       self._hospitalization_percentage[infected_indexes]
                               ) * (
                                       self._hospitalization_start[infected_indexes] >=
                                       (self._day_i - self._day_contracted[infected_indexes])
                               )

        self._need_hospitalization[infected_indexes[hospitalization_mask]] = True
        self._is_infectious[infected_indexes[hospitalization_mask]] = False

        critical_care_mask = (
                                     self._health_condition[infected_indexes] <=
                                     self._critical_care_percentage[infected_indexes]
                             ) * (
                                     self._critical_care_start[infected_indexes] >=
                                     (self._day_i - self._day_contracted[infected_indexes])
                             )

        self._need_critical_care[infected_indexes[critical_care_mask]] = True
        self._need_hospitalization[infected_indexes[critical_care_mask]] = False

        hospitalized_indexes = self._indices[self._need_hospitalization]

        healed = self._day_i - self._day_contracted[hospitalized_indexes] + \
                 self._hospitalization_start[hospitalized_indexes] > self._hospitalization_finish[hospitalized_indexes]

        healed_indexes = hospitalized_indexes[healed]

        self._need_hospitalization[healed_indexes] = False

        self._is_infected[healed_indexes] = False
        self._is_infectious[healed_indexes] = False

        self._is_immune[healed_indexes] = True

        critical_care_indexes = self._indices[self._need_critical_care]

        finished = self._day_i - self._day_contracted[critical_care_indexes] + \
                   self._critical_care_start[critical_care_indexes] > self._critical_care_finish[critical_care_indexes]

        finished_indexes = critical_care_indexes[finished]

        healed = finished_indexes[self._health_condition[finished_indexes] > self._death_percentage[finished_indexes]]

        self._need_critical_care[healed] = False

        self._is_infected[healed] = False
        self._is_infectious[healed] = False

        self._illness_days_total[healed] = self._day_i - self._day_contracted[healed]

        self._is_immune[healed] = True

        passed_away = finished_indexes[
            self._health_condition[finished_indexes] <= self._death_percentage[finished_indexes]
            ]

        self._is_alive[passed_away] = False
        self._is_infected[passed_away] = False
        self._is_infectious[passed_away] = False

        self._need_hospitalization[passed_away] = False
        self._need_critical_care[passed_away] = False

    def _heal(self):
        """
        Heals members of the population if they are infected long enough
        """
        ill_indexes = self._indices[self._is_infected]
        ill_indexes = ill_indexes[self._health_condition[ill_indexes] > self._hospitalization_percentage[ill_indexes]]

        healed = (self._day_i - self._day_contracted[ill_indexes] - self._infectious_start[ill_indexes]) >= abs(
            self._healing_days[ill_indexes]
        )

        healed_indexes = ill_indexes[healed]

        self._is_infected[healed_indexes] = False
        self._is_infectious[healed_indexes] = False

        self._illness_days_total[healed_indexes] = self._day_i - self._day_contracted[healed_indexes]

        self._is_immune[healed_indexes] = True

    def _update_restrictions(self):
        # if self._day_i == 10:
        #     self._stochastic_interactions = self._stochastic_interactions * 0.52
        #     self._mean_stochastic_interactions *= 0.52
        #
        #     self._virus.household_transmission_probability *= 1.1
        pass

    def next_day(self):
        """
        Next day of the simulation
        """
        self._update_restrictions()

        self._update_infectiousness()

        self._spread_in_cities()

        self._spread_in_households()

        self._heal()

        self._hospitalize()

        self._update_infection_probs()

        self._update_statistics()

        self._day_i += 1

    @staticmethod
    def _reset_random_seed():
        """
        Resets numpy's random seed based on unix timestamp
        with 10 nanosecond precision
        """
        current_time = time.time() * 1e8

        cp.random.seed(
            int(current_time % (2 ** 32 - 1))
        )
