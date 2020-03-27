import logging

import numpy as np
import matplotlib.pyplot as pl

from tools.config import Config
from tools.general import ensure_dir
from tools.simulation.virus import Virus
from tools.data_structure import TimeSeriesResult


def plot_pandemic(data: TimeSeriesResult, filepath=None):
    config = Config()

    virus = Virus.from_string(
        config.get('virus', 'name'),
        illness_days_mean=config.get('virus', 'infectious_days_mean'),
        illness_days_std=config.get('virus', 'infectious_days_std'),
        transmission_probability=config.get('virus', 'transmission_probability'),
        mean_periodic_interactions=config.get('virus', 'infectious_days_mean'),
        mean_stochastic_interactions=config.get('virus', 'infectious_days_mean')
    )

    fig, (ax1, ax2, ax3) = pl.subplots(
        3,
        sharex=True,
        figsize=(8, 10)
    )

    ax1.set_title('COVID19 simulation', fontsize=20)

    p1 = ax1.plot(
        data.days,
        data.unaffected,
        'tab:blue',
        linewidth=2
    )

    p2 = ax1.plot(
        data.days,
        data.infected,
        'tab:red',
        linewidth=2
    )

    p3 = ax1.plot(
        data.days,
        data.immune,
        'tab:green',
        linewidth=2
    )

    ax1.legend(
        (p1[0], p2[0], p3[0]),
        ('Susceptible', 'Infected', 'Immune')
    )

    ax1.grid(True)

    deaths = np.array(data.dead)
    deaths_daily = np.zeros(len(deaths))
    deaths_daily[1:] = np.diff(deaths)

    p4 = ax2.plot(
        data.days,
        data.new_cases,
        'tab:red',
        linewidth=2
    )

    p5 = ax2.plot(
        data.days,
        deaths_daily,
        'tab:blue',
        linewidth=2
    )

    p6 = ax2.plot(
        data.days,
        data.dead,
        'k',
        linewidth=2
    )

    ax2.legend((p4[0], p5[0], p6[0]), ('New cases', 'Dead daily', 'Dead cumulative'))

    ax2.grid(True)

    infected = np.array(data.infected)

    p7 = ax3.plot(
        data.days,
        infected * virus.asymptomatic_ratio,
        'tab:blue',
        linewidth=2
    )

    p8 = ax3.plot(
        data.days,
        infected * virus.mild_symptoms_ratio,
        'tab:green',
        linewidth=2
    )

    p9 = ax3.plot(
        data.days,
        data.hospitalized,
        'tab:red',
        linewidth=2
    )

    ax3.legend((p7[0], p8[0], p9[0],), ('Asymptomatic', 'Mild symptoms', 'Hospitalized'))
    ax3.set_xlabel('Days', fontsize=16)

    ax3.grid(True)

    pl.tight_layout()

    if filepath is None:
        pl.show()

    else:
        ensure_dir('/'.join(filepath.split('/')[:-1]))

        pl.savefig(filepath)
        pl.close()

        logging.info(f'Result plot saved to: {filepath}')



