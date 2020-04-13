import logging

import numpy as np
import matplotlib.pyplot as pl

from tools.config import Config
from tools.general import ensure_dir
from tools.simulation.virus import Virus
from tools.data_structure import TimeSeriesResult


def plot_pandemic(data: TimeSeriesResult, filepath=None, logscale=False):
    config = Config()

    virus = Virus.from_string(
        config.get('virus', 'name'),
    )

    fig, (ax1, ax2, ax3) = pl.subplots(
        3,
        sharex=True,
        figsize=(8, 10)
    )

    ax1.set_title('COVID19 simulation', fontsize=20)

    p1 = ax1.plot(
        data.days,
        data.susceptible,
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

    if logscale:
        ax1.set_yscale('log')

    ax1.grid(True)

    p4 = ax2.plot(
        data.days,
        data.hospitalized,
        'tab:red',
        linewidth=2
    )

    p5 = ax2.plot(
        data.days,
        data.critical_care,
        'tab:blue',
        linewidth=2
    )

    ax2.legend((p4[0], p5[0]), ('Hospitalized', 'Critical Care'))

    if logscale:
        ax2.set_yscale('log')

    ax2.grid(True)

    infected = np.array(data.infected)

    p6 = ax3.plot(
        data.days,
        infected * virus.asymptomatic_ratio,
        'tab:blue',
        linewidth=2
    )

    p7 = ax3.plot(
        data.days,
        infected * virus.mild_symptoms_ratio,
        'tab:green',
        linewidth=2
    )

    p8 = ax3.plot(
        data.days,
        data.new_cases,
        'tab:red',
        linewidth=2
    )

    p9 = ax3.plot(
        data.days,
        data.dead,
        'k',
        linewidth=2
    )

    ax3.legend(
        (p6[0], p7[0], p8[0], p9[0]),
        ('Asymptomatic', 'Mild symptoms', 'New cases', 'Dead cumulative')
    )
    ax3.set_xlabel('Days', fontsize=16)

    if logscale:
        ax3.set_yscale('log')

    ax3.grid(True)

    pl.tight_layout()

    if filepath is None:
        pl.show()

    else:
        ensure_dir('/'.join(filepath.split('/')[:-1]))

        pl.savefig(filepath)
        pl.close()

        logging.info(f'Result plot saved to: {filepath}')
