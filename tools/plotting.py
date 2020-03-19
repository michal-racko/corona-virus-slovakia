import logging

import matplotlib.pyplot as pl

from tools.general import ensure_dir
from tools.simulation_result import SimulationResult


def plot_pandemic(data: SimulationResult, filepath=None):
    pl.subplot(211)

    p1 = pl.plot(
        data.days,
        data.unaffected,
        'tab:blue',
        linewidth=2
    )

    p2 = pl.plot(
        data.days,
        data.infected,
        'tab:red',
        linewidth=2
    )

    p3 = pl.plot(
        data.days,
        data.immune,
        'tab:green',
        linewidth=2
    )

    pl.legend(
        (p1[0], p2[0], p3[0]),
        ('Unafected', 'Infected', 'Immune')
    )

    pl.subplot(212)

    p4 = pl.plot(
        data.days,
        data.dead,
        'k',
        linewidth=2
    )

    pl.legend((p4[0],), ('Dead',))

    pl.tight_layout()

    if filepath is None:
        pl.show()

    else:
        ensure_dir('/'.join(filepath.split('/')[:-1]))

        pl.savefig(filepath)
        pl.close()

        logging.info(f'Result plot saved to: {filepath}')
