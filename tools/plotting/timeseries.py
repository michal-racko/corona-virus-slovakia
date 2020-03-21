import logging

import matplotlib.pyplot as pl

from tools.general import ensure_dir
from tools.data_structure import TimeSeriesResult


def plot_pandemic(data: TimeSeriesResult, filepath=None):
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
        ('Unafected', 'Infected', 'Immune')
    )

    p4 = ax2.plot(
        data.days,
        data.new_cases,
        'tab:red',
        linewidth=2
    )

    ax2.legend((p4[0],), ('New cases',))

    p5 = ax3.plot(
        data.days,
        data.dead,
        'k',
        linewidth=2
    )

    ax3.legend((p5[0],), ('Dead',))
    ax3.set_xlabel('Days', fontsize=16)

    pl.tight_layout()

    if filepath is None:
        pl.show()

    else:
        ensure_dir('/'.join(filepath.split('/')[:-1]))

        pl.savefig(filepath)
        pl.close()

        logging.info(f'Result plot saved to: {filepath}')



