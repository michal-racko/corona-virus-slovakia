import matplotlib.pyplot as pl

from tools.simulation_result import SimulationResult


def plot_pandemic(data: SimulationResult):
    pl.subplot(211)

    p1 = pl.plot(
        data.days,
        data.unaffected,
        'tab:blue'
    )

    p2 = pl.plot(
        data.days,
        data.infected,
        'tab:red'
    )

    p3 = pl.plot(
        data.days,
        data.immune,
        'tab:green'
    )

    pl.legend(
        (p1[0], p2[0], p3[0]),
        ('Unafected', 'Infected', 'Immune')
    )

    pl.subplot(212)

    p4 = pl.plot(
        data.days,
        data.dead,
        'k'
    )

    pl.legend((p4[0],), ('Dead',))

    pl.tight_layout()
    pl.show()
