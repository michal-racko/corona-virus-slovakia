import numpy as np
import matplotlib.pyplot as pl

from matplotlib.patches import Rectangle


def household_age_distribution(ages, household_sizes, filepath):
    histogram, _, _, _ = pl.hist2d(
        ages,
        household_sizes,
        range=((0, 90), (0, 10))
    )

    histogram = histogram.T

    hist_percentage = histogram / histogram.sum(axis=0)
    hist_percentage[~np.isfinite(hist_percentage)] = 0

    for i, _ in enumerate(hist_percentage):
        for j, _ in enumerate(hist_percentage[0]):
            pl.text(
                (j + 0.5) * 9, i + 0.5,
                f'{hist_percentage[i][j] * 100:.1f}',
                horizontalalignment='center',
            )

    pl.title('Simulated households', fontsize=20)
    pl.ylabel('Household size', fontsize=16)
    pl.xlabel('Age', fontsize=16)

    pl.savefig(filepath)
    pl.close()


def age_distribution(ages, filepath):
    ages, counts = np.unique(ages, return_counts=True)

    pl.bar(
        ages,
        counts,
        width=10,
        color='tab:blue'
    )

    pl.title('Simulated ages', fontsize=20)
    pl.xlabel('Age', fontsize=16)

    pl.savefig(filepath)
    pl.close()


def time_ranges(infectious, healing, hospitalization, critical_care, filepath):
    colors = [
        'tab:blue',
        'tab:green',
        'tab:orange',
        'tab:red'
    ]

    labels = [
        'infectiousness start',
        'fully recovered',
        'hospitalization start',
        'critical care start',
    ]

    pl.hist(
        [
            infectious,
            healing,
            hospitalization,
            critical_care,
        ],
        color=colors,
        histtype='stepfilled',
        density=True,
        alpha=0.7,
        bins=50,
        range=(0, 25)
    )

    handles = [Rectangle((0, 0), 1, 1, color=c, ec="k") for c in colors]
    pl.legend(handles, labels)
    pl.xlabel('Days since infection', fontsize=16)

    pl.savefig(filepath)
    pl.close()


def interaction_multiplicities(stochastic_interactions, filepath, log_scale=False):
    pl.hist(
        stochastic_interactions,
        bins=len(np.unique(stochastic_interactions)),
        log=log_scale
    )

    pl.title('Simulated meeting patterns', fontsize=20)
    pl.xlabel('Interaction multiplicity', fontsize=16)

    pl.savefig(filepath)
    pl.close()
