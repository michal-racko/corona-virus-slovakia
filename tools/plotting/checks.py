import numpy as np
import matplotlib.pyplot as pl


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
