from tools.plotting.maps import Slovakia
from tools.data_structure import GeographicalResult


def plot_mortality(data: GeographicalResult, day_i=-1):
    longitudes, latitudes, values = data.get_mortalities(day_i=day_i, asratio=True)

    plotting_map = Slovakia()
    plotting_map.plot_grid(
        longitudes,
        latitudes,
        values
    )


def plot_infected(data: GeographicalResult, day_i=-1, filepath=None):
    longitudes, latitudes, values = data.get_infected(day_i=day_i, asratio=False)

    if day_i != -1:
        title = f'Simulation day: {day_i}'

    else:
        title = None

    plotting_map = Slovakia()
    plotting_map.plot_grid(
        longitudes,
        latitudes,
        values,
        cmap='Reds',
        alpha=0.85,
        title=title,
        # values_range=(0, 1000000),
        filepath=filepath
    )
