from tools.plotting.maps import Slovakia
from tools.data_structure import GeographicalResult


def plot_final_mortality(data: GeographicalResult):
    longitudes, latitudes, values = data.get_mortalities(asratio=True)

    plotting_map = Slovakia()
    plotting_map.plot_grid(
        longitudes,
        latitudes,
        values
    )
