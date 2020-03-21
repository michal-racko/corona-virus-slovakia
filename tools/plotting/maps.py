import json
import logging

import numpy as np
import matplotlib.pyplot as pl

from abc import ABCMeta
from scipy import spatial
from matplotlib.path import Path

from tools.general import ensure_dir


class Map:
    __metaclass__ = ABCMeta

    def __init__(self):
        self.max_longitude = None
        self.min_longitude = None

        self.max_latitude = None
        self.min_latitude = None

        self.image_width = None
        self.image_height = None

        self.map_image = None
        self.borders = None

        self.lon_contours = None
        self.lat_contours = None

    def _transform_xy(self,
                      longitude: np.ndarray,
                      latitude: np.ndarray) -> tuple:
        """
        Transforms longitude and latitude into
        positions in the image
        """
        x = self.image_width / (self.max_longitude - self.min_longitude) * (longitude - self.min_longitude)
        y = self.image_height / (self.max_latitude - self.min_latitude) * (self.max_latitude - latitude)

        return x, y

    def _xy_to_coords(self,
                      x: np.ndarray,
                      y: np.ndarray):
        """
        Transforms positions in the image into
        longitude and latitude

        """
        longitude = x * (self.max_longitude - self.min_longitude) / self.image_width + self.min_longitude
        latitude = self.max_latitude - y * (self.max_latitude - self.min_latitude) / self.image_height

        return longitude, latitude

    def _get_grid(self,
                  longitudes: np.ndarray,
                  latitudes: np.ndarray,
                  values: np.ndarray,
                  precision=500):
        longitude_space = np.linspace(
            self.min_longitude,
            self.max_longitude,
            precision
        )

        latitude_space = np.linspace(
            self.min_latitude,
            self.max_latitude,
            precision
        )

        X, Y = np.meshgrid(longitude_space, latitude_space)

        x = X.ravel()
        y = Y.ravel()

        xy_self = np.array([longitudes, latitudes]).T
        xy_other = np.array([x, y]).T

        tree = spatial.cKDTree(xy_self)
        mindist, minid = tree.query(xy_other)

        Z = np.reshape(values[minid], (-1, precision))

        return X, Y, Z

    def plot_grid(self,
                  longitudes: np.ndarray,
                  latitudes: np.ndarray,
                  values: np.ndarray,
                  filepath=None,
                  cmap='gnuplot',
                  alpha=0.7,
                  hide_ticks=True,
                  show_coords=True):
        """
        Makes a 2D contour plot
        """
        fig = pl.figure(figsize=(15, 9))
        ax1 = fig.add_subplot(1, 1, 1)

        ax1.imshow(self.map_image)

        X, Y, Z = self._get_grid(
            longitudes,
            latitudes,
            values
        )

        X, Y = self._transform_xy(X, Y)

        if self.borders is not None:
            xc, yc = self.borders[0], self.borders[1]

            pth = Path(np.vstack((xc, yc)).T, closed=False)

            mask = pth.contains_points(np.vstack((X.ravel(), Y.ravel())).T)
            mask = mask.reshape(X.shape)

            X = np.ma.masked_array(X, ~mask)
            Y = np.ma.masked_array(Y, ~mask)
            Z = np.ma.masked_array(Z, ~mask)

        ax1.contourf(X, Y, Z, alpha=alpha, cmap=cmap)

        if hide_ticks:
            pl.xticks([])
            pl.yticks([])

        if show_coords:
            x_lin = np.linspace(0, self.image_width, 10)
            y_lin = np.linspace(0, self.image_height, 10)

            lon_lin, lat_lin = self._xy_to_coords(x_lin, y_lin)

            pl.xticks(x_lin, np.around(lon_lin, 2))
            pl.yticks(y_lin, np.around(lat_lin, 2))

        pl.tight_layout()

        if filepath is None:
            pl.show()

        else:
            ensure_dir('/'.join(filepath.split('/')[:-1]))

            pl.savefig(filepath)
            pl.close()

            logging.info(f'Result plot saved to: {filepath}')


class Slovakia(Map):
    """
    Map of Slovakia
    """

    def __init__(self,
                 contour_file='data/maps/slovakia_contours.json',
                 map_image='data/maps/Slovakia.jpg',
                 borders='data/maps/Slovakia_borders.npy'):
        super().__init__()

        with open(contour_file) as f:
            contours = np.array(
                json.load(f)
            ).T

        self.lon_contours = contours[0]
        self.lat_contours = contours[1]

        self.min_latitude = 47.717753
        self.max_latitude = 49.627242

        self.min_longitude = 16.81299
        self.max_longitude = 22.57976

        self.image_width = 4514
        self.image_height = 2270

        self.map_image = pl.imread(map_image)
        self.borders = np.load(borders)
