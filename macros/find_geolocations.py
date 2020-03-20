import time

import pandas as pd

from geopy.geocoders import Nominatim

CSV_DATA = '../data/slovak_counties.csv'

if __name__ == '__main__':
    df = pd.read_csv(CSV_DATA)

    geolocator = Nominatim()

    longitudes = []
    latitudes = []

    for i, row in enumerate(df.iterrows()):
        d = row[1].to_dict()

        while True:
            try:
                location = geolocator.geocode(
                    f'{d["region"]} {d["county"]}'
                )

                break

            except:
                time.sleep(0.1)

        try:
            longitudes.append(location.longitude)
            latitudes.append(location.latitude)

            print(
                f'Found: {d["county"]} {location.longitude} {location.latitude}'
            )

        except:
            longitudes.append(None)
            latitudes.append(None)

    df['longitudes'] = longitudes
    df['latitudes'] = latitudes

    df.to_csv(CSV_DATA, index=False)
