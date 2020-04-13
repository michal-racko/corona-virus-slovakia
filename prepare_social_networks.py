import json

import pandas as pd
import networkx as nx

OUTPUT_DIR = 'data/social/generated'
OUTPUT_FILE = 'watts_strogatz_triple_big.json'
POPULATIONS_FILE = 'data/munic_pop.xlsx'
TOWN_LOCATIONS_FILE = 'data/obce1.xlsx'

MEAN_CONTACTS = 30
REWIRING = 0.7

if __name__ == '__main__':
    population_df = pd.read_excel(POPULATIONS_FILE)
    town_location_df = pd.read_excel(TOWN_LOCATIONS_FILE)

    munic_df = population_df.merge(
        town_location_df,
        left_on='munic',
        right_on='IDN4'
    )

    offset = 0

    interactions = {}

    print(munic_df)

    for city_i, (city_population, city_name) in enumerate(zip(munic_df.popul.tolist(), munic_df.NM4.tolist())):
        print(f'Processing city number {city_i} : {city_name}')

        n_groups = int(city_population / MEAN_CONTACTS)

        try:
            H = nx.connected_watts_strogatz_graph(int(0.7 * city_population), 50, 0.2)
            I = nx.connected_watts_strogatz_graph(int(0.2 * city_population), 70, 0.15)
            J = nx.connected_watts_strogatz_graph(int(0.1 * city_population), 100, 0.05)

            G = nx.compose(H, I)
            city_graph = nx.compose(G, J)

            # city_graph = nx.powerlaw_cluster_graph(city_population, 20, 0.05)

        except nx.exception.NetworkXError:
            city_graph = nx.relaxed_caveman_graph(n_groups, MEAN_CONTACTS, REWIRING)

        edges = city_graph.edges()

        start_vertices = [e[0] for e in edges]
        end_vertices = [e[1] for e in edges]

        current_start_vert = -1
        egde_i = 0

        for vertex_start, vertex_end in zip(start_vertices, end_vertices):
            if vertex_start != current_start_vert:
                current_start_vert = vertex_start

                egde_i = 0

            try:
                interactions[str(egde_i)]['vertex_start'].append(vertex_start + offset)
                interactions[str(egde_i)]['vertex_end'].append(vertex_end + offset)

            except KeyError:
                interactions[str(egde_i)] = {
                    'vertex_start': [vertex_start + offset],
                    'vertex_end': [vertex_end + offset],
                }

            egde_i += 1

        offset += city_population

    with open(f'{OUTPUT_DIR}/{OUTPUT_FILE}', 'w') as f:
        json.dump(interactions, f)
