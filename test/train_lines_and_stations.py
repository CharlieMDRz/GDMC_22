# -*- coding: utf-8 -*-
"""
Created on Mon May 16 15:25:42 2022

@author: CharlieMDRz
"""
from typing import Set

import cv2

from utils import *
from terrain import TerrainMaps
terrain = TerrainMaps.request()

# %%
from gdpc.toolbox import visualizeHeightmap
visualizeHeightmap(terrain.height_map)

# %%
from settlement import *
settlement = Settlement(terrain)
settlement.districts.build(terrain, n_clusters=16)
districts = settlement.districts.district_centers

# %% Plot lines
from matplotlib import pyplot as plt
def plot_height_map():
    fig, ax = plt.subplots()
    def normalize(array):
        """**Normalize the array to contain values from 0 to 1**."""
        return (array - array.min()) / (array.max() - array.min())

    array = terrain.height_map
    array = (normalize(array) * 255).astype(np.uint8)
    plt_image = cv2.cvtColor(array, cv2.COLOR_BGR2RGB)
    ax.imshow(plt_image)
    return ax
# %%
def compute_train_line(stations: Set[Position]):
    network = []
    # Compute distance array
    stations_list: List[Position] = list(stations)
    n_stations = len(stations_list)
    distance_array = np.zeros((n_stations, n_stations))
    for i, station_i in enumerate(stations_list[:-1]):
        for j, stations_j in enumerate(stations_list):
            if j > i:
                distance_array[i, j] = distance_array[j, i] = manhattan(station_i, stations_j)

    stations_to_connect = set(range(1, n_stations))
    connected_stations = {0}

    network = []
    while stations_to_connect:
        # compute distances
        neighbour = {i: argmin(connected_stations, key=(lambda j: distance_array[i, j])) for i in stations_to_connect}
        distances = {i: distance_array[i, neighbour[i]] for i in neighbour}
        # find best candidate
        i = argmin(distances.keys(), distances.__getitem__)
        j = neighbour[i]
        # update values
        stations_to_connect.remove(i)
        connected_stations.add(i)
        network.append((stations_list[i], stations_list[j]))
    
    return min_spanning_tree(stations)
    #return network

rail_sections = compute_train_line(districts)
ax = plot_height_map()
ax.scatter([p.x for p in districts], [p.z for p in districts])
for s1, s2 in rail_sections:
    ax.plot([s1.x, s2.x], [s1.z, s2.z])
plt.show()

# %%
degree = {s: sum(s in sec for sec in rail_sections) for s in districts}
rail_lines = []
network: Graph = Graph(False)
for section in rail_sections:
    network.addEdge(*section)

ax = plot_height_map()
ax.scatter([p.x for p in districts], [p.z for p in districts])
for station in network.nodes:
    station_dir_vec = sum(abs(station - neighbour) / euclidean(station, neighbour) for neighbour in network.getNeighbours(station))
    station_dir = Direction.of(*station_dir_vec.coords)
    ax.text(station.x, station.z, '---' if station_dir==Direction.East else '|', ha='center', va='center')
plt.show()


def graph_deg(graph: Graph, node):
    return len(graph.getNeighbours(node))

rail_lines = {n: [n] for n in network.nodes if graph_deg(network, n) == 1}
while True:
    try:
        n0 = next(n for n in rail_lines.keys() if graph_deg(network, rail_lines[n][-1]) == 1)
        n1 = rail_lines[n0][-1]
        n2 = network.getNeighbours(n1).difference(rail_lines[n0]).pop()
        rail_lines[n0].append(n2)
        network.removeEdge(n1, n2)
        
        ax = plot_height_map()
        ax.scatter([p.x for p in districts], [p.z for p in districts])
        for line in rail_lines.values():
            ax.plot([s.x for s in line], [s.z for s in line])
        plt.show()

    except StopIteration:
        break
