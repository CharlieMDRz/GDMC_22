# -*- coding: utf-8 -*-
"""
Éditeur de Spyder

Ceci est un script temporaire.
"""
import random

import cv2
from gdpc.toolbox import visualizeHeightmap
import numpy as np
from scipy import ndimage

from terrain import TerrainMaps
from building_seeding import BuildingType, BUILDING_ENCYCLOPEDIA

def flip_to_mc_view(a):
    return np.fliplr(a.T).T

#%% Import terrain
terrain = TerrainMaps.request()

#%% Temperature map
from utils.geometry_utils import BuildArea
def get_temperature(p):
    return terrain.biome.temperature(p)

temperature_list = [terrain.biome.temperature(p) for p in BuildArea.building_positions()]
temperature_array = np.array(temperature_list).reshape(terrain.width, terrain.length)
visualizeHeightmap(flip_to_mc_view(temperature_array), title="Temperature")

#%% Height and steepness
"""
Steepness map rework
"""
visualizeHeightmap(flip_to_mc_view(terrain.height_map), title="Height")
steepness_list = [terrain.height_map.steepness(p) for p in BuildArea.building_positions()]
steepness_array = np.array(steepness_list).reshape(terrain.width, terrain.length)
visualizeHeightmap(flip_to_mc_view(steepness_array), title="Steepness")

# %% decomposed steepness computation
height_map = flip_to_mc_view(terrain.height_map).astype(np.uint8)
x_dev = cv2.Scharr(height_map, 5, 1, 0)
z_dev = cv2.Scharr(height_map, 5, 0, 1)

h_dev = (x_dev ** 2 + z_dev ** 2) ** .5

visualizeHeightmap(h_dev, title="base terrain steepness")

# %%
#h_dev2 = h_dev ** .5
h_dev2 = cv2.GaussianBlur(h_dev, (15, 15), 0)**.5
#h_dev2 = cv2.blur(h_dev**.5, (11, 11))
visualizeHeightmap(h_dev2, title="modified terrain steepness")
visualizeHeightmap((h_dev2 > 4).astype(int), title="steep areas")


#%% water detection
"""
Water map rework
"""
import cv2
water_map = (terrain.fluid_map.water > 0).astype(np.uint8)
water_sources = cv2.connectedComponents(water_map, connectivity=8)

water_sources_size = {label: (water_sources[1] == label).sum() for label in range(water_sources[0])}

# %%
new_water_map = water_sources[1][:]

for label, size in water_sources_size.items():
    if size < 32:
        new_water_map[new_water_map == label] = 0
        
visualizeHeightmap(new_water_map, title="water map v2")


#%% Build interest
"""
Interest
"""
from building_seeding.interest.interest import InterestMap
house_interest = InterestMap(BuildingType.house, "Flat_scenario", terrain, None)

terrain_interest = house_interest.terrain_interest

do_flip_array = True
if do_flip_array:
    terrain_interest = flip_to_mc_view(terrain_interest)
terrain_interest[terrain_interest < 0] = 0
visualizeHeightmap(terrain_interest, title="Default interest")

# %%
def get_interest_array_for_weights(terrain, weights):
    scenario = "test"
    import copy
    BUILDING_ENCYCLOPEDIA[scenario] = copy.deepcopy(BUILDING_ENCYCLOPEDIA['Flat_scenario'])
    BUILDING_ENCYCLOPEDIA[scenario]["Weighting_factors"] = {'house': [0, 0, 0] + weights}
    interest = InterestMap(BuildingType.house, scenario, terrain, None)
    return interest.terrain_interest

# %% Plots interest map with random weights
for _ in range(1):
    w = np.random.random(7) #altitude, pure_water, sea_water, lava, steepness
    a = flip_to_mc_view(get_interest_array_for_weights(terrain, list(w)))
    fw = [f"{int(100*k):02d}" for k in w]
    visualizeHeightmap(a, title=f"Interest w weights {fw}")
    
# %%
n_size = 7
interest_array: np.ndarray  # todo: define this arrayin the terminal

values = interest_array.flatten()
values = values[values > -1]
p5 = np.percentile(values, 15)
p95 = np.percentile(values, 85)
extrema = np.zeros(interest_array.shape)
extrema[interest_array <= p5] = -1
extrema[interest_array >= p95] = 1

max_interest_array = ndimage.maximum_filter(interest_array, (n_size, n_size))
local_maxima = ((interest_array == max_interest_array) & (interest_array > -1)).astype(np.uint8)
good_local_maxima = local_maxima & (interest_array >= p95)

visualizeHeightmap(interest_array, local_maxima, good_local_maxima, extrema, title="Max interest locations")

# %%
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
X = np.array(np.where(good_local_maxima)).T
K = 6
viz = True

kmeans = KMeans(n_clusters=K).fit(X)
if viz:
    fig = plt.figure(figsize=(8, 8), dpi=300)
    x, y = X[:, 0], -X[:, 1]
    color = ["#" + ''.join([random.choice("ABCDEF0123456789") for j in range(6)]) for i in range(len(kmeans.cluster_centers_))]
    c = [color[cluster] for cluster in kmeans.labels_]
    fig.add_axes(plt.scatter(-y, -x, c=c))
    plt.show(figsize=(8, 8))

