import logging
from typing import Iterable, List, Dict, Set

from matplotlib import pyplot as plt
import numpy as np
import sklearn
from sklearn.cluster import KMeans

from parameters import MAX_DISTRICT_COUNT_OPTIONS
from terrain import TerrainMaps
from utils import argmax, Position

__all__ = ['build_default_district', 'build_kmeans_district', 'DistrictCluster']

N_CLUSTERS = 'n_clusters'


class DistrictCluster:

    def __init__(self, id):
        self.id = id
        self.reps: Set[Position] = set()
        self.score: float = 0.
        self.isTown: bool = False
        self.center: Position = Position(0, 0)
        self.size: int = 0


def build_default_district():
    pass


def get_possible_cluster_count(maps, **kwargs) -> List[int]:
    if kwargs.get(N_CLUSTERS, 0):
        return [kwargs.get(N_CLUSTERS)]

    approx = np.sqrt(maps.width * maps.length) // 50
    min_clusters = int(max(2, approx // 2))
    max_clusters = int(max(min_clusters, approx * 1.3))
    logging.info(f"there'll be between {min_clusters} and {max_clusters} districts")

    if max_clusters - min_clusters < MAX_DISTRICT_COUNT_OPTIONS:
        return list(range(min_clusters, max_clusters + 1))
    else:
        # logarithmic exploration
        gamma = (max_clusters / min_clusters) ** (1 / MAX_DISTRICT_COUNT_OPTIONS)
        return list({int(round(min_clusters * (gamma ** k))) for k in range(MAX_DISTRICT_COUNT_OPTIONS+1)})


def get_kmeans_samples(interest):
    """
    Builds a dataset to perform cluster analysis in order to find suitable positions to build villages
    """
    from building_seeding.interest.interest import InterestMap
    from building_seeding import BuildingType

    DOWN_SIZE = 4
    score_matrix: np.ndarray = interest[::DOWN_SIZE, ::DOWN_SIZE]  # downsized interest matrix

    n_samples: int = min(1000, score_matrix.size)  # target number of samples
    downsized_keep_rate = n_samples / score_matrix.size  # resulting portion of positions taken into account
    threshold_score = np.quantile(score_matrix, 1 - downsized_keep_rate)  # min score of the top #n_samples scores
    top_score_xz = np.where(score_matrix >= threshold_score)
    samples = [(x * DOWN_SIZE, z * DOWN_SIZE) for x, z in zip(*top_score_xz)]

    X = np.array(samples)
    print(f"{X.shape[0]} samples to select districts")
    return X


def select_best_model(X, n_clusters: List[int], visualize=False) -> KMeans:
    scores = []
    models = []
    for k in n_clusters:
        models.append(KMeans(n_clusters=k).fit(X))
        scores.append(sklearn.metrics.silhouette_score(X, models[-1].labels_))
        logging.info(f"Silhouette score for {k} clusters: {scores[-1]}")

    if visualize:
        plt.plot(n_clusters, scores)
        plt.title("Silhouette score as a function of n_clusters")
        plt.show()

    return models[argmax(scores)]


def build_district_clusters(model, X, interest) -> Dict[int, DistrictCluster]:
    n_positions = (interest > 0).sum()
    n_samples = X.shape[0]
    clusters = {}

    for label in set(model.labels_):
        cluster_X = X[model.labels_ == label]
        weights = [interest[tuple(cluster_X[i])] for i in range(cluster_X.shape[0])]
        cluster_x = cluster_X[:, 0].dot(weights) / sum(weights)
        cluster_z = cluster_X[:, 1].dot(weights) / sum(weights)

        district_cluster = DistrictCluster(label)
        district_cluster.center = Position(cluster_x, cluster_z)
        district_cluster.reps = {Position(*cluster_X[_, :2]) for _ in range(cluster_X.shape[0])}
        district_cluster.score = 1
        district_cluster.size = cluster_X.shape[0] * (n_positions / n_samples)

        clusters[label] = district_cluster

    return clusters


def build_kmeans_district(maps: TerrainMaps, **kwargs):
    from building_seeding.interest.interest import InterestMap
    from building_seeding import BuildingType
    interest: np.ndarray = InterestMap(BuildingType.house, "Flat_scenario", maps, None).terrain_interest
    X = get_kmeans_samples(interest)
    model = select_best_model(X, get_possible_cluster_count(maps, **kwargs))
    clusters = build_district_clusters(model, X, interest)

    return interest, X, model, clusters


def visualize_k_means():
    Xu = scaler.inverse_transform(X)
    x, y = Xu[:, 0], -Xu[:, 1]
    color = ["#" + ''.join([random.choice("ABCDEF0123456789") for j in range(6)]) for i in
             range(len(kmeans.cluster_centers_))]
    c = [color[cluster] for cluster in kmeans.labels_]
    plt.scatter(x, y, c=c)
    xc = scaler.inverse_transform(kmeans.cluster_centers_)[:, 0]
    yc = -scaler.inverse_transform(kmeans.cluster_centers_)[:, 1]
    plt.scatter(xc, yc, s=100, c='k', marker='+')
    plt.title(f"{n_clusters} clusters - scaling factor: {coord_scale}")
    plt.show()