import logging
import time
from typing import List, Iterable

import numba
import numpy

from utils import Position, BuildArea
from utils.algorithms.fast_astar import abs_distance, in_limits, MAX_INT, \
    _path_to_dest, _heuristic as manhattan
from utils.misc_objects_functions import index_argmin, numba_list, log_exec_time

GAMMA = 4


def hierarchical_astar(source: Position, target: Position, cost_function, get_all_paths=False) -> List[Position]:
    t0 = time.time()
    tuple_paths = tuple_hierarchical_astar(
        source.xz,
        target.xz,
        (BuildArea().width, BuildArea().length),
        lambda p, q: cost_function(Position(*p), Position(*q))
    )
    log_exec_time(t0, f"hierarchical A* from {source} to {target}")

    paths = [[Position(*r) for r in path] for path in tuple_paths]
    if get_all_paths:
        return paths
    else:
        return paths[-1]


def tuple_hierarchical_astar(source, target, dimensions, cost_function):
    """
    Custom A* algorithm - computes path with decreasing steps
    :param source: source point (x, z)
    :param target: target point (x, z)
    :param dimensions: (width, length)
    :param cost_function: cost function ((xo, zo), (xd, zd)) -> value
    """

    intermediate_paths = []
    # Compute initial step
    step = 1
    d = abs_distance(source, target)
    while step * GAMMA < d:
        step *= GAMMA
    step = min(step, GAMMA ** 2)

    def get_cumsum():
        """
        Computes target heuristic for each point in the rough path
        :return:
        """
        l = [0]
        for i in range(len(rough_path) - 1, 0, -1):
            l.append(l[-1] + cost_function(rough_path[i], rough_path[i - 1]))
        res = numba_list(reversed(l))
        return res

    rough_path = [source, target]

    while True:
        distance_map, neighbours, predecessor_map, heuristic_map = astar_env = _init(source, dimensions)
        cumsum = get_cumsum()
        clst_neighbour = source
        heuristic_idx = {}

        while neighbours and (min(distance_map[n] for n in neighbours) < MAX_INT) and abs_distance(clst_neighbour, target) >= step:

            # pick new exploration point -> point closer to target
            clst_neighbour = _closest_neighbor(astar_env, numba_list(rough_path), cumsum)
            neighbours.remove(clst_neighbour)
            neighbours = list(filter(lambda n: heuristic_idx[n] >= heuristic_idx[clst_neighbour], neighbours))
            astar_env = distance_map, neighbours, predecessor_map, heuristic_map
            known_neighbours = len(neighbours)

            # explore neighbours to this point
            _update_distances(astar_env + (cost_function,), dimensions + (step,), clst_neighbour)

            # handle new neighbours: store their heuristic index
            for new_neighbour in neighbours[known_neighbours:]:
                heuristic_idx[new_neighbour] = _heuristic_index(new_neighbour, numba_list(rough_path))

        intermediate_paths.append(_path_to_dest(predecessor_map, source, clst_neighbour, False))

        if step == 1:
            if abs_distance(clst_neighbour, target) >= step:
                return [[]]
            return intermediate_paths
        else:
            rough_path = intermediate_paths[-1]
            if rough_path[-1] != target:
                rough_path.append(target)
            step //= GAMMA


@numba.njit()
def _heuristic_index(point, path):
    distance_to_path = [manhattan(point, point2) for point2 in path]
    return index_argmin(distance_to_path)


@numba.njit()
def _heuristic(point, path, path_heuristic):
    i = _heuristic_index(point, path)
    if i == len(path) - 1:
        return manhattan(point, path[-1])
    target = path[i + 1]
    return manhattan(point, target) + path_heuristic[i+1]


@numba.njit()
def _closest_neighbor(env, path, path_heuristic):
    distance_map, neighbors = env[:2]
    heuristic_map = env[3]
    closest_neighbors = numba.typed.List()
    closest_neighbors.append((1 << 8, 1 << 8))
    min_heuristic = MAX_INT
    for neighbor in neighbors:
        x, z = neighbor
        if heuristic_map[neighbor] >= MAX_INT:
            heuristic_map[neighbor] = _heuristic(neighbor, path, path_heuristic)
        current_heuristic = distance_map[x, z] + heuristic_map[neighbor]
        if min_heuristic == MAX_INT or current_heuristic < min_heuristic:
            closest_neighbors = numba.typed.List()
            closest_neighbors.append(neighbor)
            min_heuristic = current_heuristic
        elif current_heuristic == min_heuristic:
            closest_neighbors.append(neighbor)
    return closest_neighbors[numpy.random.randint(len(closest_neighbors))]


@numba.jit(forceobj=True, parallel=True)
def _update_distances(env, dims, point):
    """
    :param env: (...)
    :param dims: (width, length, step)
    :param point: (x, z)
    """
    x, z = point  # type: int, int
    for xz in _exploration_neighbourhood(x, z, *dims):
        _update_distance(env, point, xz)


@numba.njit(cache=True)
def _exploration_neighbourhood(x, z, width, length, step):
    neighbourhood = set()
    for dx, dz in [(0, step), (step, step), (step, 0), (step, -step), (0, -step), (-step, -step), (-step, 0), (-step, step)]:
        x0, z0 = x + dx, z + dz
        if in_limits((x0, 0, z0), width, length):
            neighbourhood.add((x0, z0))
    return neighbourhood


@numba.njit
def _init(point, dims):
    x, z = point
    _distance_map = numpy.full(dims, MAX_INT)
    _distance_map[x, z] = 0
    _neighbours = [point]
    _predecessor_map = numpy.full((*dims, 2), max(dims))
    _heuristic_map = numpy.full(dims, MAX_INT)
    return _distance_map, _neighbours, _predecessor_map, _heuristic_map


@numba.jit(forceobj=True)
def _update_distance(env, updated_point, neighbor):
    distance_map, neighbors, predecessor_map, h_map, cost = env
    edge_cost = cost(updated_point, neighbor)
    if edge_cost == MAX_INT:
        return

    new_distance = distance_map[updated_point] + edge_cost
    previous_distance = distance_map[neighbor]
    if previous_distance >= MAX_INT:
        neighbors.append(neighbor)
    if previous_distance > new_distance:
        distance_map[neighbor] = new_distance
        predecessor_map[neighbor] = updated_point
