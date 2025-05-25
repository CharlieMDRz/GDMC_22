"""
Village skeleton growth
"""
import itertools
import multiprocessing as mp
import time
from typing import List, Set, Dict, Tuple

import numpy as np

from building_seeding.building_pool import BuildingPool, BuildingType
from building_seeding.district.districts import Districts
from building_seeding.interest import InterestSeeder
from building_seeding.parcel import Parcel, MaskedParcel
from parameters import MIN_PARCEL_SIZE, AVERAGE_PARCEL_SIZE, MAX_PARCELS_IN_BLOCK
from pathfinding import RoadNetwork
from terrain import TerrainMaps, ObstacleMap
from utils import *
from utils.algorithms.graphs import *
from visu.pre_processing import VisuHandler


class VillageSkeleton:

    def __init__(self, scenario: str, maps: TerrainMaps, districts: Districts, parcel_list: List[Parcel]):
        self.scenario = scenario
        self.size = (maps.width, maps.length)
        self.maps = maps
        buildable_surface = maps.width * maps.length - maps.fluid_map.as_obstacle_array.sum()
        self.building_iterator = BuildingPool(districts.buildable_surface)
        self.__parcel_list = parcel_list
        self.parcel_size = MIN_PARCEL_SIZE

        # parcel_list.append(Parcel(ghost_position, BuildingType.from_name('ghost'), maps))
        self.__interest = InterestSeeder(maps, districts, parcel_list, scenario)

    def add_parcel(self, seed: Point or Parcel, building_type: BuildingType = None):
        if isinstance(seed, Point):
            if building_type in [BuildingType.crop, BuildingType.house]:
                new_parcel = MaskedParcel(seed, building_type, self.maps)
            else:
                new_parcel = Parcel(seed, building_type, self.maps)
        elif isinstance(seed, Parcel):
            new_parcel = seed
        else:
            raise TypeError("Expected Point or Parcel, found {}".format(seed.__class__))
        self.__parcel_list.append(new_parcel)
        ObstacleMap().add_obstacle(*new_parcel.obstacle(AVERAGE_PARCEL_SIZE // 3))

    def remove_parcel(self, parcel: Parcel):
        # todo: notify interest of parcel deletion (side effect)
        self.__parcel_list.remove(parcel)
        ObstacleMap().hide_obstacle(*parcel.obstacle(), True)

    def __handle_new_road_cycles(self, cycles):
        for road_cycle in map(cycle_preprocess, cycles):
            city_block = CityBlock(road_cycle, self.maps)
            max_block_surface = MAX_PARCELS_IN_BLOCK * (AVERAGE_PARCEL_SIZE ** 2)

            if city_block.surface > max_block_surface:
                continue  # block to big, must subdivide with roads before parcels

            else:
                old_parcels = list(filter(lambda p: p.center in city_block, self.__parcel_list))

                if old_parcels:
                    block_type = [p.building_type for p in old_parcels][-1]
                    for parcel in old_parcels:
                        self.remove_parcel(parcel)
                else:
                    block_type = self.__interest.get_optimal_type(city_block.center)

                if block_type == BuildingType.windmill:
                    block_type = BuildingType.crop

                for new_parcel in city_block.parcels(block_type):
                    self.add_parcel(new_parcel)

    def grow(self, time_limit: int, do_visu: bool):
        print("Seeding parcels")
        if time_limit:
            p = mp.Process(target=time.sleep, args=(time_limit,))
            p.start()
        else:
            p = None

        map_plots = VisuHandler(do_visu, self.maps, self.__parcel_list)
        build_iter = self.building_iterator

        t0 = time.time()
        for building_type in build_iter:
            if p is not None and not p.is_alive():
                print(f"Reached time limit of {time_limit}s, timing-out generation !")
                break

            print(f"\nTrying to place {building_type.name} - #{build_iter.count} out of {build_iter.size}")

            # Village Element Seeding Process
            # building_type = self.__interest.try_to_reuse_existing_parcel(building_type)
            building_position = self.__interest.get_seed(building_type)

            if building_position is None:
                print("No suitable position found")
                continue

            print("Placed at x:{}, z:{}".format(building_position.x, building_position.z))

            # Road Creation Process
            cycles = self.maps.road_network.connect_to_network(building_position, margin=AVERAGE_PARCEL_SIZE/2)
            self.add_parcel(building_position, building_type)
            map_plots.handle_new_parcel(self.__interest[building_type])  # does nothing if not do_visu
            if cycles:
                self.__handle_new_road_cycles(cycles)
            if time_limit and time.time() - t0 >= time_limit:
                print("Time limit reached: early stopping parcel seeding")
                break

        if p is not None:
            p.terminate()


class CityBlock(Bounds):
    """
    Large parcel surrounded by roads. Handles detection of this parcel (based on a road cycle) and optional subdivision
    """

    def __init__(self, road_cycle, maps):
        self.__road_points = road_cycle
        self.__maps = maps
        max_block_surface = MAX_PARCELS_IN_BLOCK * (AVERAGE_PARCEL_SIZE ** 2)
        block_points = connected_component(GridGraph(False), mean(road_cycle).asPosition, self.connection, max_block_surface)
        p, m = point_set_as_array(block_points)
        self.__origin, self.__mask = p.view(Position), m.view(PointArray)
        super().__init__(self.__origin, Point(self.__mask.width, self.__mask.length))

    @staticmethod
    def connection(src_point: Point, dst_point: Point) -> bool:
        net: RoadNetwork = RoadNetwork()
        return net.get_distance(dst_point) > 0

    def __subdivide(self, pos: Position, mask: PointArray, ndiv: int) -> List[Tuple[Position, np.ndarray]]:
        if ndiv == 1:
            return [(pos, mask)]

        n1, n2 = ndiv // 2, ndiv - ndiv // 2
        if n1 != n2 and bernouilli(): n1, n2 = n2, n1
        optimal_score = n1 / ndiv
        if mask.width > mask.length:
            # cut along x -> 1st index
            scores = [mask[cut:, :].sum() / mask.sum() for cut in range(mask.width)]
            cut = argmin([abs(optimal_score - score) for score in scores])
            mask1, mask2 = mask[:cut, :], mask[cut:, :]
            orig1, orig2 = pos, pos + Point(cut, 0)
        else:
            # cut along z -> 2nd index
            scores = [mask[:, cut:].sum() / mask.sum() for cut in range(mask.length)]
            cut = argmin([abs(optimal_score - score) for score in scores])
            mask1, mask2 = mask[:, :cut], mask[:, cut:]
            orig1, orig2 = pos, pos + Point(0, cut)

        return self.__subdivide(orig1, mask1, n1) + self.__subdivide(orig2, mask2, n2)

    def parcels(self, btype: BuildingType) -> List[Parcel]:
        parcel_count = 1 + self.surface // (AVERAGE_PARCEL_SIZE ** 2)
        new_parcels = []
        for parcel_pos, parcel_mask in self.__subdivide(self.__origin, self.__mask, parcel_count):
            new_parcels.append(MaskedParcel(parcel_pos, btype, mc_map=self.__maps, mask=parcel_mask))
        return new_parcels

    @property
    def surface(self):
        return int(self.__mask.sum())

    @property
    def center(self):
        return self.__origin + Point(self.width // 2, self.length // 2)

    def __contains__(self, item: Point):
        return self.minx <= item.x < self.maxx and self.minz <= item.z < self.maxz and self.__mask[item - self.__origin]


def cycle_preprocess(road_points: Set[Point]) -> Set[Point]:
    # Build graph associated to the road paths
    graph: Dict[Point, Set[Point]] = {node: set() for node in road_points}  # graph as adjacency sets
    for node in road_points:
        for dx, dz in filter(any, itertools.product(range(-1, 2), range(-1, 2))):
            neighbour = node + Point(dx, dz)
            if neighbour in road_points:
                graph[node].add(neighbour)

    # Remove nodes not belonging to cycles
    def degree(_node: Point) -> int:
        return len(graph[node])

    extremities = {node for node in graph if degree(node) < 2}
    while extremities:
        node = extremities.pop()
        for neighbour in graph[node]:
            graph[neighbour].remove(node)
            if degree(neighbour) < 2:
                extremities.add(neighbour)

    # Find the larger cycle
    cycles = []
    road_points_copy = road_points.copy()
    while road_points_copy:
        to_explore = {road_points_copy.pop()}
        cycle = set()
        while to_explore:
            node1 = to_explore.pop()
            cycle.add(node1)
            to_explore.update(graph[node1].difference(cycle))
        cycles.append(cycle)

    from utils.misc_objects_functions import argmax
    return argmax(cycles, len)
