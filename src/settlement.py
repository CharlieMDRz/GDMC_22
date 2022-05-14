import itertools
import logging
import traceback
from math import exp
from random import choice
from typing import List, Dict

import numpy as np
from sortedcontainers import SortedList

from building_seeding import Districts, Parcel, VillageSkeleton, BuildingType, MaskedParcel
from generation.generators import place_sign
from parameters import MAX_HEIGHT, BUILDING_HEIGHT_SPREAD, TERRAFORM_ITERATIONS, AVERAGE_PARCEL_SIZE
from path_networks import RoadNetwork
from path_networks.rail_network import compute_train_line
from terrain import TerrainMaps
from utils import *
from utils.algorithms import min_spanning_tree, tree_distance

MEAN_ROAD_COVERED_SURFACE = 64  # to compute number of roads, max 1 external connexion per 1/64 of the settlement size
SETTLEMENT_ACCESS_DIST = 25  # maximum distance from settlement center to road net


class Settlement:
    """
    Intermediate project: generate a realistic village on a flat terrain
    """
    def __init__(self, maps):
        # type: (TerrainMaps) -> Settlement
        self._maps = maps  # type: TerrainMaps
        self.districts: Districts = Districts(self._maps.area)
        self._origin = maps.area.origin
        self._center: Point = Point(0, 0)
        self._road_network = self._maps.road_network
        self._parcels: List[Parcel] = []

    def __random_border_point(self):
        # type: () -> Point
        # width, length = self.limits.width, self.limits.length
        # # random draw to decide on what side the border will be. Favors larger sides
        # if bernouilli(width / (width + length)):
        #     # border point along x border
        #     x = 0 if bernouilli() else self.limits.width - 1
        #     z = randint(0, length - 1)
        # else:
        #     # border point along z border
        #     x = randint(0, width - 1)
        #     z = 0 if bernouilli() else self.limits.length - 1
        # return Point2D(x, z)
        return choice(self._maps.fluid_map.external_connections)

    def build_districts(self, **kwargs):
        self.districts.build(self._maps, **kwargs)
        district_centers = self.districts.district_centers

        # init road net
        if self.districts.n_districts >= 2:
            main_roads = min_spanning_tree(district_centers)
            for p1, p2 in main_roads:
                self._road_network.create_road(p1.asPosition, p2.asPosition)
        else:
            self._road_network.create_road(district_centers[0], district_centers[0])
        self.init_road_network()

        town_centers = {t.center for t in self.districts.towns.values()}
        compute_train_line(self._maps.rail_network, town_centers)

        # mark town centers
        for town_center in town_centers:
            self._parcels.append(Parcel(town_center, BuildingType.ghost, self._maps))

    def init_road_network(self):
        max_road_count = max(1, min(self.limits.width, self.limits.length) // MEAN_ROAD_COVERED_SURFACE)
        road_count = min(np.random.geometric(1. / max_road_count), max_road_count * 3 // 2)
        logging.debug('New settlement will have {} external connections B)'.format(road_count))
        out_connections = [Point(self.limits.width//2, self.limits.length//2)]

        for road_id in range(road_count):
            min_distance_to_roads = min(self.limits.width, self.limits.length) / (road_id+1)
            logging.debug('Generating road #{}'.format(road_id+1))
            # generate new border point far enough from existing points
            while True:
                new_road_point = self.__random_border_point()
                distances = [euclidean(road_point, new_road_point) for road_point in out_connections]
                distance_to_roads = min(distances)
                log_args = (str(new_road_point), distance_to_roads, min_distance_to_roads)
                if distance_to_roads >= min_distance_to_roads:
                    out_connections.append(new_road_point)
                    logging.debug('\tSettled on border point {} at {}m >= {}m'.format(*log_args))
                    break
                else:
                    logging.debug('\tDismissed point {} at {}m < {}m'.format(*log_args))
                    min_distance_to_roads *= 0.9
            print(f"[Settlement] Creating road towards border point at {out_connections[-1]}")
            self._road_network.connect_to_network(out_connections[-1])

    def build_skeleton(self, time_limit: int, do_visu: bool = False):
        village_skeleton = VillageSkeleton('Flat_scenario', self._maps, self.districts, self._parcels)
        village_skeleton.grow(time_limit, do_visu)

    def define_parcels(self, **kwargs):
        """
        Parcel extension from initialized parcels. Parcels are expended in place
        """
        print("Extending parcels")
        from terrain import ObstacleMap
        ObstacleMap().add_obstacle(Point(0, 0), self._road_network.obstacle)
        ObstacleMap().add_obstacle(Point(0, 0), self._maps.fluid_map.as_obstacle_array)
        ObstacleMap().add_obstacle(Point(0, 0), self._maps.rail_network.obstacle)
        for parcel in self._parcels:
            ObstacleMap().hide_obstacle(*parcel.obstacle(forget=True), False)
            if isinstance(parcel, MaskedParcel):
                trunk_obstacle = ObstacleMap()[parcel.minx: parcel.maxx, parcel.minz: parcel.maxz]
                parcel.add_mask(trunk_obstacle == 0)
            parcel.compute_entry_point()
            ObstacleMap().add_obstacle(*parcel.obstacle())
        expendable_parcels = SortedList(self._parcels, lambda _: _.surface)

        while expendable_parcels:
            # extend expendables parcels from smaller to larger while there still are some
            parcel = expendable_parcels.pop(0)
            if parcel.entry_point != parcel.center:
                road_dir = Direction.of(*(parcel.entry_point - parcel.center).coords)
                lateral_dir = road_dir.rotate() if bernouilli() else -road_dir.rotate()

                priority_directions = [road_dir, lateral_dir, -lateral_dir, -road_dir]
                try:
                    direction = next(filter(lambda d: parcel.is_expendable(d), priority_directions))
                    parcel.expand(direction)
                    expendable_parcels.add(parcel)
                except StopIteration:
                    logging.info(f"Cannot extend {str(parcel)} any more")

        # set parcels heights
        def define_parcels_heights(__parcel):
            # type: (Parcel) -> None
            min_y = np.percentile(__parcel.height_map, 25)
            max_y = np.percentile(__parcel.height_map, 75)
            road_y = self._maps.height_map[__parcel.entry_point]
            y = road_y
            if road_y > max_y:
                y = max_y
            elif road_y < min_y:
                y = min_y
            try:
                d = min(euclidean(__parcel.center, _.center) for _ in
                        filter(lambda p: p.building_type.name == "ghost", self._parcels)
                        )
            except ValueError:
                d = 0
            h = int(MAX_HEIGHT * exp(-d / BUILDING_HEIGHT_SPREAD))
            __parcel.set_height(y, h)

        print("Defining parcels' and buildings' heights")
        for p in self._parcels:
            p.compute_entry_point()
            define_parcels_heights(p)

    def generate(self):
        self._road_network.generate(self._maps, self.districts)
        self._maps.rail_network.generate(self._maps, self.districts)

        self.__generate_road_signs()

        for parcel in self._parcels:  # type: Parcel
            def in_bounds():
                corner1 = parcel.position + self._origin
                corner2 = corner1 + Point(parcel.width, parcel.length) - Point(1, 1)
                return corner1 in self._maps.area and corner2 in self._maps.area

            if not in_bounds():
                continue
            try:
                print("Generating", str(parcel))
                _gen = parcel.generator
                _gen.choose_sub_generator(self._parcels)
                from building_seeding.settlement import Town
                parcel_district: Town = argmin(self.districts.towns.values(), lambda s: euclidean(parcel.center, s.center))
                # _palette = get_biome_palette(parcel.biome(terrain))
                _palette = parcel_district.palette
                _gen.generate(self._maps, parcel.height_map, _palette)
            except Exception:
                traceback.print_exc()
        dump()

    @property
    def town_center(self):
        return self._center

    @property
    def limits(self):
        return self._maps.area

    def terraform(self):
        """
        Smooth out the terrain around locations that will be built on
        """
        import cv2
        from numpy import uint8
        base_height = uint8(self._maps.height_map[:])
        # Compute constructable locations: road net + parcels
        mask = np.zeros(base_height.shape)

        for pos in self._road_network.road_blocks:
            x0 = max(pos.x - 2, 0)
            x1 = min(pos.x + 3, self.limits.width)
            z0 = max(pos.z - 2, 0)
            z1 = min(pos.z + 3, self.limits.length)
            mask[x0:x1, z0:z1] = 1

        for parcel in self._parcels:
            x0 = max(parcel.minx - 2, 0)
            x1 = min(parcel.maxx + 3, self.limits.width)
            z0 = max(parcel.minz - 2, 0)
            z1 = min(parcel.maxz + 3, self.limits.length)
            mask[x0:x1, z0:z1] = 1

        mask[self._maps.fluid_map.water > 0] = 0  # discount water

        smooth_height = np.copy(base_height)
        for _ in range(TERRAFORM_ITERATIONS):
            smooth_height = cv2.blur(smooth_height, (9, 9))

        u: Position
        for u in filter(lambda u: mask[u.xz] and base_height[u.xz] != smooth_height[u.xz], BuildArea.building_positions()):
            ya = base_height[u.x, u.z]
            new_y = smooth_height[u.x, u.z]
            self._maps.height_map.update([u], [new_y])
            surface_block = self._maps.level.getBlockAt(u.abs_x, ya, u.abs_z)
            setBlock(Point(u.abs_x, u.abs_z, new_y), surface_block)

            if ya + 4 > new_y > ya:
                below_block = self._maps.level.getBlockAt(u.abs_x, ya - 1, u.abs_z)
                box = TransformBox((u.abs_x, ya, u.abs_z), (1, new_y - ya, 1))
                fillBlocks(box, below_block)

            elif new_y < ya:
                # actually an else block
                box = TransformBox((u.abs_x, new_y + 1, u.abs_z), (1, ya - new_y, 1))
                fillBlocks(box, BlockAPI.blocks.Air)

    def clean_road_network(self):
        road_map = (self._road_network.network > 0).astype(int)
        network: RoadNetwork = self._road_network

        def degree(p: Point):
            arr = road_map[max(0, p.x - 1):min(p.x + 2, road_map.shape[0]),
                           max(0, p.z - 1):min(p.z + 2, road_map.shape[1])]
            return arr.sum() - 1

        def get_neighbour(p: Point):
            for dx, dz in itertools.product(range(-1, 2), range(-1, 2)):
                if self._maps.in_limits(p + Point(dx, dz), False) and road_map[p.x + dx, p.z + dz]:
                    return p + Point(dx, dz)

        def unset_road(p: Position):
            if p in network.road_blocks:
                network.road_blocks.remove(p)
            else:
                network.special_road_blocks.remove(p)
            road_map[p.x, p.z] = 0
            network.network[p.x, p.z] = 0

        for extremity in filter(lambda node: degree(node) == 1, network.nodes):
            truncated_length = 0
            while truncated_length < AVERAGE_PARCEL_SIZE / 2 and bool(extremity) and degree(extremity) == 1:
                unset_road(extremity)
                extremity = get_neighbour(extremity)
                truncated_length += 1

    def __generate_road_signs(self):
        from building_seeding.settlement import Town
        town_centers: Dict[Position, Town] = {_.center: _ for _ in self.districts.towns.values()}
        roads = min_spanning_tree(self.districts.district_centers)
        distance_map = tree_distance(roads)
        for point in self.districts.district_centers:
            towns: List[Town] = sorted(filter(lambda u: point != u.center, town_centers.values()), key=lambda town: distance_map[point, town.center])
            towns = towns[:3] if len(towns) > 3 else towns
            y = self._maps.height_map[point] + 1
            setBlock(Point(point.x + self._origin.x, point.z + self._origin.z, y - 1), BlockAPI.blocks.PolishedDiorite)
            for dy, town in enumerate(towns):
                dir = town.center - point
                dir = Point(-dir.z, dir.x)
                pos = point + self._origin + Direction.Top.value * (y + dy)

                nom_voisine = town.name
                dist = int(distance_map[point, town.center])
                if point in town_centers and town == towns[-1]:
                    nom_ville = town_centers[point].name
                    place_sign(pos, BlockAPI.blocks.OakSign, dir, Text1=nom_ville, Text2="--------", Text3=f"{nom_voisine}", Text4=f"<--- {dist}m")
                else:
                    place_sign(pos, BlockAPI.blocks.OakSign, dir, Text2=f"{nom_voisine}", Text3=f"<--- {dist}m")
