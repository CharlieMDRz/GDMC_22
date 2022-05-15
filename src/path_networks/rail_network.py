import logging
import math
import random
import traceback
from typing import List, Set, Dict

import numpy as np
from gdpc import lookup

from generation.structure import AREA_STRUCTURE
from parameters import RAIL_ROAD_SPACING, RAIL_ROAD_PENALTY
from utils import Position, Point, manhattan, euclidean, Direction, ground_blocks, BlockAPI, \
    TransformBox, place_torch, argmin, getBlockRelativeAt, BuildArea, Singleton, clear_tree_at
from utils.algorithms.graphs import GridGraph
from .path_finder import PathFinder
from .railroad_generator import RailRoadGenerator
from .road_network import RoadNetwork, dump, road_build_cost, MAX_INT

__all__ = [
    'RailNetwork',
    'RailRoadGraph'
]

all_but_rails = [_ for _ in lookup.BLOCKS if ("Rail" not in str(_) and "Redstone" not in str(_))]


class RailNetwork(RoadNetwork, metaclass=Singleton):
    __railPathFinder: PathFinder

    def __init__(self, width, length, mc_map=None):
        super().__init__(width, length, mc_map)
        self.__maps = mc_map
        self.__limits = (width, length)
        self.__elements: Set[RailWay] = set()
        self.__connectors: Dict[Position, RailConnector] = {}
        self.__railPathFinder = PathFinder(7, RailRoadGraph(7, 15, cost=rail_road_build_cost), GridGraph(True, step=7, cost=road_build_cost))

    def add_edge(self, p1: Position, p2: Position, is_station: bool = False):

        for p in [p1, p2]:
            if p not in self.__connectors:
                self.__connectors[p] = RailConnector(p)
        c1, c2 = self.__connectors[p1], self.__connectors[p2]
        new_edge = TrainStation(c1, c2) if is_station else RailWay(c1, c2)

        c1.branch(new_edge)
        c2.branch(new_edge)
        self.__elements.add(new_edge)

    def get_road_width(self, x: Point or int, z: int = None) -> int:
        return 5

    def create_road(self, root_point=None, ending_point=None, path=None, is_station=False):
        logging.info(f"Creating rail way between {root_point} and {ending_point}")
        # rough_path: List[Position] = self.__railPathFinder.getRoughPath(root_point, ending_point)
        rough_path: List[Position] = self.__railPathFinder.getPath(root_point, ending_point)
        rough_path = [Position(p.x, p.z, self.__maps.height_map[p.x, p.z]) for p in rough_path]
        print("\n".join(map(str, rough_path)))
        path = [root_point]
        for i in range(len(rough_path)-1):
            # section nodes
            cur_start = rough_path[i]
            cur_exit = rough_path[i+1]
            past_start = rough_path[i-1] if i > 0 else cur_start
            next_exit = rough_path[i+2] if (i+2) < len(rough_path) else cur_exit

            # section direction
            start_dir = (cur_exit - past_start) / 3
            end_dir = (next_exit - cur_start) / 3

            path.extend(hermit_curve(cur_start, start_dir, cur_exit, end_dir)[1:])
            self.add_edge(cur_start, cur_exit)

        return super().create_road(path=path)

    def connect_to_network(self, target: Position, margin: int = 0) -> List[Set[Point]]:
        return super().connect_to_network(target, margin)

    def cycle_creation_condition(self, node1: Point, node2: Point) -> (List[Point], List[Point]):
        return [], []

    def generate(self, level, districts):
        RailRoadGenerator(self.network > 0).generate(level, self.__maps.height_map)
        # for rail in self.__elements:
        #     try:
        #         rail.generate(level)
        #     except AssertionError:
        #         print(f"failed to generate section {rail._connectors}")
        #         print(traceback.format_exc())
        #         continue


def hermit_curve(p0: Point, q0: Point, p1: Point, q1: Point) -> List[Point]:

    def hermit(t):
        hp0 = (1 - t) ** 2 * (1 + 2 * t)

        hp1 = t ** 2 * (3 - 2 * t)

        hq0 = t * (1 - t) ** 2

        hq1 = -(t ** 2) * (1 - t)

        return (p0 * hp0) + (p1 * hp1) + (q0 * hq0) + (q1 * hq1)

    distance = int(manhattan(p0, p1))
    curve = [hermit(0).asPosition]
    for t in np.linspace(0, 1, distance * 2):
        new_curve_point = hermit(t).asPosition
        if new_curve_point.xz != curve[-1].xz:
            curve.append(new_curve_point)
    return curve
    # curve = {hermit(i / distance).asPosition for i in range(distance + 1)}
    # ordered_curve = sorted(curve, key=lambda pt: euclidean(p0, pt))
    # return ordered_curve


class RailElement(object):
    def __init__(self, origin, destination):
        self._connectors = [origin, destination]

    def generate(self, level):
        pass

    def other_end(self, end):
        assert end in self._connectors
        if end == self._connectors[0]:
            return self._connectors[1]
        else:
            return self._connectors[0]

    @property
    def width(self):
        return 1 + abs(self._connectors[0].pos.x - self._connectors[1].pos.x)

    @property
    def length(self):
        return 1 + abs(self._connectors[0].pos.z - self._connectors[1].pos.z)

    @property
    def isStraight(self):
        return self.width == 1 or self.length == 1

    @property
    def minx(self):
        return min(_.pos.x for _ in self._connectors)

    @property
    def miny(self):
        return min(_.pos.y for _ in self._connectors)

    @property
    def minz(self):
        return min(_.pos.z for _ in self._connectors)


class RailConnector(Position):
    CAPACITY = 2

    neighbours: Dict[Direction, RailElement]
    x_orientation: bool

    def __new__(cls, position):
        pos = Position.__new__(cls, position.x, position.z, position.y)
        pos.neighbours = {}
        pos.x_orientation = False
        return pos

    def branch(self, new_edge):
        # assert not self.is_full
        other_conn = new_edge.other_end(self)
        railway_vector = other_conn.pos - self.pos
        edge_dir = Direction.of(dx=railway_vector.x, dz=railway_vector.z)

        if self.neighbours:
            old_edge = list(self.neighbours.values())[0]
            if old_edge.isStraight:
                edge_dir = -list(self.neighbours.keys())[0]
            elif new_edge.isStraight:
                self.neighbours.clear()
                self.neighbours[-edge_dir] = old_edge
        self.neighbours[edge_dir] = new_edge
        self.__compute_orientation()

    def other_end(self, end):
        ends = list(self.neighbours.values())
        assert end in ends
        if end is ends[0]:
            return ends[1]
        else:
            return ends[0]

    @property
    def is_full(self):
        return len(self.neighbours) == self.CAPACITY

    @property
    def nodes(self):
        dp = Position(1, 0) if self.x_orientation else Position(0, 1)
        return self - dp, self + dp

    def __compute_orientation(self):
        if Direction.North in self.neighbours.keys() or Direction.South in self.neighbours.keys():
            self.x_orientation = True
        elif Direction.East in self.neighbours.keys() or Direction.West in self.neighbours.keys():
            self.x_orientation = False

    def direction(self, neighbour) -> Direction:
        assert neighbour in self.neighbours.values()  # todo: sometimes fail
        return [_ for (_, value) in self.neighbours.items() if value == neighbour][0]

    @property
    def pos(self):
        return self

    def getNodes(self, direction):
        # type: (Direction) -> (Position, Position)
        return self.pos - direction.rotate().value.asPosition, self.pos + direction.rotate().value.asPosition


class RailWay(RailElement):
    DIST_BTWN_LIGHTS = 6

    def __init__(self, connector1, connector2):
        # type: (RailConnector, RailConnector) -> None
        assert isinstance(connector1, RailConnector)
        assert isinstance(connector2, RailConnector)
        RailElement.__init__(self, connector1, connector2)
        self._connectors = [connector1, connector2]  # type: List[RailConnector]

    def generate(self, level):
        # get rails
        d0, d1 = self._connectors[0].direction(self), self._connectors[1].direction(self)
        p0, q0 = self._connectors[0].getNodes(d0)
        p1, q1 = self._connectors[1].getNodes(d1)
        # todo: clear the way first ? then gen tracks, then gen lights and decorations (?)
        # generate tracks
        try:
            Rails(p0, d0, q1, d1, not self._connectors[1].other_end(self).isStraight).generate(level)
        except IndexError:
            pass

        try:
            Rails(p1, d1, q0, d0, not self._connectors[0].other_end(self).isStraight).generate(level)
        except IndexError:
            pass

        # generate lights, clear the way
        lights = []
        p0, p1 = self._connectors[0].pos, self._connectors[1].pos
        distance = manhattan(p0, p1)
        q0, q1 = d0.value.asPosition * (distance // 2), d1.value.asPosition * (-distance // 2)
        for curve_p in hermit_curve(p0, q0, p1, q1):  # type: Position
            x, y, z = curve_p.coords
            # if there are no lights or all lights are far enough and the position is unoccupied, place torch
            underground = getBlockRelativeAt(level.level, x, y + 3, z) in ground_blocks
            if lights == [] or euclidean(curve_p, lights[-1]) > self.DIST_BTWN_LIGHTS:
                lights.append(curve_p)
                if underground:
                    AREA_STRUCTURE.set(Point(x, z, y + 3), BlockAPI.blocks.SeaLantern)
                elif getBlockRelativeAt(level.level, x, y + 1, z) == 0:
                    AREA_STRUCTURE.set(Point(x, y + 1), BlockAPI.blocks.OakFence)
                    place_torch(x, y + 2, z)
                    if getBlockRelativeAt(level.level, x, y, z) == 0:
                        AREA_STRUCTURE.set(Point(x, z, y), BlockAPI.blocks.OakPlanks)
            if not underground:
                clear_tree_at(level, Point(curve_p.abs_x, curve_p.abs_z))
        dump()


class Rails(RailElement):
    DIST_BTWN_ACCELERATION = 16

    def __init__(self, connector1, direction1, connector2, direction2, links_to_curve):
        RailElement.__init__(self, connector1, connector2)
        self.__in = connector1.view(Position)  # type: Position
        self.__out = connector2.view(Position)  # type: Position
        self.__in_dir = direction1  # type: Direction
        self.__out_dir = direction2  # type: Direction
        self.__late_acceleration = links_to_curve

    def generate(self, level):
        if self.isStraight:
            self.__gen_straight_rail()
        else:
            self.__gen_spline_rail()

    def __gen_straight_rail(self):
        origin = (self.minx, self.miny, self.minz)
        size = (self.width, 1, self.length)
        box = TransformBox(origin, size).translate(*BuildArea().origin.coords)
        AREA_STRUCTURE.fill(box, BlockAPI.blocks.Stone, 1000)
        box.translate(dy=1, inplace=True)
        tunnel_box = box.expand(1, 0, 0) if self.length > self.width else box.expand(0, 0, 1)
        tunnel_box.expand(Direction.Top, inplace=True)

        AREA_STRUCTURE.fill(tunnel_box, BlockAPI.blocks.Air, 1000)
        material = "rail[shape=north_south]" if (self.width == 1) else "rail[shape=east_west]"
        AREA_STRUCTURE.fill(box, material, 1002)

        tunnel_box.translate(dy=1)
        AREA_STRUCTURE.fill(tunnel_box, BlockAPI.blocks.Air, 1001)

        accelerator_count = max(self.width, self.length) // self.DIST_BTWN_ACCELERATION
        for count in range(max(accelerator_count, 1)):
            position = self.__in_dir.value.asPosition * self.DIST_BTWN_ACCELERATION * count
            place_accelerator(self.__in + position, self.__in + position + self.__in_dir.value.asPosition)
        if self.__late_acceleration:
            place_accelerator(self.__out + self.__out_dir.value.asPosition, self.__out)

    def __gen_spline_rail(self):
        # todo: generate accelerators where possible ?
        n_points = int(manhattan(self.__in, self.__out))
        p0, p1 = self.__in, self.__out  # interpolation targets
        q0, q1 = self.__in_dir.value.asPosition * (n_points / 2), self.__out_dir.value.asPosition * (
                -n_points / 2)  # osculation targets
        accelerators = []

        def find_missing_point(_p1, _p2):
            distance = euclidean(_p1, _p2)
            direction = self.__in_dir
            for _ in range(4):
                new_point = _p1 + direction.value.asPosition
                if euclidean(new_point, _p2) < distance:
                    return new_point
                else:
                    direction = direction.rotate()

        def find_rail_block(dir1, dir2):
            if dir1 == dir2 or dir1 == -dir2:
                dir_str = axis_dir(dir1)
                if accelerators == [] or all(manhattan(a, cur_p) > self.DIST_BTWN_ACCELERATION for a in accelerators):
                    accelerators.append(cur_p)
                    return f"powered_rail[shape={dir_str}]"
                else:
                    return f"rail[shape={dir_str}]"
            else:
                rail_direction = dir1.value.asPosition + dir2.value.asPosition
                rail_x_dir = "east" if rail_direction.x == 1 else "west"
                rail_z_dir = "south" if rail_direction.z == 1 else "north"
                return f"rail[shape={rail_z_dir}_{rail_x_dir}]"

        def build_rail_block(x, y, z, block):
            # print(x, y, z, block)
            box = TransformBox((x - 1, y, z - 1), (3, 2, 3))
            AREA_STRUCTURE.fill(box, BlockAPI.blocks.Dirt, 1000)
            box.translate(dy=1, inplace=True)
            AREA_STRUCTURE.fill(box, BlockAPI.blocks.Air, 1001)
            AREA_STRUCTURE.set(Point(x, z, y), BlockAPI.blocks.Blackstone, 1002)
            # if getBlockRelativeAt(level.level, int(x), int(y) + 1, int(z)) == 0:
            AREA_STRUCTURE.set(Point(x, z, y + 1), block, 1002)
            if block.startswith("powered_rail"):
                normal_direction = in_dir.rotate()
                AREA_STRUCTURE.set(Point(x + normal_direction.x, z + normal_direction.z, y), BlockAPI.blocks.Cobblestone, 1002)
                AREA_STRUCTURE.set(Point(x + normal_direction.x, z + normal_direction.z, y + 1), BlockAPI.getTorch(redstone=True), 1002)

        curve = list(hermit_curve(p0, q0, p1, q1))

        in_dir = self.__in_dir
        cur_p: Position
        while curve:
            cur_p = curve.pop(0)

            if curve:
                nxt_p = curve[0]
                if sum(abs(cur_p - nxt_p).xz) > 1:
                    nxt_p = find_missing_point(cur_p, nxt_p)
                    curve.insert(0, nxt_p)
                nxt_dir = Direction.of(dx=(nxt_p.x - cur_p.x), dz=(nxt_p.z - cur_p.z))
            else:
                nxt_dir = self.__out_dir

            rail_block = find_rail_block(in_dir, nxt_dir)
            build_rail_block(cur_p.abs_x, cur_p.y, cur_p.abs_z, rail_block)
            in_dir = -nxt_dir

    @property
    def isStraight(self):
        x_straight = self.length == 1 and (self.__in_dir.x * self.__out_dir.x == -1)
        z_straight = self.width == 1 and (self.__in_dir.z * self.__out_dir.z == -1)
        return x_straight or z_straight

    @property
    def direction(self):
        return self.__in_dir

    @property
    def entry(self):
        return self.__in

    @property
    def exit(self):
        return self.__out


def place_accelerator(p1: Position, p2: Position, y=None):
    assert euclidean(p1, p2) == 1
    assert isinstance(p1, Point) or (isinstance(p1, Point) and y is not None)
    if isinstance(p1, Point):
        y = p1.y
    direction_str = axis_dir(Direction.of(*(p2 - p1).coords))
    AREA_STRUCTURE.set(Point(p1.x, p1.z, y + 1), f"detector_rail[shape={direction_str}])", 1003)
    AREA_STRUCTURE.set(Point(p1.x, p1.z, y), BlockAPI.blocks.Cobblestone, 1003)
    AREA_STRUCTURE.set(Point(p2.x, p2.z, y + 1), f"powered_rail[shape={direction_str}]", 1003)
    AREA_STRUCTURE.set(Point(p2.x, p2.z, y), BlockAPI.blocks.Cobblestone, 1003)


def axis_dir(direction):
    if direction == Direction.North or direction == Direction.South:
        return "north_south"
    elif direction == Direction.East or direction == Direction.West:
        return "east_west"
    raise ValueError


def compute_train_line(network: RailNetwork, stations: Set[Position]):
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
        if distance_array[i, j] > 32:
            network.create_road(stations_list[i], stations_list[j])
        else:
            stations_list[i] = stations_list[j]


class TrainStation(RailWay):
    pass


class RailRoadGraph(GridGraph):
    def __init__(self, section_length: int, curvature: float, **kwargs):
        super().__init__(True, **kwargs)
        self.__length = section_length
        self.__curvature = curvature
        self.__max_dev = math.pi - 2 * math.acos(section_length / (2 * curvature))

    def getNeighbours(self, node, **kwargs):
        def prev_angle(dx, dz):
            if dx == 0 and dz == 0:
                return (random.random() - .5) * math.pi * 2  # todo: happens for the first node (parent source = source)
            atan = math.atan(dz / dx)
            if dx >= 0:
                return atan
            else:
                return atan + math.pi
        root = kwargs.get("parent")
        prev_section: Point = node - root
        prev_direction = prev_angle(*prev_section.xz)
        min_direction = prev_direction - self.__max_dev
        max_direction = prev_direction + self.__max_dev

        if kwargs.get('target', False):
            target = kwargs.get('target')
            if euclidean(node, target) <= self.__length:
                return {target}

        neighbours = set()
        for angle in np.linspace(min_direction, max_direction, int(3 * self.__length * self.__max_dev)):
            arc_point: Point = node + Point(math.cos(angle), math.sin(angle)) * self.__length
            try:
                neighbours.add(Position(arc_point.x, arc_point.z))
            except ValueError:
                print(node, prev_section, prev_direction, angle, arc_point)

        neighbours = {_ for _ in neighbours if (2 <= _.x < self.width-2 and 2 <= _.z < self.length-2)}
        return neighbours


def rail_road_build_cost(src_point: Position, dst_point: Position):
    scale = manhattan(src_point, dst_point)
    cost = road_build_cost(src_point, dst_point)

    if cost <= MAX_INT:
        road_net: RoadNetwork = RoadNetwork.INSTANCE
        road_dist = road_net.get_distance(dst_point)
        if road_dist <= RAIL_ROAD_SPACING:
            cost += scale * RAIL_ROAD_PENALTY

    return cost
