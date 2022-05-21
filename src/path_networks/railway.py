from typing import Dict, List

import numpy as np

from generation import Generator
from generation.structure import AREA_STRUCTURE
from utils import Position, Direction, manhattan, getBlockRelativeAt, ground_blocks, euclidean, Point, BlockAPI, \
    place_torch, clear_tree_at, dump, TransformBox, BoundingBox, argmin, BuildArea


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
            x, y, z = curve_p.xyz
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
        box = TransformBox(origin, size).translate(*BuildArea().origin.xyz)
        AREA_STRUCTURE.fill(box, BlockAPI.blocks.PrismarineBricks, 1005)
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
            place_accelerator(self.__in + position + self.__in_dir.value.asPosition, self.__in + position)
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
    #
    # @property
    # def isStraight(self):
    #     x_straight = self.length == 1 and (self.__in_dir.x * self.__out_dir.x == -1)
    #     z_straight = self.width == 1 and (self.__in_dir.z * self.__out_dir.z == -1)
    #     return x_straight or z_straight

    @property
    def direction(self):
        return self.__in_dir

    @property
    def entry(self):
        return self.__in

    @property
    def exit(self):
        return self.__out


class TrainStation(Generator):
    PLATFORM_LENGTH = 7

    def __init__(self, position: Position, orientation: Direction, **kwargs):
        box = BoundingBox(position.abs_xyz, (1, 1, 1))
        super().__init__(box, **kwargs)
        self.position: Position = position
        self.orientation: Direction = orientation
        self.connectors: List[RailConnector] = []
        self.__create_connectors()

    def connect(self, station: Position):
        if all(conn.is_full for conn in self.connectors):
            if len(self.connectors) == 2:
                return None  # todo: handle stations with more than 2 neighbours
            self.__create_connectors()
        station_conn: RailConnector = argmin([conn for conn in self.connectors if not conn.is_full], lambda c: manhattan(c, station))
        return station_conn

    def __create_connectors(self):
        prev_conn_mid = (sum(self.connectors[-2:]) / 2).asPosition if self.connectors else self.position
        conn_mid: Position = prev_conn_mid + self.orientation.rotate().value * 4

        conn1 = RailConnector(conn_mid - self.orientation.value * (self.PLATFORM_LENGTH - self.PLATFORM_LENGTH // 2))
        conn2 = RailConnector(conn_mid + self.orientation.value * (self.PLATFORM_LENGTH // 2))
        conn1.branch(RailWay(conn1, conn2))
        conn2.branch(RailWay(conn1, conn2))
        self.connectors.extend((conn1, conn2))

    def generate(self, level, height_map=None, palette=None):
        conn1, conn2 = self.connectors[:2]
        rail_dir: Direction = self.orientation
        norm_dir: Point = self.orientation.rotate().value
        y = Point(0, 0, height_map[self.position.xz])
        Rails(conn1 + norm_dir + y, rail_dir, conn2 + norm_dir + y, -rail_dir, False).generate(level)
        Rails(conn2 - norm_dir + y, -rail_dir, conn1 - norm_dir + y, rail_dir, False).generate(level)


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


def place_accelerator(p1: Position, p2: Position, y=None):
    assert euclidean(p1, p2) == 1
    assert isinstance(p1, Point) or (isinstance(p1, Point) and y is not None)
    if isinstance(p1, Point):
        y = p1.y
    direction_str = axis_dir(Direction.of(*(p2 - p1).xyz))
    AREA_STRUCTURE.set(Point(p1.abs_x, p1.abs_z, y + 1), f"detector_rail[shape={direction_str}])", 1003)
    AREA_STRUCTURE.set(Point(p1.abs_x, p1.abs_z, y), BlockAPI.blocks.Cobblestone, 1003)
    AREA_STRUCTURE.set(Point(p2.abs_x, p2.abs_z, y + 1), f"powered_rail[shape={direction_str}]", 1003)
    AREA_STRUCTURE.set(Point(p2.abs_x, p2.abs_z, y), BlockAPI.blocks.Cobblestone, 1003)


def axis_dir(direction):
    if direction == Direction.North or direction == Direction.South:
        return "north_south"
    elif direction == Direction.East or direction == Direction.West:
        return "east_west"
    raise ValueError