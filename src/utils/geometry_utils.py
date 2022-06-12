from enum import Enum
from typing import Iterable, Tuple

import numpy as np

from gdpc.interface import requestBuildArea, setBuildArea
from utils.misc_objects_functions import argmax, Singleton
from utils.pymclevel.box import BoundingBox


__all__ = [
    'absolute_distance',
    'Bounds',
    'BuildArea',
    'Direction',
    'euclidean',
    'intersect',
    'manhattan',
    'manhattan2d',
    'Point',
    'Position',
    'TransformBox',
    'X_ARRAY',
    'Z_ARRAY'
]


class Point(np.ndarray):
    """
    Minecraft main coordinates are x, z, while y represents altitude (from 0 to 255)
    I choose to set y as an optional coordinate so that you can work with 2D points by just ignoring the 3rd coord
    """

    def __new__(cls, x, z, y=0):
        return np.asarray((x, y, z)).view(cls)

    def __array_finalize__(self, obj):
        if obj is None: return

    def __str__(self):
        res = "(x:{}".format(self.x)
        res += "; y:{}".format(self.y) if self.y else ""
        res += "; z:{})".format(self.z)
        return res

    def __eq__(self, other):
        return all(super(Point, self).__eq__(other))

    def __ne__(self, other):
        return not self == other

    def __bool__(self):
        return True

    def __hash__(self):
        return hash(self.xyz)

    @property
    def xyz(self):
        return self.x, self.y, self.z

    @property
    def xz(self):
        return self.x, self.z

    @property
    def x(self):
        return self[0]

    @property
    def y(self):
        return self[1]

    @property
    def z(self):
        return self[2]

    @property
    def norm(self):
        from math import sqrt
        return sqrt(self.dot(self))

    @property
    def unit(self):
        return self / self.norm

    @property
    def asPosition(self):
        return Position(self.x, self.z, self.y)

    def withCoords(self, x=None, z=None, y=None):
        return Point(
            x if x is not None else self.x,
            z if z is not None else self.z,
            y if y is not None else self.y
        )


class Position(Point):
    """
    Point with integer coordinates, represents a position in the Minecraft world. Holds x, y, z coordinates relative to
    the building area, that can be converted to MC coords with the abs_x and abs_z properties
    """

    def withCoords(self, x=None, z=None, y=None):
        return super().withCoords(x, z, y).view(Position)

    def __new__(cls, x, z, y=0, absolute_coords=False):
        if absolute_coords:
            x -= BuildArea().x
            z -= BuildArea().z
        ix, iy, iz = int(round(x)), int(round(y)), int(round(z))
        return Point.__new__(cls, ix, iz, iy)

    @property
    def abs_x(self):
        """
        Returns
        -------
        Absolute X-coordinate in the Minecraft map
        """
        return self.x + BuildArea().x

    @property
    def abs_z(self):
        """
        Returns
        -------
        Absolute Z-coordinate in the Minecraft map
        """
        return self.z + BuildArea().z

    @property
    def abs_xyz(self):
        return self.abs_x, self.y, self.abs_z


def euclidean(p1: Point, p2: Point) -> float:
    return (p1 - p2).norm


def manhattan(p1: Point, p2: Point) -> float:
    return abs(p2.x - p1.x) + abs(p2.y - p1.y) + abs(p2.z - p1.z)


def manhattan2d(p1: Point, p2: Point) -> float:
    return sum(abs(p2 - p1).xz)


def absolute_distance(p1: Point, p2: Point) -> float:
    return max(abs(p2 - p1))


def intersect(segment: Tuple[Point, Point], other_segment: Tuple[Point, Point]) -> bool:
    """
    Check if to segments intersect
    from https://www.geeksforgeeks.org/check-if-two-given-line-segments-intersect/
    """
    def onSegment(p: Point, q: Point, r: Point) -> bool:
        """
        Given three collinear points p, q, r, the function checks if
        point q lies on line segment 'pr'
        """
        if ((q.x <= max(p.x, r.x)) and (q.x >= min(p.x, r.x)) and
                (q.z <= max(p.z, r.z)) and (q.z >= min(p.z, r.z))):
            return True
        return False

    def orientation(p: Point, q: Point, r: Point) -> int:
        # to find the orientation of an ordered triplet (p,q,r)
        # function returns the following values:
        # 0 : Collinear points
        # 1 : Clockwise points
        # 2 : Counterclockwise

        # See https://www.geeksforgeeks.org/orientation-3-ordered-points/amp/
        # for details of below formula.

        val = (float(q.z - p.z) * (r.x - q.x)) - (float(q.x - p.x) * (r.z - q.z))
        if val > 0:

            # Clockwise orientation
            return 1
        elif val < 0:

            # Counterclockwise orientation
            return 2
        else:

            # Collinear orientation
            return 0

    p1, q1 = segment
    p2, q2 = other_segment

    # Find the 4 orientations required for
    # the general and special cases
    o1 = orientation(p1, q1, p2)
    o2 = orientation(p1, q1, q2)
    o3 = orientation(p2, q2, p1)
    o4 = orientation(p2, q2, q1)

    # General case
    if (o1 != o2) and (o3 != o4):
        return True

    # Special Cases

    # p1 , q1 and p2 are collinear and p2 lies on segment p1q1
    if (o1 == 0) and onSegment(p1, p2, q1):
        return True

    # p1 , q1 and q2 are collinear and q2 lies on segment p1q1
    if (o2 == 0) and onSegment(p1, q2, q1):
        return True

    # p2 , q2 and p1 are collinear and p1 lies on segment p2q2
    if (o3 == 0) and onSegment(p2, p1, q2):
        return True

    # p2 , q2 and q1 are collinear and q1 lies on segment p2q2
    if (o4 == 0) and onSegment(p2, q1, q2):
        return True

    # If none of the cases
    return False


class Direction(Enum):
    """
    Custom direction class
    """

    East = Point(1, 0, 0)
    West = Point(-1, 0, 0)
    South = Point(0, 1, 0)
    North = Point(0, -1, 0)
    Top = Point(0, 0, 1)
    Bottom = Point(0, 0, -1)

    @staticmethod
    def of(dx=0, dy=0, dz=0):
        """
        Given a 3D vector, return the closer cardinal direction
        """
        if isinstance(dx, Point):
            return Direction.of(dx.x, dx.y, dx.z)
        assert not (dx == 0 and dy == 0 and dz == 0)  # assert that at least one coordinate is not null
        # keep only one non null coordinate
        kept_dir = argmax([abs(dx), abs(dy), abs(dz)])
        if kept_dir == 0:
            dy = dz = 0
        elif kept_dir == 1:
            dx = dz = 0
        else:
            dx = dy = 0

        # each direction is set to -1 or 1
        dir_x = int(dx / abs(dx)) if dx else 0  # 1, 0, or -1
        dir_y = int(dy / abs(dy)) if dy else 0
        dir_z = int(dz / abs(dz)) if dz else 0
        assert abs(dir_x) + abs(dir_y) + abs(dir_z) == 1  # safety check that the direction is valid
        point_dir = Point(dir_x, dir_z, dir_y)
        for direction in Direction:
            if direction.value == point_dir:
                return direction

    def __eq__(self, other):
        if isinstance(other, Direction):
            return (self.x == other.x) and (self.y == other.y) and (self.z == other.z)
        return False

    def __str__(self):
        # known_dirs = {'0 0 1': 'South', '0 1 0': 'Top', '1 0 0': 'East', '0 0 -1': 'North',
        #               '0 -1 0': 'Bottom', '-1 0 0': 'West'}
        # key = ' '.join(map(str, self.value.coords))
        # name = known_dirs[key] if key in known_dirs else "Unknown"
        # return "{} Direction".format(self.name)
        return self.name

    def __hash__(self):
        return hash(str(self))

    def __neg__(self):
        return Direction.of(-self.value)

    def __abs__(self):
        return Direction.of(abs(self.value))

    def rotate(self):
        """
        Rotates direction anti-normally, East.rotate() == South
        Returns anti-normal rotation of self
        -------

        """
        return Direction.of(dx=-self.value.z, dz=self.value.x)

    @property
    def x(self):
        return self.value.x

    @property
    def y(self):
        return self.value.y

    @property
    def z(self):
        return self.value.z

    @classmethod
    def from_string(cls, dir_str):
        str_to_coord = {value.lower(): key for key, value in Direction.__known_dirs.items()}  # reversed dict
        x, y, z = list(map(int, str_to_coord[dir_str].split(' ')))
        return Direction.of(x, y, z)

    @staticmethod
    def all_directions(as_points: bool = True):
        """

        :rtype: Iterable[Direction]
        """
        directions = [_ for _ in Direction]
        if as_points:
            directions = list(map(lambda d: d.value, directions))
        from random import shuffle
        shuffle(directions)
        return iter(directions)

    @staticmethod
    def cardinal_directions(as_points: bool = True):
        """

        :rtype: Iterable[Union[Direction, Point]]
        """
        directions = [Direction.East, Direction.South, Direction.West, Direction.North]
        if as_points:
            directions = list(map(lambda d: d.value, directions))
        from random import shuffle
        shuffle(directions)
        return iter(directions)


class BuildArea(metaclass=Singleton):
    def __init__(self, build_area_json=None):
        XFROM, XTO, ZFROM, ZTO = 'xFrom', 'xTo', 'zFrom', 'zTo'
        XFROM, XTO, ZFROM, ZTO = 0, 3, 2, 5
        if build_area_json is None:
            try:
                build_area_json = list(requestBuildArea())
                build_area_json[1] = 0
                build_area_json[4] = 256
                setBuildArea(*build_area_json)
            except IOError:
                print("Connection Error -> using empty BuildArea")
                build_area_json = {XFROM: 0, XTO: 1, ZFROM: 0, ZTO: 1}

        for (key1, key2, tag) in {(XFROM, XTO, 'X'), (ZFROM, ZTO, 'Z')}:
            if build_area_json[key1] == build_area_json[key2]:
                raise ValueError("lower {} and upper {} bounds must be different !".format(tag, tag))
            elif build_area_json[key1] > build_area_json[key2]:
                build_area_json[key1], build_area_json[key2] = build_area_json[key2], build_area_json[key1]

        self.__xfrom = build_area_json[XFROM]
        self.__xto = build_area_json[XTO]
        self.__zfrom = build_area_json[ZFROM]
        self.__zto = build_area_json[ZTO]

    def __contains__(self, item):
        assert isinstance(item, Point)
        return (self.x <= item.x <= self.__xto) and (self.z <= item.z <= self.__zto)

    @property
    def x(self):
        return self.__xfrom

    @property
    def z(self):
        return self.__zfrom

    @property
    def origin(self):
        return Point(self.x, self.z)

    @property
    def width(self) -> int:
        return self.__xto - self.__xfrom + 1

    @property
    def length(self):
        return self.__zto - self.__zfrom + 1

    @property
    def rect(self):
        return self.x, self.z, self.width, self.length

    @property
    def box(self):
        return BoundingBox((self.x, 0, self.z), (self.width, 256, self.length))

    @property
    def json(self):
        return {
            "xFrom": self.x,
            "zFrom": self.z,
            "xTo": self.x + self.width,
            "zTo": self.z + self.length
        }

    @staticmethod
    def building_positions() -> Iterable[Position]:
        from itertools import product
        from utils import Position
        for x, z in product(range(BuildArea().width), range(BuildArea().length)):
            yield Position(x, z)

    def __str__(self):
        return f"build area of size {Point(self.width, self.length)} starting in {self.origin}"


class TransformBox(BoundingBox):
    """
    Adds class methods to the BoundingBox to transform the box's shape and position
    """

    def translate(self, dx=0, dy=0, dz=0, inplace=False):
        if isinstance(dx, Direction):
            return self.translate(dx.x, dx.y, dx.z, inplace)
        if inplace:
            self._origin += (dx, dy, dz)
            return self
        else:
            return TransformBox(self.origin + (dx, dy, dz), self.size)

    def split(self, dx=None, dy=None, dz=None):
        assert (dx is not None) ^ (dy is not None) ^ (dz is not None)
        if dx is not None:
            if dx < 0:
                dx += self.width
            b0 = TransformBox(self.origin, (dx, self.height, self.length))
            b1 = TransformBox((self.origin + (dx, 0, 0)), (self.size - (dx, 0, 0)))
        elif dy is not None:
            if dy < 0:
                dy += self.height
            b0 = TransformBox(self.origin, (self.width, dy, self.length))
            b1 = TransformBox((self.origin + (0, dy, 0)), (self.size - (0, dy, 0)))
        else:
            if dz < 0:
                dz += self.length
            b0 = TransformBox(self.origin, (self.width, self.height, dz))
            b1 = TransformBox((self.origin + (0, 0, dz)), (self.size - (0, 0, dz)))
        return [b0, b1]

    def expand(self, dx_or_dir, dy=None, dz=None, inplace=False):
        # if isinstance(dx_or_dir, Direction):
        if isinstance(dx_or_dir, Direction):
            direction = dx_or_dir
            dpos = (min(direction.x, 0), min(direction.y, 0), min(direction.z, 0))
            dsize = (abs(direction.x), abs(direction.y), abs(direction.z))
            expanded_box = TransformBox(self.origin + dpos, self.size + dsize)
        else:
            expanded_box = TransformBox(BoundingBox.expand(self, dx_or_dir, dy, dz))
        return self.copy_from(expanded_box) if inplace else expanded_box

    def enlarge(self, direction, reverse=False, inplace=False):
        # type: (Direction, bool, bool) -> TransformBox
        """
        For example, TransformBox((0, 0, 0), (1, 1, 1)).expand(East) -> TransformBox((-1, 0, 0), (3, 1, 1))
        """
        copy_box = TransformBox(self.origin, self.size)
        dx, dz = 1 - abs(direction.x), 1 - abs(direction.z)
        if reverse:
            dx, dz = -dx, -dz
        copy_box = copy_box.expand(dx, 0, dz)
        if inplace:
            self.copy_from(copy_box)
        else:
            return copy_box

    def copy_from(self, other):
        self._origin = other.origin
        self._size = other.size
        return self

    def __sub__(self, other):
        # type: (TransformBox) -> TransformBox
        """
        exclusion operator, only works if self is an extension of other in a single direction
        should work well with self.expand(Direction), eg box.expand(South) - box -> southern extension of box
        """
        same_coords = [self.minx == other.minx, self.maxx == other.maxx, self.minz == other.minz,
                       self.maxz == other.maxz]
        assert sum(same_coords) == 3  # only one of the 4 bool should be False
        if not same_coords[0]:  # supposedly self.minx < other.minx
            return self.split(dx=1)[0]
        elif not same_coords[1]:  # supposedly self.minx < other.minx
            return self.split(dx=self.width - 1)[1]
        elif not same_coords[2]:  # supposedly self.minx < other.minx
            return self.split(dz=1)[0]
        else:  # supposedly self.minx < other.minx
            return self.split(dz=self.length - 1)[1]

    def closest_border(self, px, z=None, relative_coords=False):
        """Gets the border point of self closer to the input point"""
        if z is not None:
            return self.closest_border(Point(px, z), None, relative_coords)

        if relative_coords:
            px = Point(px.x + self.minx, px.z + self.minz)

        if self.is_lateral(px.x, px.z):
            return px - Point(self.minx, self.minz) if relative_coords else px

        xm, zm = (self.minx + self.maxx) // 2, (self.minz + self.maxz) // 2
        corners = [Point(self.minx, zm), Point(self.maxx-1, zm), Point(xm, self.minz), Point(xm, self.maxz-1)]
        distance = [euclidean(px, _) for _ in corners]
        i = np.argmin(distance)
        if i == 0:
            return Point(0, px.z - self.minz) if relative_coords else Point(self.minx, px.z)
        elif i == 1:
            return Point(self.width - 1, px.z - self.minz) if relative_coords else Point(self.maxx - 1, px.z)
        elif i == 2:
            return Point(px.x - self.minx, 0) if relative_coords else Point(px.x, self.minz)
        else:
            return Point(px.x - self.minx, self.length - 1) if relative_coords else Point(px.x, self.maxz - 1)

    def is_corner(self, new_gate_pos):
        return self.is_lateral(new_gate_pos.x) and self.is_lateral(None, new_gate_pos.z)

    def is_lateral(self, x=None, z=None):
        assert x is not None or z is not None
        if x is None:
            return z == self.minz or z == self.maxz - 1
        if z is None:
            return x == self.minx or x == self.maxx - 1
        else:
            return self.is_lateral(x, None) or self.is_lateral(None, z)

    @property
    def surface(self):
        return self.width * self.length


class Bounds:
    def __init__(self, origin: Position, shape: Point):
        self.__origin = origin
        self.__shape = shape

    @property
    def width(self):
        return self.__shape.x

    @property
    def length(self):
        return self.__shape.z

    @property
    def minx(self):
        return self.__origin.x

    @property
    def maxx(self):
        return self.minx + self.width

    @property
    def minz(self):
        return self.__origin.z

    @property
    def maxz(self):
        return self.minz + self.length


X_ARRAY: np.ndarray = np.array([[x for z in range(BuildArea().length)] for x in range(BuildArea().width)])
Z_ARRAY: np.ndarray = np.array([[z for z in range(BuildArea().length)] for x in range(BuildArea().width)])


if __name__ == '__main__':
    print(Direction.East, Direction.East.value, Direction.East.name)
    print(list(Direction.all_directions(False)))
