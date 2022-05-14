import itertools
from math import ceil
from time import sleep
from typing import Dict

import numpy as np
from numpy import uint8
from numpy.random.mtrand import choice

from generation.generators import Generator, place_street_lamp, place_torch_post
from terrain import ObstacleMap
from utils import *
from utils.misc_objects_functions import raytrace


class RoadGenerator(Generator):

    def __init__(self, network, box: BoundingBox, maps):
        super().__init__(box)
        self.__network = network
        self.__fluids = maps.fluid_map
        self.__maps = maps
        self.__origin = Point(box.minx, box.minz)  # type: Point

    def generate(self, terrain, height_map=None, palette=None):
        from terrain import TerrainMaps
        terrain: TerrainMaps
        print("[RoadGenerator] generating road blocks...", end='')

        road_height_map = np.zeros(height_map.shape, dtype=uint8)
        network = np.zeros((self.width, self.length))
        network_palette = {}
        palette_network = {}
        x0, z0 = self.__origin.x, self.__origin.z
        sleep(.001)

        for road_block in self.__network.road_blocks.union(self.__network.special_road_blocks):
            road_width = (self.__network.get_road_width(road_block.x, road_block.z) - 1) / 2.0
            clear_tree_at(terrain, road_block + Point(x0, z0))
            for x in sym_range(road_block.x, road_width, self.width):
                for z in sym_range(road_block.z, road_width, self.length):
                    if road_height_map[x, z] > 0:
                        # if x, z is a road point or has already been computed, no need to do it now
                        continue
                    if not self.__fluids.is_water(x, z):
                        y, b = self.__compute_road_at(x, z, height_map, palette)
                        if b not in palette_network:
                            b_id = len(palette_network) + 1
                            palette_network[b] = b_id
                            network_palette[b_id] = b
                        else:
                            b_id = palette_network[b]
                        network[x, z] = b_id
                        road_height_map[x, z] = y

        for x in range(self.width):
            for z in range(self.length):
                if network[x, z]:
                    y, b_id = road_height_map[x, z], network[x, z]
                    b = network_palette[b_id]
                    xa, za = x + x0, z + z0
                    fillBlocks(TransformBox((xa, y, za), (1, 4, 1)), alpha.Air)
                    setBlock(Point(xa, za, y-1), alpha.Dirt)
                    setBlock(Point(xa, za, y), b)
                    # if "slab" not in b and "stair" not in b and bernouilli(0.1):
                    #     place_torch(terrain.level, xa, y + 1, za)
                    h = height_map[x, z]
                    if h < y:
                        pole_box = TransformBox((xa, h, za), (1, y-h, 1))
                        fillBlocks(pole_box, alpha.StoneBricks)

        self.__generate_street_lamps(terrain, palette)

        Generator.generate(self, terrain, height_map)  # generate bridges
        print("OK")

    def __generate_street_lamps(self, terrain, districts):
        unlit_array: np.ndarray = np.zeros((self.width, self.length), dtype=np.uint8)
        network = self.__network
        W, L = self.width, self.length
        for dw in range(2, -1, -1):  # dw in (2, 1, 0):
            for road_point in network.road_blocks:  # type: Point
                w0: int = network.get_road_width(road_point) // 2
                x, z = road_point.x, road_point.z
                w1 = w0 + dw
                unlit_array[max(0, x - w1):min(W, x + w1 + 1), max(0, z - w1):min(W, z + w1 + 1)] = dw
        unlit_array[ObstacleMap()[:] > 0] = 0  # lamps don't spawn on obstacles
        # del w0, w1, dw, x, z

        x0, z0 = self.__origin.x, self.__origin.z
        while unlit_array.sum():
            unlit_pos = np.where(unlit_array > 0)
            unlit_idx = np.random.randint(len(unlit_pos[0])) if len(unlit_pos[0]) > 1 else 0
            x, z = unlit_pos[0][unlit_idx], unlit_pos[1][unlit_idx]
            y = terrain.height_map[x, z]
            r = network.get_closest_road_point(Point(x, z))
            h = terrain.height_map[r.x, r.z]
            in_village = districts[x, z] < 1
            if abs(y - h) <= 2:
                if in_village:
                    place_street_lamp(x + x0, y, z + z0, 'oak', h - y)
                else:
                    place_torch_post(x + x0, y, z + z0)
            elif unlit_array[x, z] == 1:
                if in_village:
                    setBlock(Point(x, z, h), alpha.ChiseledStoneBricks)
                    place_street_lamp(x + x0, h, z + z0, 'oak')
                else:
                    setBlock(Point(x, z, h), alpha.StrippedOakWood)
                    place_torch_post(x + x0, h, z + z0)
            else:
                # unsuitable position
                unlit_array[x, z] = 0
                continue

            m = 10
            for dx, dz in filter(lambda dxz: (abs(dxz[0]) + abs(dxz[1])) <= m, itertools.product(range(-m, m+1), range(-m, m+1))):
                if 0 <= (x + dx) < W and 0 <= (z + dz) < L:
                    unlit_array[x+dx, z+dz] = 0

    def __compute_road_at(self, x, z, height_map, districts):
        # type: (int, int, array, object) -> (int, str)
        palette: RoadPalette = city_road_palette if districts[x, z] <= 1 else rusty_road_palette

        # Computes local road height depending of the height of road blocks +-1 block away
        surround_iter = itertools.product(sym_range(x, 1, self.width), sym_range(z, 1, self.length))
        surround_alt = {Point(x1, z1, height_map[x1, z1]) for (x1, z1) in surround_iter if self.__network.is_road(x1, z1)}
        if not surround_alt:
            return height_map[x, z], palette.block

        surround_y = [_.y for _ in surround_alt]
        y = mean(surround_y)
        h = y - min(surround_y)
        if h < .5:
            return int(y), palette.block
        elif h < .84:
            return int(ceil(y)), palette.slab
        else:
            mx, my, mz = mean([p.x for p in surround_alt]), y, mean([p.z for p in surround_alt])
            x_slope = sum((p.x - mx) * (p.y - my) for p in surround_alt)
            z_slope = sum((p.z - mz) * (p.y - my) for p in surround_alt)
            try:
                direction = Direction.of(dx=x_slope, dz=z_slope)
                return int(round(y - 0.33)), BlockAPI.getStairs(palette.stair, facing=direction)
            except AssertionError:
                return int(y), palette.block

    def handle_new_road(self, path):
        """
        Builds necessary bridges and stairs for every new path
        Assumes that it is called before the path is marked as a road in the RoadNetwork
        """
        smoothing_kernel = np.array([-2, 3, 6, 7, 6, 3, -2]) / 21
        if len(path) < 2:
            return

        water_margin = 5
        if any(self.__fluids.is_water(_) for _ in path):
            bridge_start = next(filter(lambda i: self.__fluids.is_water(path[i], water_margin), range(len(path))))
            bridge_end = next(filter(lambda i: i == len(path) or not self.__fluids.is_water(path[i], water_margin), range(bridge_start + 1, len(path) + 1)))

            bridge_start_pt = path[bridge_start]
            bridge_start_pt = Point(bridge_start_pt.x, bridge_start_pt.z, self.__maps.height_map[bridge_start_pt])
            bridge_end_pt = path[bridge_end-1]
            bridge_end_pt = Point(bridge_end_pt.x, bridge_end_pt.z, self.__maps.height_map[bridge_end_pt])
            bridge = Bridge(bridge_start_pt, self.__origin)
            bridge += bridge_end_pt
            if euclidean(bridge_start_pt, bridge_end_pt) >= 3:
                self.children.append(bridge)
            if bridge_start > 0:
                self.handle_new_road(path[:bridge_start])
            if bridge_end < len(path):
                self.handle_new_road(path[bridge_end:])
            return

        # carves the new path if possible and necessary to avoid steep slopes
        path_height: np.ndarray = np.array([self.__maps.height_map[_] for _ in path]).astype(float)
        if len(path_height) > max(path_height) - min(path_height):
            if len(path_height) <= 7:
                # path too short to convolve
                return

            orig_path_height = path_height.copy()
            # while there is a 2 block elevation in the path, smooth path heights
            changed = False
            elevation = abs(np.diff(path_height)).max()
            if elevation > 1:
                path_height[:3] = path_height[:3].mean()
                path_height[-3:] = path_height[-3:].mean()

            while elevation > 1:
                changed = True
                # convolving outside kernel edges works bad
                path_height[3:-3] = np.convolve(path_height, smoothing_kernel, 'valid')
                new_elevation = abs(np.diff(path_height)).max()

                if new_elevation >= elevation:
                    break
                else:
                    elevation = new_elevation

            if changed:
                path_height = [int(round(_)) for _ in path_height]
                self.__maps.height_map.update(path, path_height)
                self.children.append(CarvedRoad(path, orig_path_height, self.__origin))
                for updated_point_index in filter(lambda _: path_height[_] != orig_path_height[_], range(len(path))):
                    point = path[updated_point_index]
                    ObstacleMap().add_obstacle(point)

        # else:
        #     elevation = np.abs(np.diff(path_height))
        #     length = len(elevation)
        #     begin = next(i for i in range(length) if elevation[i] > 1)
        #     end = next(length-i for i in range(length) if elevation[length-i-1] > 1)
        #     self.children.append(RampStairs(path[begin], path[end], self.__maps.height_map))
        #     for _ in range(end-begin-1):
        #         path.pop(begin)


class Bridge(Generator):

    def __init__(self, edge, origin):
        # type: (Point, Point) -> None
        assert isinstance(edge, Point) and isinstance(origin, Point)
        init_box = TransformBox((edge.x, 0, edge.z), (1, 1, 1))
        Generator.__init__(self, init_box, entry_point=edge)
        self.__points = [edge]
        self.__origin = origin  # type: Point

    def __iadd__(self, other):
        assert isinstance(other, Point)
        self.__points.append(other)
        self._box = self._box.union(TransformBox((other.x, 0, other.z), (1, 1, 1)))
        return self

    def generate(self, level, height_map=None, palette=None):
        y0, y1 = self.__points[0].y, self.__points[-1].y
        water_height = min(y0, y1)
        self._box = TransformBox(self._box)
        self.translate(dx=self.__origin.x, dz=self.__origin.z)
        self.__straighten_bridge_points()

        heights = [height_map[_.x, _.z] for _ in self.__points]
        length = len(heights)
        target_bridge_height = water_height + 2.5  # ideal height so that boats can pass under the bridge

        h1 = np.full(length, target_bridge_height)
        h2 = np.array([y0 + (i / 2) if i <= (length / 2) else y1 + ((length-i-1) / 2) for i in range(length)]) + .5
        bridge_height = np.minimum(h1, h2)

        for p, h in zip(self.__points, bridge_height):
            ap = p + self.__origin
            self.place_block(ap.x, h, ap.z)

    @property
    def width(self):
        return abs(self.__points[-1].x - self.__points[0].x)

    @property
    def length(self):
        return abs(self.__points[-1].z - self.__points[0].z)

    def place_block(self, x, y, z):
        if y % 1 == 0:
            b = alpha.StoneBricks
        else:
            b = BlockAPI.getSlab("stone_brick", type="bottom")
        y = ceil(y)

        if self.width > self.length:
            for dz in range(-1, 2):
                setBlock(Point(x, z+dz, y), b)
        else:
            for dx in range(-1, 2):
                setBlock(Point(x+dx, z, y), b)

    def __straighten_bridge_points(self):
        o1, o2 = self.__points[0], self.__points[-1]
        if o1.x == o2.x:
            z1, z2 = sorted((o1.z, o2.z))
            self.__points = [Point(o1.x, z) for z in range(z1, z2 + 1)]
        elif o1.z == o2.z:
            x1, x2 = sorted((o1.x, o2.x))
            self.__points = [Point(x, o1.z) for x in range(x1, x2 + 1)]
        else:
            self.__points = [Point(p[0], p[2]) for p in raytrace((o1.x, 0, o1.z), (o2.x, 0, o2.z))]


class CarvedRoad(Generator):
    def __init__(self, points, heights, origin):
        # type: (List[Point], List[int], Point) -> None
        x_min, x_max = min([_.x for _ in points]), max([_.x for _ in points])
        y_min, y_max = min(heights), max(heights)
        z_min, z_max = min([_.z for _ in points]), max([_.z for _ in points])
        init_box = TransformBox((x_min, y_min, z_min), (x_max + 1 - x_min, y_max + 1 - y_min, z_max + 1 - z_min))
        Generator.__init__(self, init_box, entry_point=points[0])
        self.__points = [_ for _ in points]  # type: List[Point]
        self.__heights = [_ for _ in heights]  # type: List[int]
        self.__origin = origin

    def generate(self, level, height_map=None, palette=None):
        for i in range(len(self.__points)):
            relative_road, ground_height = self.__points[i], self.__heights[i]  # type: Point, int
            absolute_road = relative_road + self.__origin  # rp with absolute coordinates
            road_height = height_map[relative_road.x, relative_road.z]
            height = road_height + 2 - ground_height
            if height < 2:
                road_box = TransformBox((absolute_road.x - 1, road_height + 1, absolute_road.z - 1), (3, 3, 3))
                fillBlocks(road_box, alpha.Air, ground_blocks)


class RoadPalette:
    """
    Samples blocks to generate at road level
    """
    def __init__(self, block_distrib: Dict[str, float], material_distrib: Dict[str, float]):
        """
        Constructor for RoadPalette. Weights are normalized to assert they sum to 1
        :param block_distrib: dict mapping strings (from alpha) to weight
        :param material_distrib: dict mapping strings (materials, ie "stone") to weight
        """
        self.__blocks = list(block_distrib.keys())
        weights = list(block_distrib.values())
        self.__block_prob = [_ / sum(weights) for _ in weights]
        self.__materials = list(material_distrib.keys())
        weights = list(material_distrib.values())
        self.__materials_prob = [_ / sum(weights) for _ in weights]

    def __sample(self, sample_blocks):
        if sample_blocks:
            return choice(self.__blocks, p=self.__block_prob)
        else:
            return choice(self.__materials, p=self.__materials_prob)

    @property
    def block(self):
        """:return: block as str to generate"""
        return self.__sample(True)

    @property
    def slab(self):
        """:return: slab block as str to generate"""
        material = self.__sample(False)
        return f"{material}_slab"

    @property
    def stair(self):
        """:return: stairs block as str to generate"""
        material = self.__sample(False)
        return f"{material}_stairs"


city_road_palette = RoadPalette(
    {alpha.Cobblestone: 0.6, alpha.Gravel: 0.1, alpha.Stone: 0.15,
     alpha.DeadBrainCoralBlock: 0.1, alpha.DeadBubbleCoralBlock: 0.05},
    {alpha.Cobblestone: .4, alpha.StoneBrick: .2,
     alpha.Stone: .3, alpha.Andesite: .2}
)

rusty_road_palette = RoadPalette(
    {alpha.GrassPath: 8, alpha.MossyCobblestone: .7, alpha.MossyStoneBricks: .5, alpha.Cobblestone: .3},
    {'oak': 4, alpha.MossyCobblestone: 1, alpha.StoneBrick: .5, "mossy_stone_brick": .5}
)
