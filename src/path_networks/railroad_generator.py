import numpy as np

from generation import Generator
from generation.structure import AREA_STRUCTURE
from utils import TransformBox, Position, BlockAPI, BoundingBox, clear_tree_at, Point


class RailRoadGenerator(Generator):
    def __init__(self, tracks_array, **kwargs):
        super().__init__(TransformBox(), **kwargs)
        self.__is_path: np.ndarray[bool] = tracks_array

    def generate(self, level, height_map=None, palette=None):
        super().generate(level, height_map, palette)
        padded_path_array = np.pad(self.__is_path, 1).astype(int) * 2  # 2 -> spacing between rails

        padded_rail_array = np.zeros(padded_path_array.shape)
        for x, z in zip(*np.where(padded_path_array == 2)):
            padded_rail_array[(x-1): (x+2), (z-1): (z+2)] = 1  # 1 -> rails

        padded_rail_array[padded_path_array == 2] = 0  # un-rail path blocks
        rail_array = padded_rail_array[1:-1, 1:-1].astype(bool)  # un-pad array

        for x, z in zip(*np.where(rail_array)):
            rail_texture = self.rail_texture_at(rail_array, x, z)
            rail_position: Position = Position(x, z, height_map[x, z] + 1)
            ballast_box = BoundingBox(
                (rail_position.abs_x - 1, rail_position.y - 1, rail_position.abs_z - 1), (3, 1, 3)
            )
            clear_tree_at(level, Point(rail_position.abs_x, rail_position.abs_z))
            AREA_STRUCTURE.fill(ballast_box, BlockAPI.blocks.Granite, 1000)
            AREA_STRUCTURE.set(rail_position, rail_texture, 1002)

    @staticmethod
    def rail_texture_at(is_rail: np.ndarray, i: int, j: int):
        NORTH = 1
        EAST = 2
        SOUTH = 4
        WEST = 8

        rail_textures = {
            EAST + WEST: 'east_west',
            EAST: 'east_west',
            WEST: 'east_west',
            NORTH + SOUTH: 'north_south',
            NORTH: 'north_south',
            SOUTH: 'north_south',
            NORTH + EAST: 'north_east',
            NORTH + WEST: 'north_west',
            SOUTH + EAST: 'south_east',
            SOUTH + WEST: 'south_west'
        }

        goes_east = EAST if (i+1 < is_rail.shape[0] and is_rail[i + 1, j]) else 0
        goes_west = WEST if (i >= 0 and is_rail[i - 1, j]) else 0
        goes_south = SOUTH if (j+1 < is_rail.shape[1] and is_rail[i, j + 1]) else 0
        goes_north = NORTH if (j >= 0 and is_rail[i, j - 1]) else 0

        key = goes_east + goes_west + goes_south + goes_north
        try:
            shape = rail_textures[key]
            return f"rail[shape={shape}]"
        except KeyError:
            print(f"No rail texture for key {key}")
            return BlockAPI.blocks.Air
