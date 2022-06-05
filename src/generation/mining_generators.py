from itertools import product

from generation import Generator
from generation.generators import MaskedGenerator
from utils import Point, setBlock, BlockAPI


class MineGenerator(MaskedGenerator):
    def generate(self, level, height_map=None, palette=None):
        for x, z in product(range(self.width), range(self.length)):
            if self.is_masked(x, z):
                pos = Point(x + self.origin.x, z + self.origin.z, height_map[x, z] + 1)
                # setBlock(pos, BlockAPI.blocks.DiamondBlock)
        super().generate(level, height_map, palette)
