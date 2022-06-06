import logging
import traceback
from typing import Union, List

from gdpc import interface
import numpy as np

from utils import BoundingBox, Position, Point
from utils.block_utils import BuildArea


class Structure(BoundingBox):
    def __init__(self, origin, size):
        super().__init__(origin, size)
        self.__priority = np.zeros(size, dtype=np.int32)
        self.interface: interface.Interface = interface.Interface(buffering=True, caching=True)

    @classmethod
    def from_box(cls, box: BoundingBox):
        return cls(box.origin, box.size)

    def set(self, pos: Position, block_state: Union[str, List[str]], priority: int = 1, force: bool = False, **kwargs):
        """
        Place a block
        :param pos: position to place to
        :param block_state: blockstate to place
        :param priority: priority of the blockstate, will only place if no block with a >= priority has been set
        :param force: if True, and the current block block at pos has the same priority, will place
        :param kwargs: args for the Interface.placeBlock method
        :return:
        """
        # Normalize block state: could have more options
        if "glazed_terracotta" in block_state:
            rotation_id = (pos.x % 2) * 2 + (pos.z % 2)
            rotations = ['north', 'west', 'east', 'south']
            block_state += f"[facing={rotations[rotation_id]}]"

        # switch to relative coordinates if given as point with absolute coords
        if not isinstance(pos, Position):
            pos = Position(pos.x, pos.z, pos.y, True)

        if pos.abs_xyz not in self:
            logging.debug(f"Trying to set block outside build area ! @{pos}")
            # traceback.print_stack()
            return
        prev_priority = self.__priority[pos.xyz]

        if priority < prev_priority or (priority == prev_priority and not force):
            return  # not enough priority to replace current block
        if kwargs.get('replace', False):
            blocks_to_replace = kwargs.get('replace')

        self.__set(pos, block_state, priority)

    def __set(self, position: Position, block_state: str, priority: int):
        self.interface.placeBlock(*position.abs_xyz, block_state)
        self.__priority[position.xyz] = priority

    def fill(self, box: BoundingBox, blockstate: Union[str, List[str]], priority: int = 1, force=False, **kwargs):
        for x, y, z, in box.positions:
            p = Point(x, z, y)
            self.set(p, blockstate, priority, force, **kwargs)

    @property
    def altered_positions(self):
        return zip(*np.where(self.__priority > 0))

    @property
    def origin(self) -> Position:
        return Position(self.minx, self.minz, self.miny)

    def dump(self):
        self.interface.sendBlocks()


AREA_STRUCTURE: Structure = Structure((BuildArea().x, 0, BuildArea().z), (BuildArea().width, 256, BuildArea().length))
