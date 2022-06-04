import logging
import traceback
from typing import Union, List

from gdpc import interface
from numpy import zeros, int32

from utils import BoundingBox, Position, Point
from utils.block_utils import BuildArea


class Structure(BoundingBox):
    def __init__(self, origin, size):
        super().__init__(origin, size)
        self.__priority = zeros(size, dtype=int32)
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
        if isinstance(pos, Position):
            absolute_point = Point(pos.abs_x, pos.abs_z, pos.y)
            relative_point = pos
        else:
            pos: Point
            absolute_point = pos
            relative_point = pos - self.origin
        if absolute_point.xyz not in self:
            logging.debug(f"Trying to set block outside build area ! @{absolute_point}")
            # traceback.print_stack()
            return
        prev_priority = self.__priority[relative_point.xyz]

        if (priority > prev_priority) or (force and priority == prev_priority):
            x, y, z = absolute_point.xyz
            self.interface.placeBlock(x, y, z, block_state, **kwargs)
            self.__priority[relative_point.xyz] = priority

    def fill(self, box: BoundingBox, blockstate: Union[str, List[str]], priority: int = 1, force=False, **kwargs):
        for x, y, z, in box.positions:
            p = Point(x, z, y)
            self.set(p, blockstate, priority, force, **kwargs)

    @property
    def origin(self) -> Position:
        return Position(self.minx, self.minz, self.miny)

    def dump(self):
        self.interface.sendBlocks()


AREA_STRUCTURE: Structure = Structure((BuildArea().x, 0, BuildArea().z), (BuildArea().width, 256, BuildArea().length))
