from random import choice

from gdpc import direct_interface
from gdpc.interface import globalDecay, global2buildlocal, Interface, checkOutOfBounds


class InterfacePatch(Interface):
    """**Provides tools for interacting with the HTML interface**.

    All function parameters and returns are in local coordinates.
    """

    def __init__(self, x=0, y=0, z=0, buffering=False, bufferlimit=1024, caching=False, cachelimit=8192):
        """**Initialise an interface with offset and buffering**."""
        super().__init__(x, y, z, buffering, bufferlimit, caching, cachelimit)
        self.__buffering = buffering

    def placeBlock(self, x, y, z, block, replace=None, doBlockUpdates=-1, customFlags=-1):
        """**Place a block in the world depending on buffer activation**.

        Takes local coordinates, works with local and global coordinates
        """
        flags = doBlockUpdates, customFlags
        from gdpc.toolbox import isSequence
        if isinstance(replace, str):
            if self.getBlock(x, y, z) != replace:
                return '0'
        elif isSequence(replace) and self.getBlock(x, y, z) not in replace:
            return '0'

        if not isinstance(block, str) and isSequence(block):
            block = choice(block)

        if self.__buffering:
            response = self.placeBlockBuffered(x, y, z, block, self.bufferlimit, *flags)
        else:
            response = self.placeBlockDirect(x, y, z, block, *flags)

        # switch to global coordinates
        x, y, z = self.local2global(x, y, z)
        if self.caching:
            self.cache[(x, y, z)] = block
        # mark block as decayed
        if not checkOutOfBounds(x, y, z) and globalDecay is not None:
            x, y, z = global2buildlocal(x, y, z)
            globalDecay[x][y][z] = True

        return response

    def sendBlocks(self, x=0, y=0, z=0, retries=5):
        """**Send the buffer to the server and clear it**.

        Since the buffer contains global coordinates
            no conversion takes place in this function
        """
        if self.buffer == []:
            return '0'
        response = direct_interface.sendBlocks(self.buffer, x, y, z,
                                 retries, *self.bufferblockflags).split('\n')
        if all(map(lambda val: val.isnumeric(), response)):  # no errors
            self.buffer = []
            return str(sum(map(int, response)))
        else:
            # print(f"{TCOLORS['orange']}Warning: Server returned error upon "
            #       f"sending block buffer:\n\t{TCOLORS['CLR']}{repr(response)}")
            try:
                error_msg = next(res for res in response if not res.isnumeric())
                invalid_block = error_msg.split("'")[1]
                print("Invalid block", invalid_block)
                self.buffer = [_ for _ in self.buffer if invalid_block.split(':')[-1] not in _[-1]]
            except IndexError:
                print(response)
