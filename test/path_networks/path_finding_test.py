from gdpc.worldLoader import WorldSlice
import matplotlib.pyplot as plt

from path_networks import PathFinder, GridGraph, road_build_cost, ObstacleMap, RailRoadGraph
from path_networks.rail_network import rail_road_build_cost, RailNetwork
from terrain import TerrainMaps, HeightMap
from utils import BuildArea, BlockAPI, Position, getBlockRelativeAt, plot_map, Direction
from utils.algorithms.fast_astar import fast_a_star

TARGETED_BLOCK = BlockAPI.blocks.DiamondBlock


def targeted_block_at(pos: Position):
    hm: HeightMap = terrain.height_map
    y = hm.upper_height(pos)
    block_state: str = terrain.level.getBlockAt(pos.abs_x, y, pos.abs_z)
    return block_state.replace("minecraft:", '').startswith(TARGETED_BLOCK)


def main():
    ObstacleMap.from_terrain(terrain)
    source, target = list(filter(targeted_block_at, terrain.area.building_positions()))[:2]

    path_finder = PathFinder(6, GridGraph(True, step=1, cost=road_build_cost), GridGraph(True, step=6, cost=road_build_cost))
    rail_finder = PathFinder(7, RailRoadGraph(7, 15, cost=rail_road_build_cost), GridGraph(True, step=7, cost=road_build_cost))
    rail_source = terrain.rail_network.add_station(source, Direction.of(*(target-source).xyz)).connect(target).asPosition
    ax = plot_map(terrain.height_map)
    paths = []
    paths.extend(path_finder.getRoughPath(target, source) for _ in range(1))
    paths.extend(path_finder.getPath(target, source) for _ in range(1))
    paths.extend(rail_finder.getPath(rail_source, target) for _ in range(1))
    paths.extend(fast_a_star(source, target, road_build_cost) for _ in range(1))
    for path in paths:
        path_x = [p.x for p in path]
        path_z = [p.z for p in path]
        ax.plot(path_z, path_x)
    plt.legend(["rough path", "A* path", "Rails"])
    if not perf:
        plt.show()


perf = False

if __name__ == '__main__':
    terrain: TerrainMaps = TerrainMaps.request()

    if perf:
        from pstats import Stats, SortKey
        import cProfile
        stats: Stats = cProfile.run(f"main()", sort=SortKey.CUMULATIVE)
    else:
        main()
