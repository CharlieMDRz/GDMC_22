import logging
import sys

import matplotlib.pyplot as plt
import tqdm

from pathfinding.path_finder import PathFinder
from pathfinding.rail_network import rail_road_build_cost, RailRoadGraph
from pathfinding.road_network import road_build_cost
from terrain import TerrainMaps, HeightMap, ObstacleMap
from test.utils import targeted_block_at
from utils import BlockAPI, Position, plot_map, Direction
from utils.algorithms.fast_astar import fast_a_star
from utils.algorithms.graphs import GridGraph
from utils.algorithms.hierarchical_astar import hierarchical_astar

console_log = logging.StreamHandler(sys.stdout)
console_log.setLevel(logging.INFO)
logging.basicConfig(level=logging.INFO, handlers=[console_log])


TARGETED_BLOCK = BlockAPI.blocks.DiamondBlock


def main():
    logging.info("Running path finding tests")
    ObstacleMap.from_terrain(terrain)
    target_positions = []
    for position in tqdm.tqdm(terrain.area.building_positions()):
        if targeted_block_at(terrain, position, [TARGETED_BLOCK]):
            target_positions.append(position)
    print(target_positions)
    source, target = target_positions[:2]

    path_finder = PathFinder(6, GridGraph(True, step=1, cost=road_build_cost), GridGraph(True, step=6, cost=road_build_cost))
    smol_path_finder = PathFinder(1, None, GridGraph(True, step=1, cost=road_build_cost))
    rail_finder = PathFinder(7, RailRoadGraph(7, 15, cost=rail_road_build_cost), GridGraph(True, step=7, cost=road_build_cost))
    rail_source = terrain.rail_network.add_station(source, Direction.of(*(target-source).xyz)).connect(target).pos.asPosition
    ax = plot_map(terrain.height_map)
    paths, legend = [], []
    legend = sum(([lbl] * count for (lbl, count) in zip(LBL_LIST, ITER_LIST)), [])
    print(legend)
    paths.extend(path_finder.getRoughPath(target, source) for _ in range(BIG_STEP_DIJKSTRA_ITER))
    paths.extend(smol_path_finder.getRoughPath(target, source) for _ in range(SMOL_STEP_DIJKSTRA_ITER))
    paths.extend(path_finder.getPath(target, source) for _ in range(DIJK_ASR_ITER))
    paths.extend(rail_finder.getPath(rail_source, target) for _ in range(RAIL_PTH_ITER))
    paths.extend(fast_a_star(source, target, road_build_cost) for _ in range(FAST_ASR_ITER))
    for _ in range(HRCH_ASR_ITER):
        hierarchical_paths = hierarchical_astar(source, target, road_build_cost, True)
        paths.extend(hierarchical_paths)
        legend.extend(['HrchA*' for _ in range(len(paths)-1)])
    for path in paths:
        path_x = [p.x for p in path]
        path_z = [p.z for p in path]
        ax.plot(path_z, path_x)
    plt.legend(legend)
    if not perf:
        plt.show()


perf = False
BIG_STEP_DIJKSTRA_ITER = 1
SMOL_STEP_DIJKSTRA_ITER = 1
DIJK_ASR_ITER = 1
RAIL_PTH_ITER = 0
FAST_ASR_ITER = 0
HRCH_ASR_ITER = 1

ITER_LIST = [BIG_STEP_DIJKSTRA_ITER, SMOL_STEP_DIJKSTRA_ITER, DIJK_ASR_ITER, RAIL_PTH_ITER, FAST_ASR_ITER, HRCH_ASR_ITER]
LBL_LIST = ['RoughDijk', 'Dijkstra', 'PathFinder', 'Rails', 'FastA*', 'HrchA*']

if __name__ == '__main__':
    terrain: TerrainMaps = TerrainMaps.request()

    if perf:
        from pstats import Stats, SortKey
        import cProfile
        stats: Stats = cProfile.run(f"main()", sort=SortKey.CUMULATIVE)
    else:
        main()
