"""
Main call module for my GDMC submission, includes a command line parser and the outline of the algorithm
"""
import argparse
import logging
import multiprocessing as mp
import traceback
from os import sep
import time
import sys
import warnings
from typing import Dict, Union, Callable, List


warnings.filterwarnings("ignore")

# Managing dependencies and Python path
this_path = str(__file__)  # path to this file
this_path = sep.join(this_path.split(sep)[:-1])  # path to the directory where
if not this_path: this_path = "."
sys.path.insert(0, this_path + sep + 'src')  # path to our code

console_log = logging.StreamHandler(sys.stdout)
console_log.setLevel(logging.INFO)
file_log = logging.FileHandler('settlement_log_{}.log'.format(time.strftime('%Y-%m-%d_%H-%M-%S', time.gmtime())))
logging.basicConfig(
    level=logging.INFO,
    handlers=[console_log, file_log]
)

from building_seeding import district
from settlement import Settlement
from terrain import TerrainMaps, ObstacleMap
from utils import log_exec_time


def main(districts=None, seeding=None, parcels=None, generation=None, visualize=False, undo=False) -> None:
    print("Hello Settlers!")
    # get & parse building zone
    terrain: TerrainMaps = TerrainMaps.request()
    t0 = time.time()
    ObstacleMap.from_terrain(terrain)  # initialize obstacle map from the terrain
    log_exec_time(t0, "Computing obstacle map")
    settlement = Settlement(terrain)

    try:
        if districts:
            districts(settlement, visualize=visualize)
        else: return
        settlement.init_road_network()
        settlement.compute_rail_network()

        if seeding:
            # define buildings list and seed them
            seeding(settlement, do_visu=visualize)
        else: return
        settlement.clean_road_network()

        if parcels:
            # define parcels around seeds
            parcels(settlement, visualize=visualize)
        else: return

        if generation:
            # build buildings on parcels
            settlement.terraform()
            generation(settlement)
        else: return

        # Optional erasing of the generated settlement
        if undo:
            do_undo = input("Undo ? [y]/n").lower()
            if do_undo in {"", "y"}:
                terrain.undo()
    except Exception:
        print("Uncaught exception, undoing gen for safety")
        print(traceback.format_exc())
        terrain.undo()


def get_parser() -> argparse.ArgumentParser:
    """
    Parser for my gdmc submission
    :return: argument parser
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", "-s", nargs='+', type=str, default=["D1", "S0", "G", "P"])
    parser.add_argument("--visualize", "-v", action="store_true", help="Export visualization maps during run")
    parser.add_argument("--time", "-T", type=int, default=1200, help="Time limit in seconds for the whole run, negative value for no limit")

    run_modes = parser.add_mutually_exclusive_group()
    run_modes.add_argument("--undo", "-u", action="store_true", help="Undo generation after user input")
    run_modes.add_argument("--perf", "-p", action="store_true", help="Run code with profiler")
    return parser


def get_generation_options(step_list: List[str]) -> Dict[str, Union[Callable, None]]:
    """
    For all the algorithm steps, selects the algorithm to use from the args of the command line
    :param step_list: command line "steps" arg
    :return: mapping step -> function
    """
    # Mapping between options in the command line to parts of the generation algorithm
    _step_to_option = {
        "D": "districts",
        "S": "seeding",
        "P": "parcels",
        "G": "generation"
    }

    # Options for the steps arguments
    _steps_dictionary = {
        "D": {
            "0": (district.build_default_district, "Generate a single district all over the build area"),
            "1": (Settlement.build_districts, "Apply Kmeans algorithm to select multiple villages (2021 submission)")
        },

        "S": {
            "no": (lambda *o, **kw: None, "Skips seeding process"),
            "0": (Settlement.build_skeleton, "Iteratively seed positions for parcels of specific type")
        },

        "P": {
            "": (Settlement.define_parcels, "Iteratively expand parcels one direction after another")
        },

        "G": {
            "": (Settlement.generate, "Generate parcels and road network")
        }
    }

    # Functions to be used during generation, will be passed to the main function
    _gen_options = {
        "districts": None,
        "seeding": None,
        "parcels": None,
        "generation": None
    }

    assert step_list
    for step in step_list:
        step, step_variation = step[0], step[1:]
        assert step in _step_to_option
        gen_step = _step_to_option[step]
        logging.info(f"Selecting {gen_step} algorithm...")
        func, desc = _steps_dictionary[step][step_variation]
        _gen_options[gen_step] = func
        logging.info(f"\tWill use '{func.__name__}': {desc}")

    try:
        unspec_step = next(step for step in _gen_options if _gen_options[step] is None)
        print(f"The algorithm will stop right before the {unspec_step} step: no option specified")
    except StopIteration:
        print("All steps are correctly configured, settlers are ready to move in !")
    finally:
        return _gen_options


if __name__ == '__main__':

    t0 = time.time_ns()
    # Parse the command line
    args = get_parser().parse_args()
    # Inspect the steps argument
    gen_options = get_generation_options(args.steps)
    gen_options["visualize"] = args.visualize
    gen_options["undo"] = args.undo

    if args.perf:
        print("Running profiler mode...")
        # code profiler
        from pstats import Stats, SortKey
        import cProfile

        stats: Stats = cProfile.run(f"main(**gen_options)", sort=SortKey.CUMULATIVE)

    elif args.time > 0 and not args.undo:
        # run code with time limit
        print(f"Running default mode... Will time-out after {args.time} seconds")
        p = mp.Process(target=main, name="gdmc_run", kwargs=gen_options)
        p.start()
        p.join(args.time)

        if p.is_alive():
            print(f"Reached time limit of {args.time}s, timing-out generation !")
            p.terminate()
            p.join()

    else:
        print("Running endless mode...")
        main(**gen_options)

    t1 = time.time_ns()
    print(f"Generation successfully completed in {(t1 - t0)/10**9:0.3f} seconds !")
