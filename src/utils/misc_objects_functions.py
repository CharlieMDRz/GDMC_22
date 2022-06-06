import itertools
import logging
import time
from os.path import realpath, sep
from random import random
from typing import Tuple, Iterable

import cv2
from matplotlib import pyplot as plt
import numba
import numpy as np
from sortedcontainers import SortedList

__all__ = [
    'argmax',
    'argmin',
    'bernouilli',
    'get_project_path',
    'in_limits',
    'index_argmin',
    'log_exec_time',
    'mean',
    'numba_list',
    'plot_map',
    'pos_bound',
    'raytrace',
    'sections',
    'Singleton',
    'sym_range',
]


def bernouilli(success_rate=.5):
    # type: (float) -> bool
    return random() <= success_rate


def get_project_path():
    this_path = realpath(__file__)
    proj_path = sep.join(this_path.split(sep)[:-2])
    return proj_path


def sections(items: Iterable) -> Iterable[Tuple]:
    """
    Gets tuples of consecutive items of the input iterable
    :param items: sequence of objects
    :return: iterable over pairs of consecutive items
    """
    cur_items, nxt_items = itertools.tee(items)
    next(nxt_items)
    return zip(cur_items, nxt_items)


def argmin(values, key=None):
    if type(values) != list:
        values = list(values)
    if not values:
        return None
    if key is None:
        return index_argmin(numba_list(values))

    def rec_argmin(sub_values):
        if len(sub_values) == 1:
            return sub_values[0], key(sub_values[0])
        else:
            L = len(sub_values) // 2
            v0, k0 = rec_argmin(sub_values[:L])
            v1, k1 = rec_argmin(sub_values[L:])
            return (v0, k0) if k0 <= k1 else (v1, k1)

    return rec_argmin(values)[0]


def numba_list(iterator: Iterable) -> numba.typed.List:
    typed_list = numba.typed.List()
    for e in iterator:
        typed_list.append(e)
    return typed_list


def log_exec_time(start_time: float, msg: str):
    end_time: float = time.time()
    exec_time: int = int((end_time - start_time) * 1000)
    logging.info(f"{msg} took {exec_time} ms")


@numba.njit()
def index_argmin(values: numba.typed.List):
    idx_min, val_min = 0, values[0]
    for idx, val in enumerate(values):
        if val < val_min:
            idx_min, val_min = idx, val
    return idx_min


def argmax(values, key=None):
    if key is None:
        tmp = values
        values = list(range(len(tmp)))
        key = lambda x: tmp[x]

    def rec_argmax(sub_values):
        if len(sub_values) == 1:
            return sub_values[0], key(sub_values[0])
        else:
            L = len(sub_values) // 2
            v0, k0 = rec_argmax(sub_values[:L])
            v1, k1 = rec_argmax(sub_values[L:])
            return (v0, k0) if k0 >= k1 else (v1, k1)

    return rec_argmax(values)[0]


def mean(iterable):
    iter_count = 0
    iter_sum = None
    for _ in iterable:
        iter_sum = (iter_sum + _) if iter_count else _
        iter_count += 1
    return iter_sum / iter_count


def pos_bound(v, vmax=None):
    if v < 0:
        return 0
    if vmax and v >= vmax:
        return vmax
    return v


def sym_range(v, dv, vmax=None):
    """
    Range of length 2 * dv centered in v. specify vmax to bound upper value
    2 * dv must be an integer
    sym_range(3, 1) = [2, 3, 4]
    sym_range(3, 1.5) = [1, 2, 3, 4]
    """
    if dv % 1 == 0:
        v0, v1 = v - dv, v + dv + 1
    elif dv % 1 == .5:
        idv = int(dv + 0.5)
        v0, v1 = v - idv, v + idv
    else:
        raise ValueError("Expected dv to be half and integer, found {}".format(dv))
    v0, v1 = pos_bound(v0, vmax), pos_bound(v1, vmax)
    return range(int(v0), int(v1))


@numba.njit
def in_limits(xyz0: Tuple[int, int, int], width, length):
    x0, y0, z0 = xyz0
    return 0 <= x0 < width and 0 <= z0 < length


# returns an array of blocks after raytracing from (x1,y1,z1) to (x2,y2,z2)
# this uses Bresenham 3d algorithm, taken from a modified version written by Bob Pendleton
def raytrace(xyz1, xyz2):
    (x1, y1, z1) = xyz1
    (x2, y2, z2) = xyz2
    output = []

    x2 -= 1
    y2 -= 1
    z2 -= 1
    point = [x1, y1, z1]

    dx = x2 - x1
    dy = y2 - y1
    dz = z2 - z1
    x_inc = -1 if dx < 0 else 1
    l = abs(dx)
    y_inc = -1 if dy < 0 else 1
    m = abs(dy)
    z_inc = -1 if dz < 0 else 1
    n = abs(dz)
    dx2 = l << 1
    dy2 = m << 1
    dz2 = n << 1

    if l >= m and l >= n:
        err_1 = dy2 - l
        err_2 = dz2 - l
        for i in range(l):
            np = (point[0], point[1], point[2])
            output.append(np)
            if err_1 > 0:
                point[1] += y_inc
                err_1 -= dx2

            if err_2 > 0:
                point[2] += z_inc
                err_2 -= dx2

            err_1 += dy2
            err_2 += dz2
            point[0] += x_inc

    elif m >= l and m >= n:
        err_1 = dx2 - m
        err_2 = dz2 - m
        for i in range(m):
            np = (point[0], point[1], point[2])
            output.append(np)
            if err_1 > 0:
                point[0] += x_inc
                err_1 -= dy2

            if err_2 > 0:
                point[2] += z_inc
                err_2 -= dy2

            err_1 += dx2
            err_2 += dz2
            point[1] += y_inc

    else:
        err_1 = dy2 - n
        err_2 = dx2 - n
        for i in range(n):
            np = (point[0], point[1], point[2])
            output.append(np)
            if err_1 > 0:
                point[1] += y_inc
                err_1 -= dz2

            if err_2 > 0:
                point[0] += x_inc
                err_2 -= dz2

            err_1 += dy2
            err_2 += dx2
            point[2] += z_inc

    np = (point[0], point[1], point[2])
    output.append(np)
    return output


class Singleton(type):
    """
    Singleton metaclass
    """
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]


def plot_map(array, do_normalize=True):
    fig, ax = plt.subplots()

    def normalize(array):
        """**Normalize the array to contain values from 0 to 1**."""
        return (array - array.min()) / (array.max() - array.min())

    if do_normalize:
        array = (normalize(array) * 255).astype(np.uint8)
    plt_image = cv2.cvtColor(array, cv2.COLOR_BGR2RGB)
    ax.imshow(plt_image)
    return ax


if __name__ == '__main__':
    l = [-1, 3, -5, 9, 6]
    print(l, argmin(l), argmax(l))
    print(argmax(l, lambda v: v ** 2))

    sl = SortedList(l, lambda v: v ** 2)
    assert sl[3] == 6
    sl.append(5.5)
    assert sl[3] == 5.5
    print(sl[:])
