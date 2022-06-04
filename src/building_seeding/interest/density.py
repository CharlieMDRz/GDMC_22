import numpy as np

from building_seeding import Districts
from building_seeding.interest import attraction_repulsion


def density(shape, districts, lambdas):
    # type: (tuple, Districts, tuple) -> np.ndarray
    assert len(shape) == 2 and len(lambdas) == 3
    interest_matrix = np.vectorize(lambda d: attraction_repulsion(d, *lambdas))(districts[:])
    return interest_matrix

