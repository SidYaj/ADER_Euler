import numpy.typing as npt
from typing import Callable
from numba import njit # type: ignore

@njit
def dividedDifference(x: npt.NDArray, q: npt.NDArray):
    
    n = len(x)

    for j in range(1, n):
        for i in range(n - j):
            q[i] = (q[i + 1] - q[i]) / (x[i + j] - x[i])

    return q[0]