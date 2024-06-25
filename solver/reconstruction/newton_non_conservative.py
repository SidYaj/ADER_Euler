import numpy as np
import numpy.typing as npt
from typing import List, Tuple, Dict, Callable
from scipy.integrate import fixed_quad # type: ignore

class NewtonPolynomial: # constructs a Newton polynomial for a given stencil S about the ith cell

    def __init__(self, xArray: npt.NDArray, qArray: npt.NDArray, S: list[int], i: int): 
        self.xArray = xArray
        self.qArray = qArray
        self.S = S
        self.degree = len(S) - 1
        self.i = i
        self.coefficients()
        self.derivatives()

    def coefficients(self) -> None:
        A_list: List[List[float]] = []
        for row in range(self.degree + 1):
            idx = self.S[row]
            # A_row: List[float] = []
            # for column in range(self.degree + 1):
            #     f: Callable[[float], float] = lambda x: (x - self.xArray[self.i]) ** column
            #     integral_f: float = fixed_quad(f, self.xArray[idx] - 0.5 * dx, self.xArray[idx] + 0.5 * dx, n=self.degree + 1)[0]
            #     A_row.append(integral_f / dx)
            A_row = [(self.xArray[idx] - self.xArray[self.i]) ** column for column in range(self.degree + 1)] # list comprehension form (to be tested) 
            A_list.append(A_row)
        A: npt.NDArray = np.array(A_list)
        self.c: npt.NDArray = np.linalg.solve(A, self.qArray[self.S])
    
    def derivatives(self) -> None:
        d: npt.NDArray = np.zeros((self.degree + 1, self.degree + 1)) # matrix where row j contains the coefficients of the jth derivative
        d[0, :] = self.c
        for j in range(1, self.degree + 1):
            for k in range(self.degree + 1 - j):
                d[j, k] = (k + 1) * d[j - 1, k + 1]
        self.d: npt.NDArray = d

    def getValue(self, x: float, k: int = 0) -> float: # gets the value of the kth derivative of the Newton polynomial as a function of x
        c: npt.NDArray = self.d[k, :]
        sum: float = 0
        for j in range(self.degree + 1):
            sum += c[j] * (x - self.xArray[self.i]) ** j
        return sum
