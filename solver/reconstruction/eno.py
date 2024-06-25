import numpy as np
import numpy.typing as npt
from scipy.integrate import fixed_quad # type: ignore
from typing import List, Tuple, Callable
from solver.reconstruction.polynomial_matrices import PolynomialMatrices
from solver.reconstruction.functions import dividedDifference
from icecream import ic # type: ignore
from numba import njit # type: ignore
from sympy.core import Expr

@njit
def getStencil(degree: int, xArray: npt.NDArray, qArray: npt.NDArray) -> List[int]:
    S: List[int] = [degree]
    for _ in range(degree):
        idxLeft: List[int] = [S[0] - 1] + S
        idxLeftArray = np.array(idxLeft)
        idxRight: List[int] = S + [S[-1] + 1]
        idxRightArray = np.array(idxRight)
        ddLeft = dividedDifference(xArray[idxLeftArray], qArray[idxLeftArray])
        ddRight = dividedDifference(xArray[idxRightArray], qArray[idxRightArray])
        if (abs(ddLeft) < abs(ddRight)):
            S = idxLeft
        else:
            S = idxRight
    return S

class ENO:
    
    def __init__(self, xArray: npt.NDArray, qArray: npt.NDArray, degree: int, polyMat: PolynomialMatrices) -> None:

        self.xArray = xArray
        self.qArray = qArray
        self.degree = degree
        self.stencil = getStencil(degree, xArray, qArray)
        self.shiftedStencil = tuple([idx - (degree) for idx in self.stencil])
        self.polyMat = polyMat
        self.conservativeCoefficients()
        self.derivatives()

    def conservativeCoefficients(self) -> None:
        A_inv: npt.NDArray = self.polyMat.getMatrixInverse(self.shiftedStencil)
        self.c: npt.NDArray = A_inv @ self.qArray[np.array(self.stencil)]

    
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
            sum += c[j] * (x - self.xArray[self.degree]) ** j
        return sum 