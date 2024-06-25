import numpy.typing as npt
import numpy as np
from typing import List, Tuple
from icecream import ic # type: ignore
from solver.reconstruction.polynomial_matrices import PolynomialMatrices
from solver.reconstruction.functions import dividedDifference
from numba import njit # type: ignore

# Uses ENO and centred stencil closest to ENO

@njit
def getStencils(i: int, degree: int, xArray: npt.NDArray, qArray: npt.NDArray) -> Tuple[List[int], List[int]]:
    S: List[int] = [i]
    for _ in range(degree - 1): # first find AENO stencil
        idxLeft = [S[0] - 1] + S
        idxLeftArray = np.array(idxLeft)
        idxRight = S + [S[-1] + 1]
        idxRightArray = np.array(idxRight)
        ddList = [dividedDifference(xArray[idxLeftArray], qArray[idxLeftArray]), dividedDifference(xArray[idxRightArray], qArray[idxRightArray])]
        if (abs(ddList[0]) < abs(ddList[1])):
            S = idxLeft
        else:
            S = idxRight
    
    SL = [S[0] - 1] + S # left and right stencils for AENO
    SR = S + [S[-1] + 1]

    S_AENO = (SL, SR)

    S_ENO = SL if abs(dividedDifference(xArray[np.array(SL)], qArray[np.array(SL)])) < abs(dividedDifference(xArray[np.array(SR)], qArray[np.array(SR)])) else SR
    # S_ENO = S

    if degree % 2 == 0:
        S_CENT = [idx + i for idx in range(-degree // 2, degree // 2 + 1)]
        if S_ENO == S_CENT:
            return S_AENO
        else:
            return S_ENO, S_CENT
    
    else:
        S_LEFT = [idx + i for idx in range(-(degree + 1) // 2, (degree - 1) // 2 + 1)]
        S_RIGHT = [idx + i for idx in range(-(degree - 1) // 2, (degree + 1) // 2 + 1)]

        if S_ENO == S_LEFT or S_ENO == S_RIGHT:
            return S_LEFT, S_RIGHT
        else:
            leftDist = abs(S_ENO.index(i) - S_LEFT.index(i))
            rightDist = abs(S_ENO.index(i) - S_RIGHT.index(i))
            if leftDist < rightDist:
                return S_ENO, S_LEFT
            else:
                return S_ENO, S_RIGHT    

class NAenoStencil:
    
    def __init__(self, xArray: npt.NDArray, qArray: npt.NDArray, degree: int):
        self.xArray = xArray
        self.qArray = qArray
        self.degree = degree

    def stencils(self, i: int) -> Tuple[List[int], List[int]]:
        return getStencils(i, self.degree, self.xArray, self.qArray)

                
@njit(error_model='numpy')
def omega(a_L: float, a_R: float, n=3) -> float:
    ll = np.float64(a_L)
    rr = np.float64(a_R)
    x = np.abs(ll/rr)
    if np.isnan(x):
        s = 1.0
    else:
        s = x

    s_inv = 1 / s

    w = np.exp(-(s ** n)) - np.exp(-(s_inv ** n))

    return w

class NAenoPolynomial:
    
    def __init__(self, xArray: npt.NDArray, qArray: npt.NDArray, S_ENO: List[int], S_CENT: List[int], i: int, polyMat: PolynomialMatrices): 
        self.xArray = xArray
        self.qArray = qArray
        self.S_ENO = S_ENO
        self.S_CENT = S_CENT
        self.degree = len(S_ENO) - 1
        self.i = i
        self.conservativeCoefficients(polyMat)
        self.derivatives()

    def conservativeCoefficients(self, polyMat: PolynomialMatrices) -> None:

        S_ENO_shifted = tuple([idx - self.i for idx in self.S_ENO])
        A_inv_ENO: npt.NDArray = polyMat.getMatrixInverse(S_ENO_shifted)
        c_ENO: npt.NDArray = A_inv_ENO @ self.qArray[self.S_ENO]

        S_CENT = self.S_CENT
        S_CENT_shifted = tuple([idx - self.i for idx in S_CENT])
        A_inv: npt.NDArray = polyMat.getMatrixInverse(S_CENT_shifted)
        c_CENT: npt.NDArray = A_inv @ self.qArray[S_CENT]

        self.c = np.zeros(self.degree + 1)
        sum = 0
        dx = self.xArray[1] - self.xArray[0]
        for k in range(1, self.degree + 1):
            w = omega(c_ENO[k], c_CENT[k])
            self.c[k] = 0.5 * (1 + w) * c_ENO[k] + 0.5 * (1 - w) * c_CENT[k]
            integral_f = 1 / (k + 1) * ((dx/2) ** (k + 1) - (-dx/2) ** (k + 1))
            sum += self.c[k] * integral_f   
        self.c[0] = self.qArray[self.i] - 1/dx * sum
        
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
