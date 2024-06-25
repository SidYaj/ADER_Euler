import numpy.typing as npt
import numpy as np
from typing import List, Tuple
from solver.reconstruction.functions import dividedDifference
from icecream import ic # type: ignore
from solver.reconstruction.polynomial_matrices import PolynomialMatrices
from numba import njit # type: ignore

@njit
def getStencils(i: int, degree: int, xArray: npt.NDArray, qArray: npt.NDArray) -> Tuple[List[int], List[int]]:
    S: List[int] = [i]
    for _ in range(degree - 1):
        idxLeft = [S[0] - 1] + S
        idxLeftArray = np.array(idxLeft)
        idxRight = S + [S[-1] + 1]
        idxRightArray = np.array(idxRight)
        ddList = [dividedDifference(xArray[idxLeftArray], qArray[idxLeftArray]), dividedDifference(xArray[idxRightArray], qArray[idxRightArray])]
        if (abs(ddList[0]) < abs(ddList[1])):
            S = idxLeft
        else:
            S = idxRight
    
    S1 = [S[0] - 1] + S
    S2 = S + [S[-1] + 1]
    
    return S1, S2

class AenoStencil:

    def __init__(self, xArray: npt.NDArray, qArray: npt.NDArray, degree: int):
        self.xArray = xArray
        self.qArray = qArray
        self.degree = degree

    def stencils(self, i: int) -> Tuple[List[int], List[int]]:
        return getStencils(i, self.degree, self.xArray, self.qArray)

@njit
def joining_function(a_L: float, a_R: float, eps: float = np.sqrt(0.5)) -> float:
    TOL = 1e-6
    s = np.abs(a_L) / (np.abs(a_R) + TOL)
    w: float = (1 - s) / (np.sqrt(eps ** 2 + (1 - s) ** 2))
    return w
class AenoPolynomialOld:
    
    def __init__(self, xArray: npt.NDArray, qArray: npt.NDArray, SL: List[int], SR: List[int], i: int, polyMat: PolynomialMatrices): 
        self.xArray = xArray
        self.qArray = qArray
        self.SL = SL
        self.SR = SR
        self.degree = len(SL) - 1
        self.i = i
        self.conservativeCoefficients(polyMat)
        self.derivatives()

    def conservativeCoefficients(self, polyMat: PolynomialMatrices) -> None:
        
        SL_shifted = tuple([idx - self.i for idx in self.SL])
        SR_shifted = tuple([idx - self.i for idx in self.SR])
        A_inv_L: npt.NDArray = polyMat.getMatrixInverse(SL_shifted)
        A_inv_R: npt.NDArray = polyMat.getMatrixInverse(SR_shifted)
        c_L: npt.NDArray = A_inv_L @ self.qArray[self.SL] # coefficients for the left polynomial
        c_R: npt.NDArray = A_inv_R @ self.qArray[self.SR] # coefficients for the right polynomial
        self.c: npt.NDArray = np.zeros(self.degree + 1)
        sum: float = 0
        dx = self.xArray[1] - self.xArray[0]
        for k in range(1, self.degree + 1):
            w = joining_function(c_L[k], c_R[k])
            self.c[k] = 0.5 * (1 + w) * c_L[k] + 0.5 * (1 - w) * c_R[k]
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

class AenoPolynomialNew:
    
    def __init__(self, xArray: npt.NDArray, qArray: npt.NDArray, SL: List[int], SR: List[int], i: int, polyMat: PolynomialMatrices): 
        self.xArray = xArray
        self.qArray = qArray
        self.SL = SL
        self.SR = SR
        self.degree = len(SL) - 1
        self.i = i
        self.conservativeCoefficients(polyMat)
        self.derivatives()

    def conservativeCoefficients(self, polyMat: PolynomialMatrices) -> None:
        
        SL_shifted = tuple([idx - self.i for idx in self.SL])
        SR_shifted = tuple([idx - self.i for idx in self.SR])
        A_inv_L: npt.NDArray = polyMat.getMatrixInverse(SL_shifted)
        A_inv_R: npt.NDArray = polyMat.getMatrixInverse(SR_shifted)
        c_L: npt.NDArray = A_inv_L @ self.qArray[self.SL] # coefficients for the left polynomial
        c_R: npt.NDArray = A_inv_R @ self.qArray[self.SR] # coefficients for the right polynomial
        self.c: npt.NDArray = np.zeros(self.degree + 1)
        sum: float = 0
        dx = self.xArray[1] - self.xArray[0]
        for k in range(1, self.degree + 1):
            w = omega(c_L[k], c_R[k])
            self.c[k] = 0.5 * (1 + w) * c_L[k] + 0.5 * (1 - w) * c_R[k]
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