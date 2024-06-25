import numpy as np
import numpy.typing as npt
from typing import List, Tuple
from numba import njit # type: ignore

@njit
def nb_inv(A: npt.NDArray) -> npt.NDArray:
    return np.linalg.inv(A)

class PolynomialMatrices:

    def __init__(self, degree: int, dx: float) -> None:
        self.__generateListOfMatrixInverses(degree, dx)

    def __generateListOfMatrixInverses(self, degree: int, dx: float) -> None:
        
        listOfStencils: List[Tuple]  = [tuple(range(end - degree, end + 1)) for end in range(degree + 1)]  
        listOfInverses: List[npt.NDArray] = []

        for stencil in listOfStencils:
            A_list: List[List[float]] = []
            for j in stencil:
                A_row = [1 / ((k + 1) * dx) * (((j + 0.5) * dx) ** (k + 1) - ((j - 0.5) * dx) ** (k + 1)) for k in range(degree + 1)]
                A_list.append(A_row)
            A: npt.NDArray = np.array(A_list)
            A_inv: npt.NDArray = nb_inv(A)
            listOfInverses.append(A_inv)
            
        self.__dictOfInverses = dict(zip(listOfStencils, listOfInverses))
    
    def getMatrixInverse(self, stencil: Tuple) -> npt.NDArray:
        return self.__dictOfInverses[stencil]