from abc import ABC, abstractmethod
from typing import Callable, Tuple, List
import numpy as np
from numpy._typing import NDArray
import numpy.typing as npt
from enum import Enum
from numba import njit # type: ignore
from math import factorial
from functools import reduce
from solver.vector import Vector
from solver.eos import EquationOfState
from solver.riemann import Riemann
from solver.reconstruction.polynomial_matrices import PolynomialMatrices
from solver.reconstruction.eno import ENO
from solver.reconstruction.weno import WENO
from solver.cauchy_kovalevskaya import cauchyKovalevskaya

class BoundaryCondition(Enum):
    TRANSMISSIVE = 1
    PERIODIC = 2
    REFLECTIVE = 3


gaussPointsList: List[npt.NDArray] = [np.array([0]),
                                  np.array([-1/np.sqrt(3), 1/np.sqrt(3)]),
                                  np.array([-np.sqrt(3/5), 0, np.sqrt(3/5)]),
                                  np.array([-np.sqrt(3/7 + 2/7 * np.sqrt(6/5)),-np.sqrt(3/7 - 2/7 * np.sqrt(6/5)),np.sqrt(3/7 - 2/7 * np.sqrt(6/5)),np.sqrt(3/7 + 2/7 * np.sqrt(6/5))]),
                                  np.array([-1/3 * np.sqrt(5 + 2 * np.sqrt(10/7)), -1/3 * np.sqrt(5 - 2 * np.sqrt(10/7)), 0, 1/3 * np.sqrt(5 - 2 * np.sqrt(10/7)), 1/3 * np.sqrt(5 + 2 * np.sqrt(10/7))])]

gaussWeightsList: List[npt.NDArray] = [np.array([2]),
                                   np.array([1, 1]),
                                   np.array([5/9, 8/9, 5/9]),
                                   np.array([(18 - np.sqrt(30))/36, (18 + np.sqrt(30))/36, (18 + np.sqrt(30))/36, (18 - np.sqrt(30))/36]),
                                   np.array([(322 - 13 * np.sqrt(70))/900, (322 + 13 * np.sqrt(70))/900, 128/225, (322 + 13 * np.sqrt(70))/900, (322 - 13 * np.sqrt(70))/900])]

def gaussQuad(func: Callable[[float], Vector], a: float, b: float, n=5) -> Vector:
    gaussPoints: npt.NDArray = gaussPointsList[n - 1]
    gaussWeights: npt.NDArray = gaussWeightsList[n - 1]
    gaussPointsSpace = 0.5 * (b - a) * gaussPoints + 0.5 * (a + b)
    gaussValues = np.array([func(point) for point in gaussPointsSpace], dtype=Vector)
    gaussValuesWeighted = gaussValues * gaussWeights
    integral = 0.5 * (b - a) * gaussValuesWeighted.sum()
    return integral


class Solver(ABC):
    
    def __init__(self, xMin: float, xMax: float, nCells: int, nGhostCells: int, CFL: float, tMax: float, initialConditionFunction: Callable[[float], Vector], eos: EquationOfState, bcLeft: BoundaryCondition, bcRight: BoundaryCondition) -> None:
        
        self.xMin: float = xMin
        self.xMax: float = xMax
        self.nCells: int = nCells
        self.nGhostCells = self.order = nGhostCells
        self.nTotalCells: int = nCells + 2 * nGhostCells
        self.dx: float = (xMax - xMin)/nCells
        self.istart: int = nGhostCells
        self.iend: int = nCells + nGhostCells
        xStart: float = xMin - (nGhostCells - 0.5) * self.dx
        xEnd: float = xMax + (nGhostCells - 0.5) * self.dx
        self.CFL: float = CFL
        self.eos: EquationOfState = eos
        self.xArray: npt.NDArray = np.linspace(xStart, xEnd, self.nTotalCells)
        self.primitiveArray: npt.NDArray = np.array([gaussQuad(initialConditionFunction, x - 0.5 * self.dx, x + 0.5 * self.dx) / self.dx for x in self.xArray])
        self.conservativeArray: npt.NDArray = np.array([self.primitiveToConservative(Q) for Q in self.primitiveArray])
        self.bcLeft: BoundaryCondition = bcLeft
        self.bcRight: BoundaryCondition = bcRight
        self.tMax = tMax
        self.computeTimeStep()

    def fluxFunction(self, U: Vector) -> Vector:
        primitive = self.conservativeToPrimitive(U)
        rho = primitive[0]
        v = primitive[1]
        p = primitive[2]
        e = U[2]
        return Vector(rho * v, rho * v ** 2 + p, (e + p) * v)


    def primitiveToConservative(self, U: Vector) -> Vector:
        
        rho = U[0]
        v = U[1]
        p = U[2]

        mom = rho * v
        eps = self.eos.getInternalEnergy(rho, p)
        e = rho * eps + 0.5 * rho * v ** 2
        
        return Vector(rho, mom, e)
    
    def conservativeToPrimitive(self, U: Vector) -> Vector:
        
        rho = U[0]
        mom = U[1]
        e = U[2]

        v = mom / rho
        eps = (e - 0.5 * rho * v ** 2) / rho
        p = self.eos.getPressure(rho, eps)
        
        return Vector(rho, v, p)
    
    def computeTimeStep(self) -> None:
        aList = [abs(q[1]) + self.eos.getSoundSpeed(q[0], q[2]) for q in self.primitiveArray]
        aMax = reduce(max, aList)
        self.dt = self.CFL * self.dx / aMax

    def applyBoundaryConditions(self, qArray = None) -> None:

        if qArray is None:
            if self.bcLeft == BoundaryCondition.TRANSMISSIVE:
                for i in range(self.nGhostCells):
                    self.conservativeArray[i] = self.conservativeArray[self.istart]
            elif self.bcLeft == BoundaryCondition.PERIODIC:
                for i in range(self.nGhostCells):
                    self.conservativeArray[i] = self.conservativeArray[i + self.nCells]
            elif self.bcLeft == BoundaryCondition.REFLECTIVE:
                for i in range(self.nGhostCells):
                    self.conservativeArray[i] = -self.conservativeArray[2 * self.istart - 1 - i]

            if self.bcRight == BoundaryCondition.TRANSMISSIVE:
                for i in range(self.nGhostCells):
                    self.conservativeArray[self.iend + i] = self.conservativeArray[self.iend - 1]
            elif self.bcRight == BoundaryCondition.PERIODIC:
                for i in range(self.nGhostCells):
                    self.conservativeArray[self.iend + i] = self.conservativeArray[self.iend + i - self.nCells]
            elif self.bcRight == BoundaryCondition.REFLECTIVE:
                for i in range(self.nGhostCells):
                    self.conservativeArray[self.iend + i] = -self.conservativeArray[self.iend - 1 - i]
        else:
            if self.bcLeft == BoundaryCondition.TRANSMISSIVE:
                for i in range(self.nGhostCells):
                    qArray[i] = qArray[self.istart]
            elif self.bcLeft == BoundaryCondition.PERIODIC:
                for i in range(self.nGhostCells):
                    qArray[i] = qArray[i + self.nCells]
            elif self.bcLeft == BoundaryCondition.REFLECTIVE:
                for i in range(self.nGhostCells):
                    qArray[i] = -qArray[2 * self.istart - 1 - i]

            if self.bcRight == BoundaryCondition.TRANSMISSIVE:
                for i in range(self.nGhostCells):
                    qArray[self.iend + i] = qArray[self.iend - 1]
            elif self.bcRight == BoundaryCondition.PERIODIC:
                for i in range(self.nGhostCells):
                    qArray[self.iend + i] = qArray[self.iend + i - self.nCells]
            elif self.bcRight == BoundaryCondition.REFLECTIVE:
                for i in range(self.nGhostCells):
                    qArray[self.iend + i] = -qArray[self.iend - 1 - i]

    @abstractmethod
    def calculateFlux(self) -> npt.NDArray:
        pass

    def evolveSolution(self) -> None:
        flux: npt.NDArray = self.calculateFlux()
        conservativeNew: npt.NDArray = np.array([Vector()] * self.nTotalCells, dtype=Vector)
        conservativeNew[self.istart:self.iend] = self.conservativeArray[self.istart:self.iend] - self.dt / self.dx * (flux[self.istart + 1:self.iend + 1] - flux[self.istart:self.iend])

        self.conservativeArray = conservativeNew
        self.applyBoundaryConditions()
        self.primitiveArray = np.array([self.conservativeToPrimitive(Q) for Q in self.conservativeArray])

    def solveODE(self) -> None:
        t: float = 0
        t += self.dt
        flag: bool = True
        nIter = 0
        while flag:     
            # print(f"{t=}")
            
            if t == self.tMax:
                flag = False

            self.evolveSolution()
            self.computeTimeStep()

            if nIter < 10:
                self.dt *= 0.5
            
            if t + self.dt > self.tMax:
                self.dt = self.tMax - t
                t = self.tMax
            else:
                t += self.dt

            nIter += 1

    def returnData(self) -> Tuple[npt.NDArray, npt.NDArray]:
        
        return self.xArray[self.istart:self.iend], self.primitiveArray[self.istart:self.iend]
    

class FORCE(Solver):

    def __init__(self, xMin: float, xMax: float, nCells: int, CFL: float, tMax: float, initialConditionFunction: Callable[[float], Vector], eos: EquationOfState, bcLeft: BoundaryCondition, bcRight: BoundaryCondition) -> None:
        
        super().__init__(xMin, xMax, nCells, 1, CFL, tMax, initialConditionFunction, eos, bcLeft, bcRight)

    def __str__(self) -> str:
        return "FORCE"
    
    def laxFriedrichsFlux(self, q_L: Vector, q_R: Vector) -> Vector:
        flux: Vector = 0.5 * self.dx / self.dt * (q_L - q_R) + 0.5 * (self.fluxFunction(q_L) + self.fluxFunction(q_R))
        return flux

    def richtmyerFlux(self, q_L: Vector, q_R: Vector) -> Vector:
        qHalf: Vector = 0.5 * (q_L + q_R) - 0.5 * self.dt / self.dx * (self.fluxFunction(q_R) - self.fluxFunction(q_L))
        flux: Vector = self.fluxFunction(qHalf)
        return flux
    
    def forceFlux(self, q_L: Vector, q_R: Vector) -> Vector:
        f_LF: Vector = self.laxFriedrichsFlux(q_L, q_R)
        f_RI: Vector = self.richtmyerFlux(q_L, q_R)
        f_FORCE: Vector = 0.5 * (f_LF + f_RI)
        return f_FORCE

    def calculateFlux(self) -> npt.NDArray:
        flux: npt.NDArray = np.array([Vector()] * (self.nTotalCells + 1), dtype=Vector)
        for i in range(self.istart, self.iend + 1):
            flux[i] = self.forceFlux(self.conservativeArray[i - 1], self.conservativeArray[i])
        return flux
    
    
class Godunov(Solver):

    def __init__(self, xMin: float, xMax: float, nCells: int, CFL: float, tMax: float, initialConditionFunction: Callable[[float], Vector], eos: EquationOfState, bcLeft: BoundaryCondition, bcRight: BoundaryCondition) -> None:
        
        super().__init__(xMin, xMax, nCells, 1, CFL, tMax, initialConditionFunction, eos, bcLeft, bcRight)


    def __str__(self) -> str:
        return "Godunov"

    def calculateFlux(self) -> npt.NDArray:
        flux: npt.NDArray = np.array([Vector()] * (self.nTotalCells + 1), dtype=Vector)
        for i in range(self.istart, self.iend + 1):
            primitive_L: Vector = self.primitiveArray[i - 1]
            primitive_R: Vector = self.primitiveArray[i]
            rp = Riemann(primitive_L, primitive_R, self.eos)
            primitive_soln: Vector = rp.getCentreSolution()
            conservative_soln: Vector = self.primitiveToConservative(primitive_soln)
            flux[i] = self.fluxFunction(conservative_soln)
        return flux
    
    
@njit
def getCharacteristicToConservativeMatrix(v: float, cs: float, gamma: float) -> npt.NDArray:
    return np.array([[(2 * (gamma - 1))/(2 * cs * v + gamma * v ** 2 + 2 * cs ** 2 - v ** 2 - 2 * gamma * cs * v), 1, (2 * (gamma - 1))/(gamma * v ** 2 - 2 * cs * v + 2 * cs ** 2 - v ** 2 + 2  * gamma * cs * v)], 
                     [(2 * (v - cs) * (gamma - 1))/(2 * cs * v + gamma * v ** 2 + 2 * cs ** 2 - v ** 2 - 2 * gamma * cs * v), v, (2 * (v + cs) * (gamma - 1))/(gamma * v ** 2 - 2 * cs * v + 2 * cs ** 2 - v ** 2 + 2 * gamma * cs * v)], 
                     [1, v ** 2 / 2, 1]])

@njit
def getConservativeToCharacteristicMatrix(v: float, cs: float, gamma: float) -> npt.NDArray:
    return np.array([[(v * (2 * cs - v + gamma * v) * (2 * cs * v + gamma * v ** 2 + 2 * cs ** 2 - v ** 2 - 2 * gamma * cs * v))/(8 * (gamma * cs ** 2 - cs ** 2)), -((cs - v + gamma * v)*(2 * cs * v + gamma * v ** 2 + 2 * cs ** 2 - v ** 2 - 2 * gamma * cs * v))/(4 * (gamma * cs ** 2 - cs ** 2)), (2 * cs * v + gamma * v ** 2 + 2 * cs ** 2 - v ** 2 - 2 * gamma * cs * v)/(4 * cs ** 2)], 
                     [(2 * cs ** 2 - gamma * v ** 2 + v ** 2)/(2 * cs ** 2), (v * (gamma - 1))/cs ** 2, -(gamma - 1)/cs**2], 
                     [-(v * (2 * cs + v - gamma * v) * (gamma * v ** 2 - 2 * cs * v + 2 * cs ** 2 - v ** 2 + 2 * gamma * cs * v))/(8 * (gamma * cs ** 2 - cs ** 2)), ((v + cs - gamma * v)*(gamma * v ** 2 - 2 * cs * v + 2 * cs ** 2 - v ** 2 + 2 * gamma * cs * v))/(4 * (gamma * cs ** 2 - cs ** 2)), (gamma * v ** 2 - 2 * cs * v + 2 * cs ** 2 - v ** 2 + 2 * gamma * cs * v)/(4*cs**2)]])

@njit 
def getJacobian(rho: float, q: float, e: float, gamma: float) -> npt.NDArray:
    return np.array([[0.0, 1.0, 0.0],
                     [q ** 2 * (0.5 * gamma - 1.5) / rho ** 2, q * (3 - gamma)/rho, gamma - 1],
                     [q * (-gamma * rho * e + gamma * q ** 2 - q ** 2) / rho ** 3, (gamma * rho * e - 1.5 * gamma * q ** 2 + 1.5 * q ** 2) / rho ** 2, gamma * q / rho]])


@njit
def solveLinearRiemannProblem(u_L_array: npt.NDArray, u_R_array: npt.NDArray, v0: float, cs0: float, gamma: float) -> npt.NDArray:
    C: npt.NDArray = getCharacteristicToConservativeMatrix(v0, cs0, gamma)
    C_inv: npt.NDArray = getConservativeToCharacteristicMatrix(v0, cs0, gamma)
    v_L = C_inv @ u_L_array
    v_R = C_inv @ u_R_array
    eigs = np.array([v0 - cs0, v0, v0 + cs0])
    v: npt.NDArray = v_L * (eigs > 0) + v_R * (eigs < 0)
    u: npt.NDArray = C @ v
    return u


class ADER_ENO(Solver):

    def __init__(self, xMin: float, xMax: float, nCells: int, order: int, CFL: float, tMax: float, initialConditionFunction: Callable[[float], Vector], eos: EquationOfState, bcLeft: BoundaryCondition, bcRight: BoundaryCondition) -> None:
        super().__init__(xMin, xMax, nCells, order, CFL, tMax, initialConditionFunction, eos, bcLeft, bcRight)
        self.order = order
        self.polyMat = PolynomialMatrices(order - 1, self.dx)
    
    def __str__(self) -> str:
        return f"ADER_ENO_{self.order}"
    
    def calculateFlux(self) -> npt.NDArray:
        
        degree = self.order - 1
        flux: npt.NDArray = np.array([Vector()] * (self.nTotalCells + 1), dtype=Vector)

        pList_0: List[ENO] = []
        pList_1: List[ENO] = []
        pList_2: List[ENO] = []
        C_list: List[npt.NDArray] = []
        C_inv_list: List[npt.NDArray] = []

        for i in range(self.istart - 1, self.iend + 1):
            extendedStencil = np.arange(i - degree, i + degree + 1)
            cellPrimitiveValues: Vector = self.primitiveArray[i]
            rho: float = cellPrimitiveValues[0]
            v: float = cellPrimitiveValues[1]
            p: float = cellPrimitiveValues[2]
            cs: float = self.eos.getSoundSpeed(rho, p)
            C_inv: npt.NDArray = getConservativeToCharacteristicMatrix(v, cs, self.eos.gamma)
            C: npt.NDArray = getCharacteristicToConservativeMatrix(v, cs, self.eos.gamma)
            C_list.append(C)
            C_inv_list.append(C_inv)
            stencilConservativeValues: npt.NDArray = self.conservativeArray[extendedStencil]
            stencil_xValues: npt.NDArray = self.xArray[extendedStencil]
            stencilCharacteristicValues: npt.NDArray = np.array([C_inv @ conservativeVector.toNumpyArray() for conservativeVector in stencilConservativeValues])
            characteristicArray0: npt.NDArray[np.float64] = np.array([vec[0] for vec in stencilCharacteristicValues])
            characteristicArray1: npt.NDArray[np.float64] = np.array([vec[1] for vec in stencilCharacteristicValues])
            characteristicArray2: npt.NDArray[np.float64] = np.array([vec[2] for vec in stencilCharacteristicValues])
            pList_0.append(ENO(stencil_xValues, characteristicArray0, degree, self.polyMat))
            pList_1.append(ENO(stencil_xValues, characteristicArray1, degree, self.polyMat))
            pList_2.append(ENO(stencil_xValues, characteristicArray2, degree, self.polyMat))

        if degree <= 1:
            nGaussPoints = 1
        elif degree <= 3:
            nGaussPoints = 2
        elif degree <= 5:
            nGaussPoints = 3
        elif degree <= 7:
            nGaussPoints = 4
        elif degree <= 9:
            nGaussPoints = 5

        gaussPoints = gaussPointsList[nGaussPoints - 1]
        gaussWeights = gaussWeightsList[nGaussPoints - 1]

        gaussPointsMappedTime: npt.NDArray = self.dt / 2 * gaussPoints + self.dt / 2

        for idx, i in enumerate(range(self.istart, self.iend + 1)):
            xInterface: float = self.xArray[i] - 0.5 * self.dx
            drho_x: List[float] = []
            dq_x: List[float] = []
            dE_x: List[float] = []
            drho_t: List[float] = []
            dq_t: List[float] = []
            dE_t: List[float] = []

            for k in range(degree + 1):
                w0_L: float = pList_0[idx].getValue(xInterface, k)
                w0_R: float = pList_0[idx + 1].getValue(xInterface, k)
                w1_L: float = pList_1[idx].getValue(xInterface, k)
                w1_R: float = pList_1[idx + 1].getValue(xInterface, k)
                w2_L: float = pList_2[idx].getValue(xInterface, k)
                w2_R: float = pList_2[idx + 1].getValue(xInterface, k)
                w_L: npt.NDArray = np.array([w0_L, w1_L, w2_L])
                w_R: npt.NDArray = np.array([w0_R, w1_R, w2_R])
                conservative_L = Vector.fromNumpyArray(C_list[idx] @ w_L)
                conservative_R = Vector.fromNumpyArray(C_list[idx + 1] @ w_R)            
                if k == 0:
                    primitive_L = self.conservativeToPrimitive(conservative_L)
                    primitive_R = self.conservativeToPrimitive(conservative_R)
                    rp = Riemann(primitive_L, primitive_R, self.eos)
                    primitive_soln = rp.getCentreSolution()
                    conservative_soln = self.primitiveToConservative(primitive_soln)
                    rho_0 = primitive_soln[0]
                    v_0 = primitive_soln[1]
                    p_0 = primitive_soln[2]
                    cs_0 = self.eos.getSoundSpeed(rho_0, p_0)
                
                else:
                    conservative_soln = Vector.fromNumpyArray(solveLinearRiemannProblem(conservative_L.toNumpyArray(), conservative_R.toNumpyArray(), v_0, cs_0, self.eos.gamma))
                
                drho_x.append(conservative_soln[0])
                dq_x.append(conservative_soln[1])
                dE_x.append(conservative_soln[2])

                dCons_t = cauchyKovalevskaya(np.array(drho_x, dtype=np.float64), np.array(dq_x, dtype=np.float64), np.array(dE_x, dtype=np.float64), k, self.eos.gamma)
                drho_t.append(dCons_t[0])
                dq_t.append(dCons_t[1])
                dE_t.append(dCons_t[2])


            gaussValues = np.array([Vector()] * nGaussPoints, dtype=Vector)

            for m, t in enumerate(gaussPointsMappedTime):
                gaussValuesRho = sum(drho_t[k] * t ** k / factorial(k) for k in range(degree + 1))
                gaussValuesQ = sum(dq_t[k] * t ** k / factorial(k) for k in range(degree + 1))
                gaussValuesE = sum(dE_t[k] * t ** k / factorial(k) for k in range(degree + 1))
                gaussValues[m] = Vector(gaussValuesRho, gaussValuesQ, gaussValuesE)

            fluxValues = np.array([self.fluxFunction(value) for value in gaussValues], dtype=Vector)

            integral: Vector = 0.5 * (fluxValues * gaussWeights).sum()
            flux[i] = integral

        return flux
    

class ADER_WENO(Solver):

    def __init__(self, xMin: float, xMax: float, nCells: int, order: int, CFL: float, tMax: float, initialConditionFunction: Callable[[float], Vector], eos: EquationOfState, bcLeft: BoundaryCondition, bcRight: BoundaryCondition) -> None:
        super().__init__(xMin, xMax, nCells, order, CFL, tMax, initialConditionFunction, eos, bcLeft, bcRight)
        self.order = order
        self.polyMat = PolynomialMatrices(order - 1, self.dx)
    
    def __str__(self) -> str:
        return f"ADER_WENO_{self.order}"
    
    def calculateFlux(self) -> npt.NDArray:
        
        degree = self.order - 1
        flux: npt.NDArray = np.array([Vector()] * (self.nTotalCells + 1), dtype=Vector)

        if degree <= 1:
            nGaussPoints = 1
        elif degree <= 3:
            nGaussPoints = 2
        elif degree <= 5:
            nGaussPoints = 3
        elif degree <= 7:
            nGaussPoints = 4
        elif degree <= 9:
            nGaussPoints = 5

        gaussPoints = gaussPointsList[nGaussPoints - 1]
        gaussWeights = gaussWeightsList[nGaussPoints - 1]

        gaussPointsMappedTime: npt.NDArray = self.dt / 2 * gaussPoints + self.dt / 2

        for idx, i in enumerate(range(self.istart, self.iend + 1)):
            drho_x: List[float] = []
            dq_x: List[float] = []
            dE_x: List[float] = []
            drho_t: List[float] = []
            dq_t: List[float] = []
            dE_t: List[float] = []

            interfaceLeftState: Vector = self.conservativeArray[i - 1]
            interfaceRightState: Vector = self.conservativeArray[i]
            interfaceAverageState: Vector = 0.5 * (interfaceLeftState + interfaceRightState)
            interfacePrimitiveState = self.conservativeToPrimitive(interfaceAverageState)
            rho: float = interfacePrimitiveState[0]
            v: float = interfacePrimitiveState[1]
            p: float = interfacePrimitiveState[2]
            cs: float = self.eos.getSoundSpeed(rho, p)
            C: npt.NDArray = getCharacteristicToConservativeMatrix(v, cs, self.eos.gamma)
            C_inv: npt.NDArray = getConservativeToCharacteristicMatrix(v, cs, self.eos.gamma) 

            extendedStencilInterface = np.arange(i - 1 - degree, i + degree + 1)
            stencilConservativeValues: npt.NDArray = self.conservativeArray[extendedStencilInterface]
            stencil_xValues: npt.NDArray = self.xArray[extendedStencilInterface]
            stencilCharacteristicValues = np.array([C_inv @ conservativeState.toNumpyArray() for conservativeState in stencilConservativeValues])
            characteristicArray0: npt.NDArray = np.array([vec[0] for vec in stencilCharacteristicValues])
            characteristicArray1: npt.NDArray = np.array([vec[1] for vec in stencilCharacteristicValues])
            characteristicArray2: npt.NDArray = np.array([vec[2] for vec in stencilCharacteristicValues])

            weno_0 = WENO(stencil_xValues, characteristicArray0, degree)
            weno_1 = WENO(stencil_xValues, characteristicArray1, degree)
            weno_2 = WENO(stencil_xValues, characteristicArray2, degree)
            dw0_L, dw0_R = weno_0.reconstructAtInterface()
            dw1_L, dw1_R = weno_1.reconstructAtInterface()
            dw2_L, dw2_R = weno_2.reconstructAtInterface()
            for k in range(degree + 1):
                
                w_L: npt.NDArray = np.array([dw0_L[k], dw1_L[k], dw2_L[k]])
                w_R: npt.NDArray = np.array([dw0_R[k], dw1_R[k], dw2_R[k]])
                conservative_L = Vector.fromNumpyArray(C @ w_L)
                conservative_R = Vector.fromNumpyArray(C @ w_R)            
                if k == 0:
                    primitive_L = self.conservativeToPrimitive(conservative_L)
                    primitive_R = self.conservativeToPrimitive(conservative_R)
                    rp = Riemann(primitive_L, primitive_R, self.eos)
                    primitive_soln = rp.getCentreSolution()
                    conservative_soln = self.primitiveToConservative(primitive_soln)
                    rho_0 = primitive_soln[0]
                    v_0 = primitive_soln[1]
                    p_0 = primitive_soln[2]
                    cs_0 = self.eos.getSoundSpeed(rho_0, p_0)
                
                else:
                    conservative_soln = Vector.fromNumpyArray(solveLinearRiemannProblem(conservative_L.toNumpyArray(), conservative_R.toNumpyArray(), v_0, cs_0, self.eos.gamma))
                
                drho_x.append(conservative_soln[0])
                dq_x.append(conservative_soln[1])
                dE_x.append(conservative_soln[2])

                dCons_t = cauchyKovalevskaya(np.array(drho_x, dtype=np.float64), np.array(dq_x, dtype=np.float64), np.array(dE_x, dtype=np.float64), k, self.eos.gamma)
                drho_t.append(dCons_t[0])
                dq_t.append(dCons_t[1])
                dE_t.append(dCons_t[2])


            gaussValues = np.array([Vector()] * nGaussPoints, dtype=Vector)

            for m, t in enumerate(gaussPointsMappedTime):
                gaussValuesRho = sum(drho_t[k] * t ** k / factorial(k) for k in range(degree + 1))
                gaussValuesQ = sum(dq_t[k] * t ** k / factorial(k) for k in range(degree + 1))
                gaussValuesE = sum(dE_t[k] * t ** k / factorial(k) for k in range(degree + 1))
                gaussValues[m] = Vector(gaussValuesRho, gaussValuesQ, gaussValuesE)

            fluxValues = np.array([self.fluxFunction(value) for value in gaussValues], dtype=Vector)

            integral: Vector = 0.5 * (fluxValues * gaussWeights).sum()
            flux[i] = integral

        return flux