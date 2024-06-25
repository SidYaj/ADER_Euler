import numpy as np
import numpy.typing as npt
from solver.vector import Vector
from solver.eos import EquationOfState
from typing import Tuple, List, Callable

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

class Riemann:

    def __init__(self, u_L: Vector, u_R: Vector, eos: EquationOfState) -> None:
        self.u_L = u_L
        self.u_R = u_R
        self.eos = eos
        self.calculateSoundSpeed()
        self.calculatePStar()
        self.calculateRhoStar()
        self.calculateVStar()
    
    def calculateSoundSpeed(self) -> None:
        self.cs_L = self.eos.getSoundSpeed(self.u_L[0], self.u_L[2])
        self.cs_R = self.eos.getSoundSpeed(self.u_R[0], self.u_R[2])

    def f_K(self, p_star: float, side: str) -> float:
        gamma = self.eos.gamma
        if side == "left":
            u_K = self.u_L
            cs_K = self.cs_L
        elif side == "right":
            u_K = self.u_R
            cs_K = self.cs_R
        else:
            raise Exception("Invalid side (left/right)")
        if p_star > u_K[2]:
            A_K = 2 / ((gamma + 1) * u_K[0])
            B_K = (gamma - 1) / (gamma + 1) * u_K[2]
            return (p_star - u_K[2]) * np.sqrt(A_K / (p_star + B_K))
        else:
            return ((2 * cs_K)/(gamma - 1)) * ((p_star / u_K[2]) ** ((gamma - 1)/(2 * gamma)) - 1)
        
    def f(self, p_star: float) -> float:
        return self.f_K(p_star, "right") + self.f_K(p_star, "left") + (self.u_R[1] - self.u_L[1])
    
    def dfdp_star(self, p_star: float) -> float:
        EPS = 0.0001
        h = np.sqrt(EPS) * p_star
        h_inv = 1 / h

        return 1/12 * h_inv * (-self.f(p_star + 2 * h) + 8 * self.f(p_star + h) - 8 * self.f(p_star - h) + self.f(p_star - 2 * h))
    
    def calculatePStar(self) -> None:
        
        p_star_old = 0.5 * (self.u_L[2] + self.u_R[2])
        EPS = 0.001
        TOL = 1e-6
        while True:
            p_star_new = p_star_old - self.f(p_star_old) / self.dfdp_star(p_star_old)
            
            if p_star_new < 0:
                p_star_new = EPS
            
            if abs((p_star_new - p_star_old)/p_star_old) < TOL:
                break

            p_star_old = p_star_new

        self.p_star = p_star_new

    def calculateRhoStar(self) -> None:
        gamma = self.eos.gamma
        if self.p_star < self.u_L[2]:
            self.rho_star_L = self.u_L[0] * (self.p_star / self.u_L[2]) ** (1 / gamma)
        else:
            self.rho_star_L = self.u_L[0] * ((self.p_star / self.u_L[2] + (gamma - 1)/(gamma + 1))/((gamma - 1)/(gamma + 1) * self.p_star / self.u_L[2] + 1))

        if self.p_star < self.u_R[2]:
            self.rho_star_R = self.u_R[0] * (self.p_star / self.u_R[2]) ** (1 / gamma)
        else:
            self.rho_star_R = self.u_R[0] * ((self.p_star / self.u_R[2] + (gamma - 1)/(gamma + 1))/((gamma - 1)/(gamma + 1) * self.p_star / self.u_R[2] + 1))

    def calculateVStar(self) -> None:
        self.v_star = 0.5 * (self.u_L[1] + self.u_R[1]) + 0.5 * (self.f_K(self.p_star, "right") - self.f_K(self.p_star, "left"))

    def getStarStates(self) -> Tuple[float, float, float, float]:
        return self.rho_star_L, self.rho_star_R, self.v_star, self.p_star


    def calculateShockSpeed(self, side: str) -> float:
        gamma = self.eos.gamma
        if side == "left":
            S = self.u_L[1] - self.cs_L * np.sqrt((gamma + 1)/(2 * gamma) * self.p_star / self.u_L[2] + (gamma - 1)/(2 * gamma))
        elif side == "right":
            S = self.u_R[1] + self.cs_R * np.sqrt((gamma + 1)/(2 * gamma) * self.p_star / self.u_R[2] + (gamma - 1)/(2 * gamma))  

        return S          

    def getRarefactionState(self, side: str, x: float = 0, x0: float = 0, t: float = 1, t0: float = 0) -> Vector:
        
        gamma = self.eos.gamma
        
        if side == "left":
            rho = self.u_L[0] * (2/(gamma + 1) + (gamma - 1)/((gamma + 1) * self.cs_L) * (self.u_L[1] - (x - x0)/(t - t0))) ** (2 / (gamma - 1))
            v = 2 / (gamma + 1) * (self.cs_L + (gamma - 1)/2 * self.u_L[1] + (x - x0)/(t - t0))
            p = self.u_L[2] * (2/(gamma + 1) + (gamma - 1)/((gamma + 1) * self.cs_L) * (self.u_L[1] - (x - x0)/(t - t0))) ** (2 * gamma / (gamma - 1))
        
        elif side == "right":
            rho = self.u_R[0] * (2/(gamma + 1) - (gamma - 1)/((gamma + 1) * self.cs_R) * (self.u_R[1] - (x - x0)/(t - t0))) ** (2 / (gamma - 1))
            v = 2 / (gamma + 1) * (-self.cs_R + (gamma - 1)/2 * self.u_R[1] + (x - x0)/(t - t0))
            p = self.u_R[2] * (2/(gamma + 1) - (gamma - 1)/((gamma + 1) * self.cs_R) * (self.u_R[1] - (x - x0)/(t - t0))) ** (2 * gamma / (gamma - 1))

        else:
            raise Exception("Invalid side (left/right)")

        return Vector(rho, v, p)
    
    def getCentreSolution(self) -> Vector:
        gamma = self.eos.gamma
        cs_star_L = np.sqrt(gamma * self.p_star / self.rho_star_L)
        cs_star_R = np.sqrt(gamma * self.p_star / self.rho_star_R)

        if self.p_star > self.u_L[2]:
            S = self.calculateShockSpeed("left")
            if S > 0:
                return self.u_L
        else:
            if self.u_L[1] - self.cs_L > 0:
                return self.u_L
            elif (self.u_L[1] - self.cs_L) * (self.v_star - cs_star_L) < 0:
                return self.getRarefactionState("left")
            
        if self.p_star > self.u_R[2]:
            S = self.calculateShockSpeed("right")
            if S < 0:
                return self.u_R
            
        else:
            if self.u_R[1] + self.cs_R < 0:
                return self.u_R
            elif (self.u_R[1] + self.cs_R) * (self.v_star + cs_star_R) < 0:
                return self.getRarefactionState("right")
            
        if self.v_star < 0:
            return Vector(self.rho_star_R, self.v_star, self.p_star)
        else:
            return Vector(self.rho_star_L, self.v_star, self.p_star)
        
    
    def computeSolution(self, x: float, x0: float, t: float):
        gamma = self.eos.gamma
        cs_star_L = np.sqrt(gamma * self.p_star / self.rho_star_L)
        cs_star_R = np.sqrt(gamma * self.p_star / self.rho_star_R)
        u_star_L = Vector(self.rho_star_L, self.v_star, self.p_star)
        u_star_R = Vector(self.rho_star_R, self.v_star, self.p_star)
        if self.p_star > self.u_L[2]:
            
            if x < self.calculateShockSpeed("left") * t + x0:
                return self.u_L
            elif x < self.v_star * t + x0:
                return u_star_L
            
            if self.p_star > self.u_R[2]:
                if x < self.calculateShockSpeed("right") * t + x0:
                    return u_star_R
                else:
                    return self.u_R
            else:
                if x < (self.v_star + cs_star_R) * t + x0:
                    return u_star_R
                elif x < (self.u_R[1] + self.cs_R) * t + x0:
                    return self.getRarefactionState("right", x, x0, t)
                else:
                    return self.u_R
        else:
            if x < (self.u_L[1] - self.cs_L) * t + x0:
                return self.u_L
            elif x < (self.v_star - cs_star_L) * t + x0:
                return self.getRarefactionState("left", x, x0, t)
            elif x < self.v_star * t + x0:
                return u_star_L
            
            if self.p_star > self.u_R[2]:
                if x < self.calculateShockSpeed("right") * t + x0:
                    return u_star_R
                else:
                    return self.u_R
            else:
                if x < (self.v_star + cs_star_R) * t + x0:
                    return u_star_R
                elif x < (self.u_R[1] + self.cs_R) * t + x0:
                    return self.getRarefactionState("right", x, x0, t)
                else:
                    return self.u_R


    def constructExactSolution(self, xMin: float, xMax: float, x0: float, nCells: int, t: float) -> Tuple[npt.NDArray, npt.NDArray]:
        
        dx = (xMax - xMin) / nCells
        xStart = xMin + 0.5 * dx
        xEnd = xMax - 0.5 * dx
        xArray: npt.NDArray = np.linspace(xStart, xEnd, nCells)
        primitiveArray: npt.NDArray = np.array([gaussQuad(lambda xx: self.computeSolution(xx, x0, t), x - 0.5 * dx, x + 0.5 * dx) / dx for x in xArray])
        return xArray, primitiveArray


