from dataclasses import dataclass
from math import sqrt

@dataclass
class EquationOfState:

    gamma: float

    def getPressure(self, density: float, internalEnergy: float) -> float:
        
        p = (self.gamma - 1) * density * internalEnergy
        return p
    
    def getInternalEnergy(self, density: float, pressure: float) -> float:

        e = pressure / ((self.gamma - 1) * density)
        return e
    
    def getSoundSpeed(self, density: float, pressure: float) -> float:
        return sqrt(self.gamma * pressure / density)