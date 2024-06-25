from solver.solver import getCharacteristicToConservativeMatrix, getConservativeToCharacteristicMatrix, getJacobian
from math import sqrt

rho = 1
p = 1
q = 1
v = q / rho
gamma = 1.4
cs = sqrt(gamma * p / rho)
e = p / (gamma - 1) + 0.5 * rho * v ** 2
C = getCharacteristicToConservativeMatrix(v, cs, gamma)
C_inv = getConservativeToCharacteristicMatrix(v, cs, gamma)
J = getJacobian(rho, q, e, gamma)
D = C_inv @ J @ C
print(f"{C=}")
print(f"{C_inv=}")
print(f"{D=}")
print(f"{v - cs = }")
print(f"{v = }")
print(f"{v + cs = }")