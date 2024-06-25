from __future__ import annotations
import numpy as np
import numpy.typing as npt

class Vector:

    def __init__(self, x=0.0, y=0.0, z=0.0) -> None:
        self.__x = x
        self.__y = y
        self.__z = z

    def __str__(self) -> str:
        return f'({self.__x:.5f}, {self.__y:.5f}, {self.__z:.5f})'

    def __add__(self, other: Vector) -> Vector:
        return Vector(self.__x + other.__x, self.__y + other.__y, self.__z + other.__z)
    
    def __iadd__(self, other: Vector) -> Vector:
        return self + other
    
    def __sub__(self, other: Vector) -> Vector:
        return Vector(self.__x - other.__x, self.__y - other.__y, self.__z - other.__z)
    
    def __isub__(self, other: Vector) -> Vector:
        return self - other
    
    def __mul__(self, other: float) -> Vector:
        return Vector(self.__x * other, self.__y * other, self.__z * other)
    
    def __imul__(self, other: float) -> Vector:
        return self * other
    
    def __rmul__(self, other: float) -> Vector:
        return self * other
    
    def __truediv__(self, other: float) -> Vector:
        return Vector(self.__x / other, self.__y / other, self.__z / other)
    
    def __itruediv__(self, other: float) -> Vector:
        return self / other
    
    def __eq__(self, other: object) -> bool:
        if type(other) != Vector:
            return False
        else:
            if abs(self.__x - other.__x) < 1e-12 and abs(self.__y - other.__y) < 1e-12 and abs(self.__z - other.__z) < 1e-12:
                return True
            else:
                return False
    
    def __getitem__(self, key) -> float:
        if key == 0:
            return self.__x
        elif key == 1:
            return self.__y
        elif key == 2:
            return self.__z
        else:
            raise Exception("Out of Vector bounds")
        
    def __setitem__(self, key, value: float) -> None:
        if key == 0:
            self.__x = value
        elif key == 1:
            self.__y = value
        elif key == 2:
            self.__z = value
        else:
            raise Exception("Out of Vector bounds")
        
    def toNumpyArray(self) -> npt.NDArray:
        return np.array([self.__x, self.__y, self.__z])
    
    @classmethod
    def fromNumpyArray(cls, array: npt.NDArray) -> Vector:
        array = array.flatten()
        assert len(array) == 3
        return Vector(array[0], array[1], array[2])
        
