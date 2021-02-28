from Variable.Base import Base as BaseVariable
import numpy as np

x = BaseVariable(np.array([[1, 2, 3], [4, 5, 6]]))
print(x.shape)
print(x.ndim)
print(x.size)
print(x.dtype)
print(len(x))
print(x)