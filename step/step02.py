from Function import Square, Exp
from Variable import Base as BaseVariable
import numpy as np

A = Square()
B = Exp()
C = Square()

x = BaseVariable(np.array(0.5))
a = A(x)
b = B(a)
y = C(b)
print(y.data)
