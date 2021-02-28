from Function import Square, Exp
from Variable.Base import Base as BaseVariable
import numpy as np
from utils.differentiation import numerical_diff


f = Square()
x = BaseVariable(np.array(2.0))
dy = numerical_diff(f, x)
print(dy)


def f(x):
    A = Square()
    B = Exp()
    C = Square()
    return C(B(A(x)))


x = BaseVariable(np.array(0.5))
dy = numerical_diff(f, x)
print(dy)
