from Function import Square, Exp
from Variable.Base import Base as BaseVariable
import numpy as np

A = Square()
B = Exp()
C = Square()

x = BaseVariable(np.array(0.5))
a = A(x)
b = B(a)
y = C(b)

y.grad = np.array(1.0)
b.grad = C.backward(y.grad)
a.grad = B.backward(b.grad)
x.grad = A.backward(a.grad)
print(x.grad)