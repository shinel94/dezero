from Variable.Base import Base as BaseVariable
from Function import square, exp
import numpy as np

x = BaseVariable(np.array(0.5))
a = square(x)
b = exp(a)
y = square(b)

y.grad = np.array(1.0)
y.backward()
print(x.grad)


x = BaseVariable(np.array(0.5))
y = square(exp(square(x)))
y.grad = np.array(1.0)
y.backward()
print(x.grad)

x = BaseVariable(np.array(0.5))
y = square(exp(square(x)))
y.backward()
print(x.grad)