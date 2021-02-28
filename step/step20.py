from Variable.Base import Base as BaseVariable
import numpy as np
from Function import add, mul

a = BaseVariable(np.array(3.0))
b = BaseVariable(np.array(2.0))
c = BaseVariable(np.array(1.0))

y = add(mul(a, b), c)

y.backward()

print(y)
print(a.grad)
print(b.grad)

a.cleargrad()
b.cleargrad()
c.cleargrad()
y = a * b + c
y.backward()

print(y)
print(a.grad)
print(b.grad)

