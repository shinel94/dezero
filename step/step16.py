from Variable.Base import Base as BaseVariable
from Function import square, add
import numpy as np

x = BaseVariable(np.array(2.0))
a = square(x)
y = add(square(a), square(a))
y.backward()
print(y.data)
print(x.grad)

a.cleargrad()
x.cleargrad()
z = add(square(a), square(x))
z.backward()
print(z.data)
print(x.grad)