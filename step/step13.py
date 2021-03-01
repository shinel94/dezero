from Variable.Base import Base as BaseVariable
import numpy as np
from Function import add, square

x = BaseVariable(np.array(2.0))
y = BaseVariable(np.array(3.0))

z = add(square(x), square(y))
z.backward()
print(z.data)
print(x.grad)
print(y.grad)