from Variable.Base import Base as BaseVariable
import numpy as np
from Function import add, mul

x = BaseVariable(np.array(2.0))
y = x + np.array(3.0)
print(y)

x = BaseVariable(np.array(2.0))
y = x + 3.0
print(y)


x = BaseVariable(np.array(2.0))
y = 3.0 + x
print(y)

x = BaseVariable(np.array(2.0))
y = 3.0 * x + 2
print(y)

x = BaseVariable(np.array([1.0]))
y = np.array([2.0]) + x
print(y)