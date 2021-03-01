from Variable.Base import Base as BaseVariable
import numpy as np
import Function

x = BaseVariable(np.array(2.0))
y = -x
print(y)

x = BaseVariable(np.array(2.0))
y1 = 2.0 - x
y2 = x - 1.0
print(y1)
print(y2)


x = BaseVariable(np.array(8.0))
y1 = 2.0 / x
y2 = x / 4.0
print(y1)
print(y2)

x = BaseVariable(np.array(2.0))
y = x ** 3
print(y)
