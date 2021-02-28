from Variable.Base import Base as BaseVariable
import numpy as np
from Function import Add, add

xs = [BaseVariable(np.array(2)), BaseVariable(np.array(3))]
f = Add()
ys = f(*xs)
y = ys
print(y.data)

xs = [BaseVariable(np.array(2)), BaseVariable(np.array(3))]
f = Add()
ys = f(*xs)
y = ys
print(y.data)

x0 = BaseVariable(np.array(2))
x1 = BaseVariable(np.array(3))
y = add(x0, x1)
print(y.data)