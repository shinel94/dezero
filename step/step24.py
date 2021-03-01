from Variable.Base import Base as BaseVariable
import numpy as np
from Function import sphere, matyas, goldstein

x = BaseVariable(np.array(1.0))
y = BaseVariable(np.array(1.0))
z = sphere(x, y)
z.backward()
print(x.grad, y.grad)


x = BaseVariable(np.array(1.0))
y = BaseVariable(np.array(1.0))
z = matyas(x, y)
z.backward()
print(x.grad, y.grad)


x = BaseVariable(np.array(1.0))
y = BaseVariable(np.array(1.0))
z = goldstein(x, y)
z.backward()
print(x.grad, y.grad)


