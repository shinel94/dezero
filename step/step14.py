from Variable import Base as BaseVariable
from Function import add
import numpy as np

x = BaseVariable(np.array(3.0))
y = add(x, x)
print(y.data)
y.backward()
print(x.grad)

x.cleargrad()
y = add(add(x, x), x)
y.backward()
print(x.grad)