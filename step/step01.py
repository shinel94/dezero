import numpy as np
from Variable.Base import Base as BaseVariable
from Function.Square import Square

data = np.array(10)

x = BaseVariable(data)
print(data)

f = Square()
y = f(x)
print(y.data)



