from Variable.Base import Base as BaseVariable
from Function import square
import numpy as np

for i in range(100):
    x = BaseVariable(np.random.randn(10000))
    y = square(square(square(x)))
    y.backward()
    print(x.data)
