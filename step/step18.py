from Variable.Base import Base as BaseVariable
import numpy as np
from Function import add, square
from Config.Base import Base as Config
from utils.configs import using_config, no_grad

x0 = BaseVariable(np.array(1.0))
x1 = BaseVariable(np.array(1.0))
t = add(x0, x1)
y = add(x0, t)
y.backward()

print(y.grad, t.grad)
print(x0.grad, x1.grad)


Config.enable_backprop = True
x = BaseVariable(np.ones((100, 100, 100)))
y = square(square(square(x)))
y.backward()

Config.enable_backprop = False
x = BaseVariable(np.ones((100, 100, 100)))
y = square(square(square(x)))

with using_config('enable_backprop', False):
    x = BaseVariable(np.array(2.0))
    y = square(x)
    print(y.data)

with no_grad():
    x = BaseVariable(np.array(2.0))
    y = square(x)
    print(y.data)