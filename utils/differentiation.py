from Function.Base import Base as BaseFunction
from Variable.Base import Base as BaseVariable


def numerical_diff(f: BaseFunction, x: BaseVariable, eps=1e-4):
    x0 = BaseVariable(x.data - eps)
    x1 = BaseVariable(x.data + eps)
    y0 = f(x0)
    y1 = f(x1)
    return (y1.data - y0.data) / (2 * eps)
