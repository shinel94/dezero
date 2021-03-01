from Function.Base import Base as BaseFunction
from Variable.Base import Base as BaseVariable
from utils.trasnform import as_array

class Sub(BaseFunction):

    def forward(self, x0, x1):
        y = x0 - x1
        return y

    def backward(self, gy):
        return gy, -gy


def sub(x0, x1):
    x1 = as_array(x1)
    return Sub()(x0, x1)

def rsub(x0, x1):
    x1 = as_array(x1)
    return Sub()(x1, x0)

BaseVariable.__sub__ = sub
BaseVariable.__rsub__ = rsub
