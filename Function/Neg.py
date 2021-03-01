from Function.Base import Base as BaseFunction
from Variable.Base import Base as BaseVariable

class Neg(BaseFunction):
    def forward(self, x):
        return -x

    def backward(self, gy):
        return -gy

def neg(x):
    return Neg()(x)

BaseVariable.__neg__ = neg