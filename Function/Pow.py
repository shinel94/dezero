from Variable.Base import Base as BaseVariable
from Function.Base import Base as BaseFunction


class Pow(BaseFunction):

    def __init__(self, c):
        self.c = c

    def forward(self, x):
        y = x ** self.c
        return y

    def backward(self, gy):
        x = self.inputs[0].data
        c = self.c
        gx = c * x ** (c - 1) * gy
        return gx

def pow(x, c):
    return Pow(c)(x)

BaseVariable.__pow__ = pow