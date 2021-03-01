from Function.Base import Base, BaseVariable
from utils.trasnform import as_array


class Add(Base):
    def forward(self, x0, x1):
        return x0 + x1

    def backward(self, gy):
        return gy, gy


def add(x0, x1):
    x1 = as_array(x1)
    return Add()(x0, x1)

BaseVariable.__add__ = add
BaseVariable.__radd__ = add
