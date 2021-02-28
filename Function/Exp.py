from Function.Base import Base
import numpy as np


class Exp(Base):
    def forward(self, x):
        return np.exp(x)

    def backward(self, gy):
        x = self.input.data
        return np.exp(x) * gy


def exp(x):
    return Exp()(x)
