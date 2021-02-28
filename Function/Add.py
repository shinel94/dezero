from Function.Base import Base, BaseVariable


class Add(Base):
    def forward(self, x0, x1):
        return x0 + x1

    def backward(self, gy):
        return gy, gy


def add(x0, x1):
    return Add()(x0, x1)

BaseVariable.__add__ = add