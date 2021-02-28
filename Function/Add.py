from Function import Base


class Add(Base):
    def forward(self, x0, x1):
        return x0 + x1

    def backward(self, gy):
        pass


def add(x0, x1):
    return Add()(x0, x1)