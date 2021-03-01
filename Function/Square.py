from Function.Base import Base as BaseFunction


class Square(BaseFunction):
    def forward(self, x):
        return x**2

    def backward(self, gy):
        x = self.inputs[0].data
        return 2 * x * gy


def square(x):
    return Square()(x)
