from abc import abstractmethod, ABCMeta
from Variable.Base import Base as BaseVariable
from utils.trasnform import as_array

class Base(metaclass=ABCMeta):
    def __call__(self, a_input: BaseVariable, *args, **kwargs):
        x = a_input.data
        y = self.forward(x)
        self.output = BaseVariable(as_array(y))
        self.output.set_creator(self)
        self.input = a_input
        return self.output

    @abstractmethod
    def forward(self, x):
        raise NotImplementedError

    @abstractmethod
    def backward(self, gy):
        raise NotImplementedError
