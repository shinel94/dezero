import weakref
from abc import abstractmethod, ABCMeta
from Variable.Base import Base as BaseVariable
from utils.trasnform import as_array
from Config.Base import Base as Config


class Base(metaclass=ABCMeta):
    # def __call__(self, a_input: BaseVariable, *args, **kwargs):
    def __call__(self, *a_inputs, **kwargs):

        # x = a_input.data
        # y = self.forward(x)
        # self.output = BaseVariable(as_array(y))
        # self.output.set_creator(self)
        # self.input = a_input
        xs = [x.data for x in a_inputs]
        ys = self.forward(*xs)
        if not isinstance(ys, tuple):
            ys = (ys, )
        outputs = [BaseVariable(as_array(y)) for y in ys]

        if Config.enable_backprop:
            self.generation = max([x.generation for x in a_inputs])
            for output in outputs:
                output.set_creator(self)
            self.inputs = a_inputs
            self.outputs = [weakref.ref(output) for output in outputs]
        return outputs if len(outputs) > 1 else outputs[0]

    @abstractmethod
    def forward(self, *x):
        raise NotImplementedError

    @abstractmethod
    def backward(self, gy):
        raise NotImplementedError
