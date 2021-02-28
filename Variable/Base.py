import numpy as np


class Base:
    def __init__(self, data: np.ndarray):
        if data is not None:
            if not isinstance(data, np.ndarray):
                raise TypeError('{} 데이터 타입은 지원하지 않습니다.'.format(type(data)))
        self.data = data
        self.grad = None
        self.creator = None

    def set_creator(self, func):
        self.creator = func

    def backward(self):
        # if self.creator is not None:
        #     self.creator.input.grad = self.creator.backward(self.grad)
        #     self.creator.input.backward()
        if self.grad is None:
            self.grad = np.ones_like(self.data)
        funcs = [self.creator]
        while len(funcs) != 0:
            f = funcs.pop()
            # x, y = f.inputs, f.outputs
            # x.grad = f.backward(y.grad)
            # if x.creator is not None:
            #     funcs.append(x.creator)
            gys = [output.grad for output in f.outputs]
            gxs = f.backward(*gys)
            if not isinstance(gxs, tuple):
                gxs = (gxs, )

            for x, gx in zip(f.inputs , gxs):
                x.grad = gx
                if x.creator is not None:
                    funcs.append(x.creator)