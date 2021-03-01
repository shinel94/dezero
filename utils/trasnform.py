import numpy as np
from Variable.Base import Base as BaseVariable


def as_array(x):
    if np.isscalar(x):
        return np.asarray(x)
    return x


def as_variable(obj):
    if isinstance(obj, BaseVariable):
        return obj
    return BaseVariable(obj)