"""Neural Network package for TensorForge."""

from tensorforge.nn import init
from tensorforge.nn.activations import ReLU, Sigmoid, Softmax, Tanh
from tensorforge.nn.init import kaiming_uniform_, ones_, uniform_, zeros_
from tensorforge.nn.linear import Linear
from tensorforge.nn.losses import CrossEntropyLoss, MSELoss
from tensorforge.nn.module import Module
from tensorforge.nn.parameter import Parameter
from tensorforge.nn.sequential import Sequential

__all__ = [
    "Module",
    "Parameter",
    "Linear",
    "ReLU",
    "Sigmoid",
    "Tanh",
    "Softmax",
    "MSELoss",
    "CrossEntropyLoss",
    "Sequential",
    "init",
    "uniform_",
    "zeros_",
    "ones_",
    "kaiming_uniform_",
]
