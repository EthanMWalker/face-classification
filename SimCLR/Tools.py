import torch
import torch.nn as nn
from functools import partial


# dictionary for storing the different activation functions
# this will make it easier to change in the future
activation_func = {
  'relu':         nn.ReLU(inplace=True),
  'leaky_relu':   nn.LeakyReLU(negative_slope=0.01, inplace=True),
  'selu':         nn.SELU(inplace=True),
  'elu':          nn.ELU(inplace=True),
  'hardshrink':   nn.Hardshrink(),
  'none':         nn.Identity()
}

# define a dictionary for the initialization types
initialization = {
  'xavier': partial(nn.init.xavier_normal_, gain=1.0),
  'he': partial(
      nn.init.kaiming_normal_, a=0, mode='fan_in',
      nonlinearity='leaky_relu'
  ),
  'orthogonal': partial(nn.init.orthogonal_, gain=1)
}