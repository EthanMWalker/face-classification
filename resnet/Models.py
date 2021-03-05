import torch.nn as nn

from resnet.Layers import ResNetEncoder, ResNetDecoder
from resnet.Components import Conv2dPadded
from resnet.Tools import initialization

class ResNet(nn.Module):
  '''
  The final ResNet, made of an encoder and a decoder
  '''

  def __init__(self, in_channels, n_classes, init='xavier', *args, **kwargs):
    super().__init__()
    self.init_func = initialization[init]
    self.encoder = ResNetEncoder(in_channels, *args, **kwargs)
    self.decoder = ResNetDecoder(
    self.encoder.blocks[-1].out_channels, n_classes
    )
    self.initialize()

  def initialize(self):

    # define the initialize function we will use on the modules
    def init(w):
    if type(w) in [nn.Linear, nn.Conv2d, Conv2dPadded]:
      self.init_func(w.weight)
    # apply the initializations
    self.encoder.apply(init)
    self.decoder.apply(init)

  def forward(self, x):
    x = self.encoder(x)
    x = self.decoder(x)
    return x