import torch.nn as nn

from resnet.Layers import ResNetEncoder, ResNetDecoder
from resnet.Components import Conv2dPadded
from resnet.Tools import initialization
import torch.nn.functional as F

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



class ResNetSimCLR(nn.Module):
    
    def __init__(self, in_channels, out_channels):
        super(ResNetSimCLR,self).__init__()
        
        resnet = ResNetEncoder(in_channels, out_channels)
        num_ftrs = resnet.fc.in_features

        self.features = nn.Sequential(*list(resnet.children())[:-1])

        # projection MLP
        self.l1 = nn.Linear(num_ftrs, num_ftrs)
        self.l2 = nn.Linear(num_ftrs, out_channels)
        
        
    def forward(self, x):
        h = self.features(x)
        h = h.squeeze()

        x = self.l1(h)
        x = F.relu(x)
        x = self.l2(x)
        return h, x 