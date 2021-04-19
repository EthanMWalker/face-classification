import torch.nn as nn

from SimCLR.Layers import ResNetEncoder, ResNetDecoder
from SimCLR.Components import Conv2dPadded, ProjectionHead
from SimCLR.Tools import initialization
import torch.nn.functional as F
from SimCLR.Loss import RingLoss

class ResNet(nn.Module):
  '''
  The final ResNet, made of an encoder and a decoder
  '''

  def __init__(self, in_channels, n_classes, init='xavier', *args, **kwargs):
    super(ResNet, self).__init__()
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

class RingLossResNet(nn.Module):

  def __init__(self, in_dim, n_classes, loss_weight, *args, **kwargs):
    super().__init__()

    self.resnet = ResNet(in_dim, n_classes, *args, **kwargs)
    self.loss = RingLoss(loss_weight)

  def forward(self, x, rep_only=False):
    out = self.resnet(x)
    if rep_only:
      return out
    loss = self.loss(x)
    return out, loss



class ResNetSimCLR(nn.Module):
    
  def __init__(self, in_channels, n_classes, d_hidden=1024, mlp_layers=2,
              *args, **kwargs):
    '''
    Parameters:
      in_channels (int): number of channels in the input
      d_rep (int): dimension of the representation, or the dimension of
          the output of the resnet
      n_classes (int): output dimension for the projection head
      d_hidden (int): hidden features in the projection head
      mlp_layers (int): number of layers in the projection head
    
    kwargs you might want to know:
      blocks_sizes (list(int)): list of the sizes of the hidden dimensions 
          in each of the blocks of the encoder
      blocks_layers (list(int)): list of the number of layers in each
          of the blocks
    
    The resnet defaults to a 16 layers
    '''

    super(ResNetSimCLR, self).__init__()
    self.init_func = initialization['orthogonal']

    self.in_channels = in_channels
    self.n_classes = n_classes

    self.resnet = ResNetEncoder(in_channels, *args, **kwargs)

    self.projection_head = ProjectionHead(
      self.resnet.blocks[-1].out_channels,
      n_classes, d_hidden, n_layers=mlp_layers
    )
      
      
  def forward(self, x):
    h = self.resnet(x)
    x = self.projection_head(h)
    return h, x 
