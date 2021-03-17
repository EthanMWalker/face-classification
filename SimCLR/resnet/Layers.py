import torch.nn as nn

from resnet.Blocks import ResNetBlock
from resnet.Tools import activation_func


class ResNetLayer(nn.Module):
  '''
  A ResNet layer class that creates a layer from n blocks. 
  This layer will have one block that scales then the rest compute
  '''

  def __init__(self, in_channels, out_channels, block=ResNetBlock,
                n=3, activation='relu', *args, **kwargs):
    super().__init__()
    self.in_channels = in_channels
    self.out_channels = out_channels

    # if there is scaling between layers we need to account for that
    if in_channels != out_channels:
      resampling = 2
    else:
      resampling = 1

    # not exactly what the paper describes, but it preforms well
    self.blocks = nn.Sequential(
      block(
        in_channels, out_channels, resampling=resampling,
        *args, **kwargs
      ),
      *[
        block(
          out_channels, out_channels,
          *args, **kwargs, resampling=1
        ) for _ in range(n-1)
      ],
      nn.BatchNorm2d(out_channels),
      activation_func[activation]
    )

  def forward(self, x):
    return self.blocks(x)


class ResNetEncoder(nn.Module):
  '''
  The encoder expands the feature size by stacking layers with
  increasing feature sizes
  I tried to base this on table 1 of the paper
  '''

  def __init__(self, in_channels=3, blocks_sizes=[2**i for i in [5, 6, 7, 8]],
           blocks_layers=[4, 4, 4, 4], activation='relu', block=ResNetBlock,
           *args, **kwargs):
    super().__init__()
    self.blocks_sizes = blocks_sizes

    # this first block will bring the data in to be processed
    # this is based on table 1 of the paper
    self.first_block = nn.Sequential(
      nn.Conv2d(
        in_channels, self.blocks_sizes[0],
        kernel_size=7, stride=2, padding=3, bias=False
      ),
      nn.BatchNorm2d(self.blocks_sizes[0]),
      activation_func[activation],
      nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
    )

    # organize the in and out sizes in the adjoining pairs
    in_out_sizes = list(zip(blocks_sizes[:-1], blocks_sizes[1:]))

    # the first layer maintains the size from the first block
    # but the subsequent layers begin to scale
    self.blocks = nn.ModuleList(
      [
        ResNetLayer(
          blocks_sizes[0], blocks_sizes[0],
          n=blocks_layers[0], activation=activation, block=block,
          *args, **kwargs
        ),
        *[ResNetLayer(
          in_channels, out_channels,
          n=n, activation=activation, block=block,
          *args, **kwargs
        ) for (in_channels, out_channels), n in zip(in_out_sizes, blocks_layers[1:])]
      ]
    )

    self.AAP = nn.AdaptiveAvgPool2d((1, 1))

  def forward(self, x):
    x = self.first_block(x)
    for block in self.blocks:
      x = block(x)
    x = self.AAP(x)
    x = x.view(x.shape[0], -1)
    return x


class ResNetDecoder(nn.Module):
  '''
  The decoder takes the output from the encoder and creates a
  final output
  This uses an adaptive pool and a linear layer
  '''

  def __init__(self, in_features, n_classes):
    super().__init__()
    self.linear = nn.Linear(in_features, n_classes)

  def forward(self, x):
    x = self.linear(x)
    return x
