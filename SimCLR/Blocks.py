import torch.nn as nn
from SimCLR.Tools import activation_func
#from Components import Conv2dPadded

#from functools import partial

class ResNetBlock(nn.Module):
  '''
  base block class that will include the logic for the skip connection
  This block has 3 convolution/batchnorm layers between the skip

      this isn't quite what's in the papter but it's close
      I would do better if I had a start middle and end block
  '''

  def __init__(self, in_channels, mid_channels,out_channels,
           resampling=1, conv=None, activation='relu',
           use_batch=True, use_dropout=True):
    super().__init__()
    self.in_channels = in_channels
    self.mid_channels = mid_channels
    self.out_channels = out_channels
    self.activation = activation
    self.activate = activation_func[activation]
    self.resampling = resampling
    self.use_batch = use_batch


    # set the convolution type
#    if conv is None:
#      self.conv1 = nn.Conv2d(
#          self.in_channels, self.mid_channels,
#          kernel_size=1,  bias=False
#        )
#      self.conv3 = nn.Conv2d(
#          self.mid_channels, self.mid_channels,
#          kernel_size=1,  bias=False
#        ),
#    else:
#      self.conv = conv

    # apply dropout
    if use_dropout:
      dropout = nn.Dropout2d(p=.2, inplace=True)
    else:
      dropout = nn.Identity()

    # here we have three blocks between the skips
    self.blocks = nn.Sequential(
      nn.Conv2d(self.in_channels, self.mid_channels, kernel_size=1, bias=False),
      nn.BatchNorm2d(self.mid_channels),
      self.activate,
      nn.Conv2d(self.mid_channels, self.mid_channels, kernel_size=3, bias=False),
      nn.BatchNorm2d(self.mid_channels),
      self.activate,
      nn.Conv2d(self.mid_channels, self.out_channels, kernel_size=1,  bias=False),
      nn.BatchNorm2d(self.out_channels),
      dropout
    )

    # implement the skip with convolution and batchnorm if needed
    if self.should_apply_skip:
      self.skip = nn.Sequential(
        nn.Conv2d(
          self.in_channels, self.out_channels,
          kernel_size=1, stride=self.resampling, bias=False
        ),
        nn.BatchNorm2d(self.out_channels)
      )
    else:
      self.skip = None

  def forward(self, x):

    # define the residual
    if self.should_apply_skip:
      residual = self.skip(x)
    else:
      residual = x

    # apply the blocks
    x = self.blocks(x)
    # add the skip
    x += residual
    return x

  # handy function for a convolution + batchnorm
  def conv_batchnorm(self, in_channels, out_channels, *args, **kwargs):

    if self.use_batch:
      return nn.Sequential(
        self.conv(
          in_channels, out_channels, *args, **kwargs
        ),
        nn.BatchNorm2d(out_channels)
      )
    else:
      return self.conv(
        in_channels, out_channels, *args, **kwargs
      )

  @property
  def should_apply_skip(self):
    return self.in_channels != self.out_channels
