import torch.nn as nn

class Conv2dPadded(nn.Conv2d):
  '''
  A padded convolution filter (I think thats the right name)
  This adds padding to nn.Conv2d based on the kernel size
  '''

  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)
    # adds padding based on the kernal size
    self.padding = (self.kernel_size[0] // 2, self.kernel_size[1] // 2)

