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

class ProjectionHead(nn.Module):
  '''
  Projection Head for SimCLR-2 model
  '''

  def __init__(self, d_in, d_out, d_hidden, n_layers, dropout=.1):
    super().__init__()

    self.d_in = d_in
    self.d_out = d_out
    self.d_hidden = d_hidden

    self.activation = nn.RELU(inplace=True)
    self.dropout = nn.Dropout(dropout)

    self.layers = nn.ModuleList(
      [
        nn.Linear(d_in, d_hidden, bias=False),
        *[
          nn.Linear(d_hidden, d_hidden, bias=False) for _ in range(n_layers-2)
        ],
        nn.Linear(d_hidden, d_out, bias=False)
      ]
    )

  def forward(self, x):

    for layer in self.layers:
      x = layer(x)
      x = self.activation(x)
      x = self.dropout(x)
    
    return x