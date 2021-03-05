import torch 
import torch.nn as nn

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
      x = self.dropout(x)
    
    return x