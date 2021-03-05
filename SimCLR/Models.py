import torch
import torch.nn

from resnet.Models import ResNet
from SimCLR.Loss import NTXentLoss


class SimCLR(nn.Module):
  '''
  SimCLR class

  does training using contrastive loss and returns a trained resnet
  '''

  def __init__(self, *args, **kwargs):
    super().__init__()

    return

