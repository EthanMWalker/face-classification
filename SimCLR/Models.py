import torch
import torch.nn as nn

from resnet.Models import ResNet
from SimCLR.Loss import NTCrossEntropyLoss


class SimCLR(nn.Module):
  '''
  SimCLR class

  does training using contrastive loss and returns a trained resnet
  '''

  def __init__(self, *args, **kwargs):
    super().__init__()

    self.resnet = ResNet()
    self.criterion = NTCrossEntropyLoss().to(self.device)
    self.optimizer = torch.optim.Adam(self.resnet.parameters())

    return
  
  @property
  def device(self):
    return torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

  def train(self):
    pass