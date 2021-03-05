import torch
import torch.nn as nn
import torch.nn.functional as F

from resnet.Models import ResNetSimCLR
from SimCLR.Loss import NTCrossEntropyLoss
from SimCLR.Components import ProjectionHead

from tqdm import tqdm


class SimCLR:
  '''
  SimCLR class

  does training using contrastive loss and returns a trained resnet
  '''

  def __init__(self, in_channels=3, d_rep=1024, n_classes=10, *args, **kwargs):

    # define the model
    self.model = ResNetSimCLR(in_channels, d_rep, n_classes, *args, **kwargs)
    self.model = self.model.to(self.device)
  
  @property
  def device(self):
    return torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

  def train(self, dataset, batch_size, temperature,
            n_epochs=90, log_steps=100, ave_size=2000):
    
    # create train loader
    train_loader = torch.utils.data.DataLoader(
      dataset, batch_size=batch_size, shuffle=True, num_workers=2
    )

    # trainers
    criterion = NTCrossEntropyLoss(
      temperature, batch_size, self.device
    ).to(self.device)
    optimizer = torch.optim.Adam(self.model.parameters())
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
      optimizer, T_max=len(train_loader), eta_min=0, last_epoch=-1
    )

    losses = []

    for epoch in tqdm(range(n_epochs)):
      running_loss = 0
      for i, (xis, xjs) in enumerate(train_loader):
        self.optimizer.zero_grad()
        
        xis = xis.to(self.device)
        xjs = xjs.to(self.device)
        
        # Get representations and projections
        his, zis = self.model(xis)
        hjs, zjs= self.model(xjs)
        
        # normalize
        zis = F.normalize(zis, dim=1)
        zjs = F.normalize(zjs, dim=1)
        
        loss = criterion(zis, zjs)
        
        # optimize
        loss.backward()
        self.optimizer.step()

        # record loss
        if i%ave_size == ave_size-1:
          losses.append(running_loss / ave_size)
          running_loss = 0

      if epoch >= 10:
        self.scheduler.step()

    return self.get_model(), losses

  def get_model(self):
    net = self.model.resnet
    head_layer = self.model.projection_head.layers[0]
    return net, head_layer