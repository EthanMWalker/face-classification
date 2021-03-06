import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader


from resnet.Models import ResNetSimCLR
from Loss import NTCrossEntropyLoss

from tqdm import tqdm




class SimCLR:
  '''
  SimCLR class

  does training using contrastive loss and returns a trained resnet
  '''

  def __init__(self, in_channels=3, d_rep=1024, n_classes=10, batch_size=128,
               *args, **kwargs):

    # define the model
    self.model = ResNetSimCLR(in_channels, d_rep, n_classes, *args, **kwargs)
    self.model = self.model.to(self.device)
    
    self.batch_size = batch_size
  
  @property
  def device(self):
   
      #return torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    return 'cpu'

  def load_data(self, dataset, s, input_shape):
    # create data loader
    data_loader = DataLoader(dataset, batch_size=self.batch_size, 
                    drop_last=True, shuffle=True, num_workers=2)
    
    return data_loader
      

  def train(self, dataloader, temperature, n_epochs=90, log_steps=100, 
            ave_size=2000,save_size=10):
    


    # trainers
    criterion = NTCrossEntropyLoss(temperature, self.batch_size, 
                                   self.device).to(self.device)
    
    optimizer = torch.optim.Adam(self.model.parameters())
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
      optimizer, T_max=len(dataloader), eta_min=0, last_epoch=-1
    )

    losses = []

    for epoch in tqdm(range(n_epochs)):
      running_loss = 0
      i = 0
      for (xis, xjs), _ in dataloader:
        print("i",i)
        i += 1
        optimizer.zero_grad()
        
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
        optimizer.step()

        # record loss
        if i%save_size == 0:
          losses.append(running_loss / save_size)
          running_loss = 0

      if epoch >= 10:
        scheduler.step()

    return self.get_model(), losses

  def get_model(self):
    net = self.model.resnet
    head_layer = self.model.projection_head.layers[0]
    return net, head_layer