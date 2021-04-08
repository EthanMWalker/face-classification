import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from torchlars import LARS

from SimCLR.Models import ResNetSimCLR
from SimCLR.Loss import NTCrossEntropyLoss

from tqdm import tqdm

class BaseModel:
  def __init__(self, model=None, in_channels=3, n_classes=5,
              batch_size=128, *args, **kwargs):
    if model is None:
      # define the model
      self.model = ResNetSimCLR(in_channels, n_classes, *args, **kwargs)
    else:
      # use a pretrained model
      self.model = model
    self.model = self.model.to(self.device)
    self.num_params = sum(p.numel() for p in self.model.parameters())
    
    self.batch_size = batch_size
    
  @property
  def device(self):
    return torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
  
  def load_model(self, path):
    self.model.load_state_dict(torch.load(path)['model_state_dict'])

  def load_data(self, dataset):
    # create data loader
    data_loader = DataLoader(dataset, batch_size=self.batch_size, 
                    drop_last=True, shuffle=True, num_workers=2)
    return data_loader

class Train(BaseModel):

  def __init__(self, model=None, in_channels=3, n_classes=5,
              batch_size=64, *args, **kwargs):

    super().__init__(
      model, in_channels, n_classes, batch_size, *args, **kwargs
    )
    
  def return_model(self):
    return self.model.resnet.state_dict(), self.model.projection_head.layers[0].state_dict()
  
  def train(self, dataloader, temperature, ckpt_path, n_epochs=90, 
          save_size=10):
    
    # trainers
    criterion = NTCrossEntropyLoss(temperature, self.batch_size, 
                                   self.device).to(self.device)
    optimizer = LARS(torch.optim.SGD(self.model.parameters(), lr=4))
    # optimizer = optimizer.to(self.device)

    losses = []

    for epoch in range(n_epochs):
      with tqdm(total=len(dataloader)) as progress:
        running_loss = 0
        i = 0
        for (xis, xjs), _ in dataloader:
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
          running_loss += loss.item()
        
          # optimize
          loss.backward()
          optimizer.step()
        
          # update tqdm
          progress.set_description('train loss:{:.4f}'.format(loss.item()))
          progress.update()
        
          # record loss
          if i%save_size == (save_size-1):
            losses.append(running_loss / save_size)
            running_loss = 0
        
        # save model
        if epoch%10 == 0:
          torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': losses,},ckpt_path + f'{epoch}')
    

    return self.return_model(), losses

class FineTune(BaseModel):
  def __init__(self, model=None, in_channels=3, n_classes=5,
              batch_size=30, *args, **kwargs):
    super().__init__(
      model, in_channels, n_classes, batch_size, *args, **kwargs
    )
  
  def fine_tune(self, dataloader, ckpt_path, n_epochs=90, save_size=10):
    '''
    This fine tuning is designed for normal cross entropy loss training
    '''

    # trainers
    criterion = nn.CrossEntropyLoss().to(self.device)
    
    optimizer = torch.optim.SGD(
      self.model.parameters(), lr=.001, momentum=.9
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
      optimizer, T_max=len(dataloader), eta_min=0, last_epoch=-1
    )

    losses = []

    for epoch in range(n_epochs):
      with tqdm(total=len(dataloader)) as progress:
        running_loss = 0
        i = 0
        for data in dataloader:
          i += 1
          optimizer.zero_grad()

          x = data[0].to(self.device)
          y = data[1].to(self.device)
        
          # Get representations and projections
          h, z = self.model(x)
        
          loss = criterion(z, y)
          running_loss += loss.item()
        
          # optimize
          loss.backward()
          optimizer.step()
        
          # update tqdm
          progress.set_description('tune loss:{:.4f}'.format(loss.item()))
          progress.update()
        
          # record loss
          if i%save_size == 0:
            losses.append(running_loss / save_size)
            running_loss = 0

        if epoch >= 10:
          scheduler.step()
        
        # save model
        if epoch%10 == 0:
          torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,},ckpt_path)

    return self.model, losses

class Validate(BaseModel):
  '''
  Validate class

  Takes a trained ResNetSimCLR model and computes accuracy 
  '''
  
  def __init__(self, model=None, in_channels=3, n_classes=5,
              batch_size=128, *args, **kwargs):
    super().__init__(
      model, in_channels, n_classes, batch_size, *args, **kwargs
    )

  def validate(self, dataloader):
    # validate test/validation set on trained model
    
    # load trained model
    optimizer = torch.optim.Adam(self.model.parameters())
    
    self.model.eval()
    
    total, correct = 0, 0
    actual = []
    predicted = []

    # validate 
    with tqdm(total=len(dataloader)) as progress:
      for x, y in dataloader:

        optimizer.zero_grad()
      
        x = x.to(self.device)
        y = y.to(self.device)
      
        # Get representations and projections
        h, z = self.model(x)

        _, predict = torch.max(z.data, 1)
      
        total += y.shape[0]
        correct += (predict == y).sum().item()

        actual.extend(y.cpu())
        predicted.extend(predict.cpu())

        # update tqdm
        progress.set_description('validating')
        progress.update()
          
    return correct/total, actual, predicted


class SimCLR:
  def __init__(self, model=None, in_channels=3, n_classes=5, 
              train_batch_size=80, tune_batch_size=10, train_temp=.5):

    if model is None:
      self.trainer = Train(
        None, in_channels, n_classes, train_batch_size
      )
    else:
      self.trainer = Train(model, batch_size=train_batch_size)
    
    self.tune_batch_size = tune_batch_size
    self.train_batch_size = train_batch_size
    self.train_temp = train_temp
    self.n_classes = n_classes
  
  def make_tuner(self, model):
    '''
    Parameters:
      model (tuple(ResNet, nn.Linear)): the state dicts of the resnet
        and first layer of the projection head of for a simclr model
    '''
    resnet_dict = model[0]
    head_dict = model[1]
    in_channels = self.trainer.model.in_channels
    n_classes = self.trainer.model.n_classes

    model = ResNetSimCLR(
      in_channels, self.n_classes, mlp_layers=3, blocks_layers=[3,3,3,3]
    )

    model.resnet.load_state_dict(resnet_dict)
    model.projection_head.layers[0].load_state_dict(head_dict)

    self.tuner = FineTune(model, batch_size=self.tune_batch_size)
  
  def make_validator(self, model):
    '''
    Parameters:
      model (ResNetSimCLR): a full resnet with projection head
    '''
    self.validator = Validate(model, batch_size=self.tune_batch_size)
  
  def train(self, data, epochs, path):
    dataloader = self.trainer.load_data(data)
    model, losses = self.trainer.train(
      dataloader, self.train_temp, path, epochs
    )
    return model, losses
  
  def tune(self, data, epochs, path):
    dataloader = self.tuner.load_data(data)
    model, losses = self.tuner.fine_tune(
      dataloader, path, epochs
    )
    return model, losses
  
  def validate(self, data):
    dataloader = self.validator.load_data(data)
    acc, actual, predicted = self.validator.validate(dataloader)
    return acc, actual, predicted

  def full_model_maker(self, train_data, tune_data, val_data, n_cycles=10, 
                      train_epochs=10, tune_epochs=50, 
                      train_path='training.tar', tune_path='tuning.tar'):
    
    train_loss = []
    tune_loss = []
    accuracy = []

    for i in range(n_cycles):
      model, losses = self.train(train_data,train_epochs, train_path)
      train_loss.extend(losses)

      self.make_tuner(model)
      model, losses = self.tune(tune_data, tune_epochs, tune_path)
      tune_loss.extend(losses)

      self.make_validator(self.tuner.model)
      acc, actual, predicted = self.validate(val_data)
      accuracy.append(acc)
#    
    results = (
      self.tuner.model, train_loss, tune_loss,
     accuracy, actual, predicted
    )
    return results
