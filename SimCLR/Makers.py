import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from SimCLR.Models import ResNet
from SimCLR.Loss import MarginalLoss

from tqdm import tqdm

class BaseModel:
  def __init__(self, model=None, in_channels=3, n_classes=10,
              batch_size=128, *args, **kwargs):
    if model is None:
      # define the model
      self.model = ResNet(in_channels, n_classes, *args, **kwargs)
    else:
      # use a pretrained model
      self.model = model
    self.model = self.model.to(self.device)
    self.num_params = sum(p.numel() for p in self.model.parameters())
    
    self.batch_size = batch_size
    
  @property
  def device(self):
    return torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
  
  def load_model(self, path):
    self.model.load_state_dict(torch.load(path)['model_state_dict'])

  def load_data(self, dataset):
    # create data loader
    data_loader = DataLoader(dataset, batch_size=self.batch_size, 
                    drop_last=True, shuffle=True, num_workers=2)
    return data_loader

class Train(BaseModel):

  def __init__(self, model=None, in_channels=3, n_classes=10,
              batch_size=128, *args, **kwargs):

    super().__init__(
      model, in_channels, n_classes, batch_size, *args, **kwargs
    )
    
  def return_model(self):
    return self.model.state_dict()
  
  def train(self, dataloader, ckpt_path, n_epochs=90, 
          save_size=10):
    
    # trainers
    criterion = MarginalLoss(self.batch_size, 
                                   self.device).to(self.device)
    balance = .5
    criterionCE = nn.CrossEntropyLoss().to(self.device)
    optimizer = torch.optim.SGD(
            self.model.parameters(),lr=.001, momentum=.9
            )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=len(dataloader),eta_min=0, last_epoch=-1
            )

    losses = []

    for epoch in range(n_epochs):
      with tqdm(total=len(dataloader)) as progress:
        running_loss = 0
        i = 0
        for x, y in dataloader:
          i += 1
          optimizer.zero_grad()
        
          x = x.to(self.device)
          y = y.to(self.device)
        
          # Get representations and projections
          out =  self.model(x)
        
        
          loss = criterionCE(out,y) 
          loss += balance*criterion(out,y)
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
        if epoch >= 10:
            scheduler.step()
        
        # save model
        if epoch%10 == 0:
          torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss},ckpt_path)
    

    return self.model, losses#self.return_model(), losses


class Validate(BaseModel):
  '''
  Validate class

  Takes a trained ResNet model and computes accuracy 
  '''
  
  def __init__(self, model=None, in_channels=3, n_classes=10,
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
        out  = self.model(x)
        

        _, predict = torch.max(out, 1)
      
        total += y.shape[0]
        correct += (predict == y).sum().item()

        actual.extend(y.cpu())
        predicted.extend(predict.cpu())

        # update tqdm
        progress.set_description('validating')
        progress.update()
          
    return correct/total, actual, predicted


class SimCLR:
  def __init__(self, model=None, in_channels=3, n_classes=10, 
              train_batch_size=256, val_batch_size=256):

    if model is None:
      self.trainer = Train(
        None, in_channels, n_classes, train_batch_size
      )
    else:
      self.trainer = Train(model, batch_size=train_batch_size)
    
    self.train_batch_size = train_batch_size
    self.val_batch_size = val_batch_size
  
  
  def make_validator(self, model):
    '''
    Parameters:
      model (ResNetSimCLR): a full resnet with projection head
    '''
    #in_channels = self.trainer.model.in_channels
    #n_classes = self.trainer.model.n_classes
    #model 
    self.validator = Validate(model, batch_size=self.val_batch_size)
  
  def train(self, data, epochs, path):
    dataloader = self.trainer.load_data(data)
    model, losses = self.trainer.train(
      dataloader, path, epochs
    )
    return model, losses
  
  
  def validate(self, data):
    dataloader = self.validator.load_data(data)
    acc, actual, predicted = self.validator.validate(dataloader)
    return acc, actual, predicted

  def full_model_maker(self, train_data, val_data, n_cycles=10, 
                      train_epochs=10, 
                      train_path='training.tar' ):
    
    train_loss = []
    accuracy = []

    for i in range(n_cycles):
      model, losses = self.train(train_data,train_epochs, train_path)
      train_loss.extend(losses)


      self.make_validator(model)
      acc, actual, predicted = self.validate(val_data)
      accuracy.append(acc)
    
    results = (
      model, train_loss, 
      accuracy, actual, predicted
    )
    return results
