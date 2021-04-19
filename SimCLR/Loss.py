import torch
import torch.nn as nn
import numpy as np

class MarginalLoss(nn.Module):
  '''
  Marginal loss
  '''

  def __init__(self, batch_size, device):
    super().__init__()
    
    self.batch_size = batch_size
    self.device = device
    self.theta = 1.2
    self.xi = .3
    
    
  def forward(self,x,y):
    m_scalar =  1/(self.batch_size**2 - self.batch_size)

    total = 0
    norms = [i/torch.linalg.norm(i) for i in x] 
    indicator = [[1 if i==j else -1 for i in y] for j in y]

   

    for i in range(self.batch_size):
      for j in range(i, self.batch_size):
        norm_value = torch.linalg.norm( norms[i] - norms[j],2)**2
        
        total += self.xi +indicator[i][j]* max((self.theta - norm_value), 0)
            
    return m_scalar*total
 
class RingLoss(nn.Module):
    '''
    Ring loss based on the paper
    Ring loss: Convex Feature Normalization for Face Recognition
    https://arxiv.org/pdf/1803.00130.pdf

    this is to be used in conjunction with another loss, it encourages the
    model to place logits within a "ring"
    it requires its own optimizer as well
    '''

    def __init__(self, loss_weight):
      super().__init__()
      self.weight = loss_weight
      self.radius = nn.Parameter(torch.Tensor(1))


    def forward(self, x):

      # if the radius is negative then set it to the mean
      if self.radius.data.item() < 0:
        self.radius.data.fill_(x.mean().item())

      # compute loss
      x = torch.linalg.norm(x, ord=2, dim=1)
      x = x - self.radius
      x = torch.pow(torch.abs(x), 2).mean()
      x = x/2
      loss = x * self.weight

      return loss
      


class NTCrossEntropyLoss(nn.Module):
  '''
  NT cross entropy loss
  '''

  def __init__(self, temperature, batch_size, device):
    super().__init__()

    self.temperature = temperature
    self.batch_size = batch_size
    self.device = device
    self.cosine_similarity = nn.CosineSimilarity(dim=-1)
    self.criterion = nn.CrossEntropyLoss(reduction='sum')

  @property
  def negative_representations_mask(self):
    pos_mask = np.zeros(2*self.batch_size)
    pos_mask[[0,self.batch_size]] = 1
    neg_mask = ~pos_mask.astype(bool)
    return torch.from_numpy(neg_mask).to(self.device)
  
  def similarity(self,x,y):
    tmps = []
    for i in range(2*self.batch_size):
      tmp = self.cosine_similarity(x, torch.roll(y,-i,dims=0))
      tmps.append(tmp)
    
    return torch.stack(tmps)
    
  def forward(self, rep1, rep2):

    dbl_batch = 2*self.batch_size

    reps = torch.cat([rep1, rep2], dim=0)

    sims = self.similarity(reps, reps)
    pos_sims = sims[self.batch_size].view(dbl_batch,1)
    neg_sims = sims[self.negative_representations_mask]
    neg_sims = neg_sims.view(dbl_batch, -1)

    logits = torch.cat([pos_sims, neg_sims], dim=1)
    logits /= self.temperature

    labels = torch.zeros(dbl_batch).to(self.device).long()

    loss = self.criterion(logits, labels)
    return loss / dbl_batch
