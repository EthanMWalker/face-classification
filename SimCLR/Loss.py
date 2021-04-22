import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

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


class RingLoss(nn.Module):
  '''
  Ring loss based on the paper 
    Ring loss: Convex Feature Normalization for Face Recognition
    https://arxiv.org/pdf/1803.00130.pdf

  this is to be used within a model and in conjunction with another 
  loss, it encourages the model to place logits within a "ring"
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

class AngularSoftmax(nn.Module):
  '''
  Angular Softmax loss based on the paper
    SphereFace: Deep Hypersphere Embedding for Face Recognition
    https://arxiv.org/pdf/1704.08063.pdf

  This is to be used within a model as it needs to be optimized with the
  model
  '''
  
  def __init__(self, in_dim, out_dim, m=1.35, eps=1e-10):
    super().__init__()
    self.m = m
    self.eps = eps
    self.s = 64.0
    self.W = nn.Linear(in_dim, out_dim, bias=False)
  
  def forward(self, x, y, repr=False):

    for param in self.W.parameters():
      param = F.normalize(param, p=2, dim=1)

    # norms = torch.linalg.norm(x, ord=2, dim=1)
    x = F.normalize(x, p=2, dim=1)

    prods = self.W(x)
    if repr:
      return prods

    cos_theta = torch.diag(prods.transpose(0,1)[y])
    cos_theta = torch.clamp(cos_theta, -1+self.eps, 1-self.eps)
    numer_theta = torch.acos(cos_theta)
    numer = self.s * torch.cos(self.m * numer_theta)

    denom = [
      self.s*torch.cat([prods[i,:yi],prods[i,yi+1:]]) for i,yi in enumerate(y)
    ]
    denom = torch.stack(denom)
    denom = torch.exp(numer) + torch.sum(torch.exp(denom), dim=1)

    loss = numer - torch.log(denom)

    return -torch.mean(loss)
    