import torch
import torch.nn as nn
import numpy as np

class NTXentLoss(nn.Module):
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


  # @property
  # def negative_representations_mask(self):
    
  #   # create identity map
  #   identity_mask = np.eye(2*self.batch_size)

  #   # maps for the first representation to the second and the second to
  #   # the first
  #   lh_mask = np.eye(2*self.batch_size, k=self.batch_size)
  #   rh_mask = np.eye(2*self.batch_size, k=-self.batch_size)

  #   # add masks to get the positive mask
  #   postive_mask = identity_mask + lh_mask + rh_mask

  #   # get the negative mask and send it to the torch device
  #   negative_mask = 1 - postive_mask
  #   mask = torch.from_numpy(negative_mask).type(torch.bool).to(self.device)
  #   return mask
  
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

    representations = torch.cat([rep1, rep2], dim=0)

    similarities = self.similarity(representations, representations)
    positive_similarities = similarities[self.batch_size].view(dbl_batch,1)
    negative_similarities = similarities[self.negative_representations_mask]
    negative_similarities = negative_similarities.view(dbl_batch, -1)

    logits = torch.cat([positive_similarities, negative_similarities], dim=1)
    logits /= self.temperature

    labels = torch.zeros(dbl_batch).to(self.device).long()

    loss = self.criterion(logits, labels)
    return loss / dbl_batch
