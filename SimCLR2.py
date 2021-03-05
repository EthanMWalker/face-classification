#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  5 10:33:47 2021

@author: becca
"""
import NTXentLoss
from resnet.Models import ResNetSimCLR
import torch.nn.functional as F
import torch

class SIMCLR2():
    
    def __init__(self, dataset, batch_size,temperature):
        self.data = dataset
        self.temperature
        self.batch_size = batch_size
        self.n_epochs = 90
        self.log_steps = 100
        
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.nt_xent = NTXentLoss(temperature, batch_size, self.device)
            
        print("Running on:", self.device)
        
        
    def train(self):
        model = ResNetSimCLR(**self.config["model"]).to(self.device)
          

        optimizer = torch.optim.Adam(model.parameters(), 3e-4, weight_decay=eval(self.config['weight_decay']))

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=len(train_loader), eta_min=0,
                                                               last_epoch=-1)
        
        for epoch in (range(self.n_epochs)):
            for (xis, xjs) in data:
                optimizer.zero_grad()
                
                xis = xis.to(self.device)
                xjs = xjs.to(self.device)
                
                # Get representations and projecttions
                his, zis = model(xis)
                hjs, zjs= model(xjs)
                
                # normalize
                zis = F.normalize(zis, dim=1)
                zjs = F.normalize(zjs, dim=1)
                
                loss = self.nt_xent(zis, zjs)
                
                # record loss
                
                loss.backward()
                optimizer.step()
            if epoch >= 10:
                scheduler.step()      