

import torch
import torch.nn as nn

import numpy as np




class VarianceLoss(nn.Module):
    def __init__(self):
        super(VarianceLoss,self).__init__()

    def forward(self,x,s,y):
        sigma = s**2
        mu = x
        #print('-------------losses-----------------------')
        #print(sigma)
        #print(mu)
        #print(y)
        log1 = 0.5 * torch.neg(torch.log(sigma)).exp() 
        #print(log1)
        mse = (y - mu)**2
        #print(mse)
        log2 = 0.5 * torch.log(sigma)
        #print(log2)
        totloss = torch.sum(log1*mse+log2)
        #print(totloss)
        #exit(0)
        return totloss