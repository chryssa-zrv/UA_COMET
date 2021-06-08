from typing import Dict, List, Union

import torch
import torch.nn as nn
from tqdm import tqdm
from operator import itemgetter
import numpy as np

from comet.metrics import RegressionReport
from comet.models.model_base import ModelBase
from comet.models.utils import average_pooling, max_pooling, move_to_cpu, move_to_cuda

import statistics


class VarianceLoss(nn.Module):
    def __init__(self):
        super(VarianceLoss,self).__init__()

    def forward(self,x,y):
        sigma = x[1]
        mu = x[0]
        log1 = 0.5 * torch.neg(torch.log(sigma)).exp() 
        mse = (y - mu)**2
        log2 = 0.5 * torch.log(sigma)
        totloss = torch.sum(log1*mse+log2)
        return totloss