from typing_extensions import override
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    def __init__(self):
        super(Model,self).__init__()
    def forward(self, input):
        output=input+1
        return output