import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F
import torch.nn as nn
import matplotlib.pyplot as plt

# 通过 functional.conv2d 实现卷积
input = torch.tensor([[1, 2, 0, 3, 1],
                      [0, 1, 2, 3, 1],
                      [1, 2, 1, 0, 0],
                      [5, 2, 3, 1, 1],
                      [2, 1, 0, 1, 1]], dtype=torch.float32)

input = torch.reshape(input, (1, 1, 5, 5))  # (batch_size, channels, height, width)


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3,padding=0)
        self.maxpool=nn.MaxPool2d(kernel_size=2,ceil_mode=True)#ceil_mode=True不舍弃边界
        self.relu=nn.ReLU()
        self.linear=nn.Linear(in_features=2,out_features=1)
    def forward(self,x):
        x=self.conv(x) 
        x=self.maxpool(x)
        x=self.linear(x)
        x=self.relu(x)
        return x

net=Net()
output=net(input)
print(output.shape)
