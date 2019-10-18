import sys, os
import torch
import torch.nn as nn
import torch.nn.functional as F

# Set the device variable to be either cuda or cpu
device = torch.device("cuda")
print(device)

x = torch.rand(3,3)
print(x)

x = x.to(device)
print(x)
