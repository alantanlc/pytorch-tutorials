# Reference: https://pytorch.org/tutorials/beginner/blitz/data_parallel_tutorial.html#dummy-dataset

# Import PyTorch modules and define parameters
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# Parameters and DataLoaders
input_size = 5
output_size = 2
batch_size = 30
data_size = 100

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
print(torch.cuda.is_available())

# Make a dummy (random) dataset. You just need to implement the getitem
class RandomDataset(Dataset):

    def __init__(self, size, length):
        self.len = length
        self.data = torch.randn(length, size)

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return self.len

rand_loader = DataLoader(dataset=RandomDataset(input_size, data_size), batch_size=batch_size, shuffle=True)

# Implement a model that takes an input, performs a linear operation, and gives an output.
class Model(nn.Module):

    # Our model

    def __init__(self, input_size, output_size):
        super(Model, self).__init__()
        self.fc = nn.Linear(input_size, output_size)

    def forward(self, input):
        output = self.fc(input)
        print("\tIn Model: input size", input.size(), "output size", output.size())

        return output

# Create Model and DataParallel
# First, we make a model instance and check if we have multiple GPUs.
# If we have multiple GPUs, we can wrap our model using nn.DataParallel.
# Then we can put our model on GPUs by mode.to(device)
model = Model(input_size, output_size)
if torch.cuda.device_count() > 1:
    print("Let's use", torch.cuda.device_count(), "GPUs!")
    # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
    model = nn.DataParallel(model)
model.to(device)

# Run the Model: Now we can see the sizes of input and output tensors.
for data in rand_loader:
    input = data.to(device)
    output = model(input)
    print("Outside: input size", input.size(), "output_size", output.size())

# Results
# If you have no GPU or one GPU, when we batch 30 inputs and 30 outputs,
# the model gets 30 and outputs 30 as expected.
# But if you have multiple GPUs, then you can get results like this (refer to bottom of page of the following reference link).
