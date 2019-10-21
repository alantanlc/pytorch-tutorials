from __future__ import print_function
import torch

# Construct a 5x3 matrix, uninitialized
x = torch.empty(5, 3)
print(x)

# Construct a randomly initialized matrix
x = torch.rand(5, 3)
print(x)

# Construct a matrix filled zeros and of dtype long
x = torch.zeros(5, 3, dtype=torch.long)
print(x)

# Construct a tensor directly from data
x = torch.tensor([5.5, 3])
print(x)

# Create a tensor based on an existing tensor. These methods will reuse properties of the input tensor, e.g. dtype, unless new values are provided by user
x = x.new_ones(5, 3, dtype=torch.double)    # new_* methods take in sizes
print(x)

x = torch.rand_like(x, dtype=torch.float)   # override dtype!
print(x)    # result has the same size

# Get its size
print(x.size())

# Addition: syntax 1
y = torch.rand(5,3)
print(x+y)

# Addition: syntax 2
print(torch.add(x, y))

# Addition: providing an output tensor as argument
result = torch.empty(5, 3)
torch.add(x, y, out=result)
print(result)

# Addition: in-place
# adds x to y
y.add_(x)
print(y)

# NOTE: Any operation that mutates a tensor in-place is post fixed with an _. For example: x.copy_(x), x.t_(), will change x.

# You can use standard NumPy-like indexing with all bells and whistles!
print(x[:, 1])

# Resizing: If you want to resize/reshape tensor, you can use torch.view
x = torch.rand(4, 4)
y = x.view(16)
z = x.view(-1, 8)   # the size -1 is inferred from other dimensions
print(x.size(), y.size(), z.size())

if torch.cuda.is_available():
    device = torch.device("cuda")   # a CUDA device object
    y = torch.ones_like(x, device=device)   # directly create a tensor on GPU
    x = x.to(device)    # or just use string ``.to("cuda")``
    z = x + y
    print(z)
    print(z.to("cpu", torch.double))    # ``.to`` can also change dtype together!