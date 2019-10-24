import torch

# torch.Tensor is the central class of the package. If you set its attribute .requires_grad as True, it starts to track all operations on it.
# When you finish your computation you can call .backward() and have all gradients computed automatically.
# The gradient for this tensor will be accumulated into .grad attribute

# To stop a tensor from tracking history, you can call .detach() to detach it from the computation history, and to prevent future computation from being tracked.

# To prevent tracking history (and using memory), you can also wrap the code block in 'with torch,no_grad():'. This can be particularly helpful when evaluating a model because the model may have trainable parameters with requires_grad=True, but for which we don't need the gradients.

# There's on more class whic is vey important for autograd implementation - a Function.

# Tensor and Function are interconnected and build up and acyclic graph, that encodes a complete history of computation.
# Each tensor has a .grad_fn attribute that references a Function that has created the Tensor (except for Tensors created by the user - their grad_fn is None).

# If you want to compute the derivatives, you can call .backward() on a Tensor. If Tensor is a scalar (i.e. it holds a one element data), you don't need to specify any arguments to backward(), however if it has more elements, you need to specify a gradient argument that is a tensor of matching shape.

# Create a tensor and set requires_grad=True to track computation with it
x = torch.ones(2, 2, requires_grad=True)
print(x)