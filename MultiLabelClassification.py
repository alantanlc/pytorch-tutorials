import torch
import torch.nn as nn
import torch.optim as optim
import copy

model = nn.Linear(20, 5)    # predict logits for 5 classes
x = torch.randn(2, 20)
y = torch.tensor([ [1., 0., 1., 0., 0.], [1., 0., 1., 0., 0.] ])  # get classA and classC as active

criterion = nn.BCEWithLogitsLoss()
optimizer = optim.SGD(model.parameters(), lr=1e-1)

best_model_wts = copy.deepcopy(model.state_dict())
best_loss = 0.0

for epoch in range(20):
    optimizer.zero_grad()
    output = model(x)
    loss = criterion(output, y)
    loss.backward()
    optimizer.step()
    print('Loss: {:.3f}'.format(loss.item()))

    if loss.item() > best_loss:
        best_loss = loss.item()
        best_model_wts = copy.deepcopy(model.state_dict())

output = model(x)
print(output)

print(model.state_dict())