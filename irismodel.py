import torch.nn as nn
import torch.optim as optim
import torch

class IrisModel(nn.Module):
    def __init__(self):
        super(IrisModel, self).__init__()
        self.layer0 = nn.Linear(4, 3)
        self.logsoftmax = nn.LogSoftmax()

    def forward(self, i):
        output = self.layer0(i)
        output = self.logsoftmax(output)
        return output

def train(X, y, epochs=10):
    model = IrisModel()
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters())
    
    for i in range(epochs):
        for x, y in list(zip([torch.FloatTensor(x.to_numpy()) for x in list(zip(*X.iterrows()))[1]], 
                             [torch.LongTensor(z) for z in list(y)])):
            optimizer.zero_grad()
            output = model(x)
            loss = loss_fn(output, y)
            loss.backward()
            optimizer.step()

    return model