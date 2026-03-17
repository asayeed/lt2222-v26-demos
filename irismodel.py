import torch.nn as nn
import torch.optim as optim
import torch
import numpy as np

class IrisModel(nn.Module):
    def __init__(self):
        super(IrisModel, self).__init__()
        self.layer0 = nn.Linear(4, 3)
        self.logsoftmax = nn.LogSoftmax(dim=0)

    def forward(self, i):
        output = self.layer0(i)
        output = self.logsoftmax(output)
        return output

def train(X, Y, epochs=10):
    model = IrisModel()
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters())
    X_instances = [torch.FloatTensor(x.to_numpy()) for x in list(zip(*X.iterrows()))[1]]
    Y_instances = [torch.LongTensor(np.array([z])) for z in list(Y)]
    #instances = list(zip([torch.FloatTensor(x.to_numpy()) for x in list(zip(*X.iterrows()))[1], 
                             #[torch.Tensor(z) for z in list(Y)]))
    print(Y)
    print(Y_instances)
    for i in range(epochs):        
        for x, y in zip(X_instances, Y_instances):
            optimizer.zero_grad()
            output = model(x).unsqueeze(0)
            print("output {} y {}".format(output, y))
            loss = loss_fn(output, y)
            loss.backward()
            optimizer.step()

    return model

def test(model, X):
    model.eval()
    with torch.no_grad():
        X_instances = [torch.FloatTensor(x.to_numpy()) for x in list(zip(*X.iterrows()))[1]]
        accumulate = []
        for x in X_instances:
            output = model(x)
            accumulate.append(output)
    return accumulate
        
