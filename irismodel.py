import torch.nn as nn

class IrisModel(nn.Module):
    def __init__(self):
        super(IrisModel, self).__init__()
        self.layer0 = nn.Linear(4, 3)
        self.logsoftmax = nn.LogSoftmax()

    def forward(self, i):
        output = self.layer0(i)
        output = self.logsoftmax(output)
        return output

def train(X, y):
    pass