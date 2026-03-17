import torch.nn as nn
import torch
from torch.utils.data import Dataset, DataLoader
import pickle

class ReviewDataset(Dataset):
    def __init__(self, filename):
        super(ReviewDataset, self).__init__()
        with open(filename, "rb") as inputfile:
            self.review_df = pickle.load(inputfile)

    def __len__(self):
        return len(self.review_df)

    def __getitem__(self, i):
        return (review_df.iloc[i]['vector'], review_df.iloc[i]['sentiment'])

class ReviewModel(nn.Module):
    def __init__(self, inputsize=100, hiddensize=50):
        super(ReviewModel, self).__init__()
        self.linear0 = nn.Linear(inputsize, hiddensize)
        self.tanh = nn.Tanh()
        self.linear1 = nn.Linear(hiddensize, hiddensize)
        self.relu = nn.Tanh()
        self.linear2 = nn.Linear(hiddensize, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(data):
        output = self.linear0(data)
        output = self.tanh(output)
        output = self.linear1(output)
        output = self.relu(output)
        output = self.linear2(output)
        return self.sigmoud(output)

def train(inputfile, inputsize=100, hiddensize=50, epochs=10):
    review_dataset = ReviewDataset(inputfile)
    review_loader
    model = ReviewModel(inputsize, hiddensize)
    loss_fn = nn.BCELoss()
    optimizer = optim.Adam(model.parameters())
    
