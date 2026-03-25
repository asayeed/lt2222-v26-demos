import torch.nn as nn
import torch
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pickle
import tqdm

class ReviewDataset(Dataset):
    def __init__(self, filename):
        super(ReviewDataset, self).__init__()
        with open(filename, "rb") as inputfile:
            self.review_df = pickle.load(inputfile)
        self.review_df['vectensor'] = self.review_df['vectors'].apply(lambda x: torch.FloatTensor(x))

    def __len__(self):
        return len(self.review_df)

    def __getitem__(self, i):
        X = self.review_df.iloc[i]['vectensor']
        y = self.review_df.iloc[i]['sentiment']
        return (X, y)

class ReviewModel(nn.Module):
    def __init__(self, inputsize=100, hiddensize=50):
        super(ReviewModel, self).__init__()
        self.linear0 = nn.Linear(inputsize, hiddensize)
        self.tanh = nn.Tanh()
        self.linear1 = nn.Linear(hiddensize, hiddensize)
        self.relu = nn.Tanh()
        self.linear2 = nn.Linear(hiddensize, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, data):
        output = self.linear0(data)
        output = self.tanh(output)
        output = self.linear1(output)
        output = self.relu(output)
        output = self.linear2(output)
        return self.sigmoid(output)

def train(inputfile, inputsize=100, hiddensize=50, batch_size=5, epochs=10):
    review_dataset = ReviewDataset(inputfile)
    review_loader = DataLoader(review_dataset, shuffle=True, batch_size=batch_size)
    model = ReviewModel(inputsize, hiddensize)
    loss_fn = nn.BCELoss()
    optimizer = optim.Adam(model.parameters())

    for i in range(epochs):
        lossall = 0
        for j, batch in enumerate(tqdm.tqdm(review_loader)):
            optimizer.zero_grad()
            X, y = batch
            output = torch.reshape(model(X), (5,))
            y = y.float()
            #print("output {} y {} y type {}".format(output, y, type(y)))
            loss = loss_fn(output, y)
            loss.backward()
            optimizer.step()
            lossall += loss
        print(loss)

    return model

    
