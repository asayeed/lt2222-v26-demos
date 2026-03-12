import os
import sys
from irismodel import IrisModel, train
import pickle
import pandas as pd
from argparse import ArgumentParser
from sklearn.preprocessing import MinMaxScaler
from random import choices, sample

if __name__ == "__main__":
    argparser = ArgumentParser(
        prog="Iris classifier",
        description="Classifies the Iris data",
        epilog="Not about eye parts.")

    argparser.add_argument("filename", help="The filename of the Iris data pickle.")
    argparser.add_argument("-t", "--trainingsize", help="The proportion of training data in [0, 1.0].")
    args = argparser.parse_args()

    with open(args.filename, "rb") as irisfile:
        iris = pickle.load(irisfile)

    scaler = MinMaxScaler((-1, 1))
    X = iris.data.features 
    y = iris.data.targets
    classes = list(set(iris.data.targets['class']))
    y['class'] = iris.data.targets.apply(lambda x: classes.index(x['class']), axis=1)
    
    scaler.fit(X)
    X_scaled = scaler.transform(X)
    X_scaled = pd.DataFrame(X_scaled, index=X.index, columns=X.columns)

    train_indices = sample(range(len(X_scaled)), k=int(float(args.trainingsize) * len(X_scaled)))

    X_train = X_scaled.iloc[sorted(train_indices)]
    X_test = X_scaled.iloc[~X_scaled.index.isin(X_train.index)]

    y_train = y['class'][X_train.index]
    y_test = y['class'][X_test.index]

    
    
    print("training size: {} testing size: {}".format(len(X_train), len(X_test)))

    model = train(X_train, y_train)
    