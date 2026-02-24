import numpy as np

class Perceptron:
    def __init__(self, features=10):
        self.weights = np.random.normal(0, 1, features)

    def result(self, x):
        return np.dot(self.weights, x)
        
    def train(self, x, y):
        self.weights = self.weights + (y - sum(self.result(x)))

    def predict(self, x):
        return 1 if self.result(x) > 0.5 else 0