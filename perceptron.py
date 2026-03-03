import numpy as np

def sigmoid(x):
    return 1/(1 + np.exp(-x))

class Perceptron:
    def __init__(self, features=10):
        self.weights = np.random.normal(0, 1, features)

    def result(self, x):
        result = np.dot(self.weights, x)
        return result
        
    def update(self, x, y):
        loss = y - self.predict_sigmoid(x)
        self.weights = self.weights + loss
        return loss

    def predict_sigmoid(self, x):
        return sigmoid(self.result(x))
    
    def predict(self, x):
        return 1 if sigmoid(self.result(x)) > 0.5 else 0

    def test(self, X):
        Xnp = X.to_numpy()
        results = []
        for i in range(len(Xnp)):
            results.append(self.predict(Xnp[i]))
        return results
    
    def train(self, X, y, epochs=10):
        Xnp = X.to_numpy()
        ynp = y.to_numpy()
        total_loss = []
        for e in range(epochs):
            epoch_loss = []
            for i in range(len(Xnp)):
                epoch_loss.append(self.update(Xnp[i], ynp[i]))
            total_loss.append(np.mean(epoch_loss))
        return total_loss