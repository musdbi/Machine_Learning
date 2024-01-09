from models.classifier import Classifier
import numpy as np


class SVM(Classifier):
    def __init__(self, C=1.0, batch_size=100, eta=0.001, epochs=1000):
        # C = error term
        self.C = C
        self.batch_size = batch_size
        self.eta = eta
        self.epochs = epochs
        self.w = None
        self.b = None

    def __str__(self):
        return f"SVM, eta = {self.eta}"

    # Hinge Loss Function / Calculation
    def hingeloss(self, w, b, x, y):
        # Regularizer term
        reg = 0.5 * (w * w)

        for i in range(x.shape[0]):
            # Optimization term
            opt_term = y[i] * ((np.dot(w, x[i])) + b)

            # calculating loss
            loss = reg + self.C * max(0, 1 - opt_term)
        return loss[0][0]

    def train(self, X, y):
        # Adapting labels for svm
        y = np.where(y == 0, -1, 1)

        # The number of features in X
        n_features = X.shape[1]

        # The number of Samples in X
        n_samples = X.shape[0]

        c = self.C

        # Creating ids from 0 to n_samples - 1
        ids = np.arange(n_samples)

        # Shuffling the samples randomly
        np.random.shuffle(ids)

        # creating an array of zeros
        w = np.zeros((1, n_features))
        b = 0
        loss = []

        # Gradient Descent logic
        for i in range(self.epochs):
            # Calculating the Hinge Loss
            l = self.hingeloss(w, b, X, y)

            # Appending all loss
            loss.append(l)

            # Starting from 0 to the number of samples with self.batch_size as interval
            for batch_initial in range(0, n_samples, self.batch_size):
                gradw = 0
                gradb = 0

                for j in range(batch_initial, batch_initial + self.batch_size):
                    if j < n_samples:
                        x = ids[j]
                        # Calulating the condition for classifying
                        ti = y[x] * (np.dot(w, X[x].T) + b)

                        if ti > 1:
                            gradw += 0
                            gradb += 0
                        else:
                            # Calculating the gradients

                            # w.r.t w
                            gradw += c * y[x] * X[x]
                            # w.r.t b
                            gradb += c * y[x]

                # Updating weights and bias
                w = w - self.eta * w + self.eta * gradw
                b = b + self.eta * gradb

        self.w = w.flatten()
        self.b = b

        return loss

    def predict(self, X):
        predictions = np.dot(X, self.w) + self.b
        return np.where(predictions > 0, 1, 0)

    def error_rate(self, X, y):
        return np.mean(self.predict(X) != y)
