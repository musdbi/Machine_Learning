from models.classifier import Classifier
import numpy as np


class LDA(Classifier):
    """
    Linear Discriminant Analysis
    """

    def __init__(self):
        self.w = None
        self.b = None

    def __str__(self):
        return "LDA"

    def train(self, X, y):
        n0 = np.count_nonzero(y == 0)
        n1 = np.count_nonzero(y == 1)

        mu_0 = np.mean(X[y == 0], axis=0)
        mu_1 = np.mean(X[y == 1], axis=0)

        pi_0 = np.mean(y == 0)
        pi_1 = np.mean(y == 1)

        centered_0 = X[y == 0] - mu_0
        centered_1 = X[y == 1] - mu_1

        Xtemp0 = np.cov(centered_0, rowvar=False)  # is the covariance matrix
        Xtemp1 = np.cov(centered_1, rowvar=False)  # same

        Sigma = (n0 * Xtemp0 + n1 * Xtemp1) / (n0 + n1)
        iSigma = np.linalg.pinv(Sigma)

        self.w = np.dot(iSigma, (mu_0 - mu_1))
        self.b = -(1 / 2) * np.dot(
            (mu_0 - mu_1), np.dot(iSigma, (mu_0 + mu_1))
        ) + np.log(pi_0 / pi_1)

    def predict(self, X):
        """
        This function gives the class prediction for the given vector for the linear discriminant analysis
        params:
            x(array): the vector we want to predict the label
            beta (array): the parameters vector

          returns:
            class (int): the class of x
        """
        predictions = np.dot(X, self.w) + self.b

        # return 1 where predictions are negative, 0 if positive
        return np.where(predictions < 0, 1, 0)

    def error_rate(self, X, y):
        return np.mean(self.predict(X) != y)
