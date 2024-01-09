from models.classifier import Classifier
import numpy as np


class LogisticRegression(Classifier):
    def __init__(self, eta, stuck_criteria, itermax=10000):
        """
        eta (float): the step
        stuck_criteria (float): the distance between two B is too small
                                and we consider got close enough
        itermax (int): the maximum number of iterations
        Bhat (np.array): the parameter of the model we want to calculate
        """
        self.eta = eta
        self.stuck_criteria = stuck_criteria
        self.itermax = itermax
        self.Bhat = None

    def __str__(self):
        return f"Régression Logistique, learning rate = {self.eta}"

    def train(self, X, y):
        """
        This funtion execute the gradient descent algorithm to maximize a function
        params:
          X (np.array): the data
          y (np.array): the labels
        returns:
          Bi: the vector of estimated parameters
          loss: the list of losses at each iteration
        """
        # Useful variables
        i = 1
        notconv = True
        err_correct = 1.0e-15

        # initialization
        Bi = np.random.normal(size=(X[0].size, 1)).flatten()
        loss = []  # list of costs at each iteration
        lold = np.NINF  # numpy constant for negative infinite

        # Gradient descent algorithm
        while (i <= self.itermax) and (notconv):
            Bold = Bi
            # Gradient calculus
            grad = (y - self.sigmoid(X, Bold)) @ X

            # Update of current estimation
            Bi = Bold + self.eta * grad

            # Calculus of current loss
            li = np.sum(
                (y * np.log(self.sigmoid(X, Bi) + err_correct))
                + (1 - y) * np.log(1 - self.sigmoid(X, Bi) + err_correct)
            )

            loss.append(li)

            # Convergence criteria 1
            if np.linalg.norm(Bold - Bi) < self.stuck_criteria:
                notconv = False
                # print("Distance Bi / Bold close enough")

            # Convergence criteria 2
            if li < lold:
                notconv = False
                # print("Cost rise")

            i += 1
            lold = li

        self.Bhat = Bi
        return loss

    def sigmoid(self, X, b):
        return 1 / (1 + np.exp(-np.dot(X, b)))

    def predict(self, X):
        """
        This function gives the class prediction for the given vector for the lienar regression
        params:
          X(array): the vector we want to predict the label
        returns:
           class (int): the class of X
        """
        predictions = self.sigmoid(X, self.Bhat)
        return np.where(predictions > 1 / 2, 1, 0)

    def error_rate(self, X, y):
        """
        This function calculate the error rate for given parameter
        params:
          X (array): the data
          y(array): the labels
          beta (array): the parameters vector
        returns:
          error_rate (float): the percentage of error on X, y according to beta
        """
        return np.mean(self.predict(X) != y)
