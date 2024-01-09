from abc import ABC, abstractmethod


class Classifier:
    @abstractmethod
    def train(self, X, y):
        pass

    @abstractmethod
    def predict(self, X):
        pass

    @abstractmethod
    def error_rate(self, X, y):
        pass
