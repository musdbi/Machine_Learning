import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from models.classifier import Classifier
from models.logistic_regression import LogisticRegression
from models.linear_discriminant_analysis import LDA


def cross_validation(X, y, model: Classifier):
    """
    params:
      X (array): the data
      y(array): the labels
      model (Classifier): the model we want to test: "LR" or "LDA"

    returns:
      error_rate(float): the list of the percentages of error on X, y according to parameter
    """
    # Instanciation
    predictions_accuracy = np.array([])
    if isinstance(model, LogisticRegression):
        X = np.column_stack((X, np.ones(X.shape[0])))

    # Processing
    for i in range(len(y)):
        # Extraction of data at index i
        X_train = np.delete(X, i, axis=0)
        X_test = X[i].reshape(1, X[i].size)

        y_train = np.delete(y, i, axis=0)
        y_test = y[i]

        # Model training
        model.train(X_train, y_train)

        # Prediction for data i
        predictions_accuracy = np.append(
            predictions_accuracy, model.predict(X_test) == y_test
        )
    return np.mean(predictions_accuracy == False)


def display_boundary(X, y, models):
    """
    Graphic representation of the given model decision boundary
    """

    fig, axes = plt.subplots(1, len(models), figsize=(15, 5))

    # Initializing grid points
    x_min, x_max, y_min, y_max = (
        X[:, 0].min() - 1,
        X[:, 0].max() + 1,
        X[:, 1].min() - 1,
        X[:, 1].max() + 1,
    )
    x1, x2 = np.meshgrid(np.arange(x_min, x_max, 0.02), np.arange(y_min, y_max, 0.02))

    # Display for each model
    for i in range(len(models)):
        model = models[i]
        # Captions
        axes[i].set_xlabel("Composante 1")
        axes[i].set_ylabel("Composante 2")
        axes[i].set_title(f"{model} \nfrontière de décision")
        cm_bright = ListedColormap(["#FF0000", "#0000FF"])
        axes[i].grid(True)

        # Linear equation of the decision boundary: ax + by + c
        if isinstance(models[i], LogisticRegression):
            boundary_points = model.Bhat[0] * x1 + model.Bhat[1] * x2 + model.Bhat[2]
        else:
            boundary_points = model.w[0] * x1 + model.w[1] * x2 + model.b

        boundary_points[boundary_points > 0] = 1
        boundary_points[boundary_points <= 0] = -1

        if isinstance(models[i], LDA):
            boundary_points = -boundary_points

        # Background fill with class color
        axes[i].contourf(x1, x2, boundary_points, cmap=cm_bright, alpha=0.8)

        # Display data
        axes[i].scatter(X[:, 0], X[:, 1], c=y, cmap=cm_bright, edgecolors="k")

        # Display decision boundary
        axes[i].contour(x1, x2, boundary_points, colors="k", linewidths=0.3)

    plt.tight_layout()
    plt.show()
