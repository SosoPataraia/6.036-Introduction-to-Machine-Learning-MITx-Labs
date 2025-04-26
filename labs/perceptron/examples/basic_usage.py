"""Basic usage example of the Perceptron implementation."""
import numpy as np
import matplotlib.pyplot as plt
from perceptron.perceptron import Perceptron
def main():
    X = np.array([[1, 1], [2, 2], [3, 3], [10, 10], [11, 11], [12, 12]])
    y = np.array([0, 0, 0, 1, 1, 1])
    clf = Perceptron(learning_rate=0.1, max_iter=100, verbose=True)
    clf.fit(X, y)
    y_pred = clf.predict(X)
    print(f"\nPredictions: {y_pred}")
    print(f"Actual:      {y}")
    clf.plot_decision_boundary(X, y, title="Perceptron Decision Boundary - Basic Example")
if __name__ == "__main__":
    main()