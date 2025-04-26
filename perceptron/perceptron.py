"""
A Python implementation of the Perceptron algorithm from scratch using NumPy.

This implementation includes:
- Detailed documentation for professional presentation
- Type hints for better code clarity
- Additional utility methods
- Comprehensive error handling
- Visualization capabilities
- Unit tests for verification
- Example usage with real dataset

Author: [Your Name]
GitHub: [Your GitHub URL]
"""

import numpy as np
from typing import Union, Optional
import matplotlib.pyplot as plt
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.metrics import accuracy_score


class Perceptron(BaseEstimator, ClassifierMixin):
    """A perceptron classifier implementation from scratch.
    
    The perceptron is a fundamental algorithm in machine learning that serves as 
    the building block for neural networks. This implementation follows the 
    original Rosenblatt perceptron algorithm for binary classification.
    
    Parameters
    ----------
    learning_rate : float, default=0.01
        The step size at each iteration (between 0.0 and 1.0)
    max_iter : int, default=1000
        Maximum number of iterations over the training dataset
    random_state : int, optional
        Seed for random weight initialization
    verbose : bool, default=False
        If True, prints progress during training
        
    Attributes
    ----------
    weights_ : ndarray of shape (n_features,)
        Weights assigned to the features
    bias_ : float
        Bias term
    n_iter_ : int
        Actual number of iterations completed
    errors_ : list
        Number of misclassifications in each epoch
        
    Examples
    --------
    >>> from sklearn.datasets import load_iris
    >>> from sklearn.model_selection import train_test_split
    >>> X, y = load_iris(return_X_y=True)
    >>> X = X[y != 2]  # Binary classification
    >>> y = y[y != 2]
    >>> X_train, X_test, y_train, y_test = train_test_split(X, y)
    >>> clf = Perceptron()
    >>> clf.fit(X_train, y_train)
    >>> accuracy = clf.score(X_test, y_test)
    """
    
    def __init__(self, learning_rate: float = 0.01, max_iter: int = 1000, 
                 random_state: Optional[int] = None, verbose: bool = False):
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.random_state = random_state
        self.verbose = verbose
        
    def _unit_step_func(self, x: np.ndarray) -> np.ndarray:
        """Heaviside step function activation.
        
        Parameters
        ----------
        x : ndarray
            Input array
            
        Returns
        -------
        ndarray
            1 if x >= 0, else 0
        """
        return np.where(x >= 0, 1, 0)
    
    def _initialize_weights(self, n_features: int) -> None:
        """Initialize weights and bias with small random values."""
        rgen = np.random.RandomState(self.random_state)
        self.weights_ = rgen.normal(loc=0.0, scale=0.01, size=n_features)
        self.bias_ = 0.0
        
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'Perceptron':
        """Fit the perceptron to the training data.
        
        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Training data
        y : ndarray of shape (n_samples,)
            Target values (binary: -1 or 1, or 0 or 1)
            
        Returns
        -------
        self
            Fitted perceptron
        """
        # Input validation
        X, y = check_X_y(X, y)
        y = self._validate_targets(y)
        
        n_samples, n_features = X.shape
        self._initialize_weights(n_features)
        self.errors_ = []
        
        # Ensure targets are {-1, 1}
        y_ = np.where(y == 0, -1, 1)
        
        for epoch in range(self.max_iter):
            errors = 0
            for idx, x_i in enumerate(X):
                linear_output = np.dot(x_i, self.weights_) + self.bias_
                y_pred = self._unit_step_func(linear_output)
                y_pred = np.where(y_pred == 0, -1, 1)  # Map to {-1, 1}
                
                update = self.learning_rate * (y_[idx] - y_pred)
                self.weights_ += update * x_i
                self.bias_ += update
                errors += int(update != 0.0)
                
            self.errors_.append(errors)
            if self.verbose and epoch % 50 == 0:
                print(f"Epoch {epoch + 1}/{self.max_iter}, Errors: {errors}")
            
            # Early stopping if converged
            if errors == 0:
                if self.verbose:
                    print(f"Converged at epoch {epoch + 1}")
                break
                
        self.n_iter_ = epoch + 1
        self.classes_ = np.array([0, 1])  # For sklearn compatibility
        return self
    
    def _validate_targets(self, y: np.ndarray) -> np.ndarray:
        """Ensure targets are binary (0/1 or -1/1)."""
        unique_y = np.unique(y)
        if len(unique_y) != 2:
            raise ValueError("Perceptron is for binary classification only")
            
        # Convert to 0/1 if needed
        if not (np.all(unique_y == [0, 1]) or np.all(unique_y == [-1, 1])):
            return np.where(y == unique_y[0], 0, 1)
        return y
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels for samples in X.
        
        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Test samples
            
        Returns
        -------
        ndarray of shape (n_samples,)
            Predicted class labels (0 or 1)
        """
        check_is_fitted(self)
        X = check_array(X)
        
        linear_output = np.dot(X, self.weights_) + self.bias_
        y_pred = self._unit_step_func(linear_output)
        return y_pred
    
    def decision_function(self, X: np.ndarray) -> np.ndarray:
        """Compute the decision function for each sample.
        
        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Test samples
            
        Returns
        -------
        ndarray of shape (n_samples,)
            Confidence scores
        """
        check_is_fitted(self)
        X = check_array(X)
        return np.dot(X, self.weights_) + self.bias_
    
    def plot_decision_boundary(self, X: np.ndarray, y: np.ndarray, 
                              title: str = "Perceptron Decision Boundary"):
        """Visualize the decision boundary (for 2D data only).
        
        Parameters
        ----------
        X : ndarray of shape (n_samples, 2)
            Input features (must be 2D)
        y : ndarray of shape (n_samples,)
            Target values
        title : str, optional
            Plot title
        """
        if X.shape[1] != 2:
            raise ValueError("Plotting only supported for 2D data")
            
        plt.figure(figsize=(8, 6))
        
        # Plot data points
        plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Paired, edgecolors='k')
        
        # Plot decision boundary
        x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                             np.arange(y_min, y_max, 0.02))
        
        Z = self.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)
        plt.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.Paired)
        
        plt.xlabel('Feature 1')
        plt.ylabel('Feature 2')
        plt.title(title)
        plt.show()


def test_perceptron():
    """Run unit tests on the Perceptron implementation."""
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split
    
    print("Running Perceptron unit tests...")
    
    # Test 1: Simple linearly separable data
    X = np.array([[1, 1], [2, 2], [3, 3], [10, 10], [11, 11], [12, 12]])
    y = np.array([0, 0, 0, 1, 1, 1])
    
    clf = Perceptron(max_iter=100)
    clf.fit(X, y)
    preds = clf.predict(X)
    assert accuracy_score(y, preds) == 1.0, "Failed on simple test case"
    
    # Test 2: Random data
    X, y = make_classification(n_samples=100, n_features=2, n_redundant=0, 
                              n_informative=2, random_state=42)
    y = np.where(y == 0, -1, 1)  # Test with {-1, 1} labels
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    clf = Perceptron(max_iter=1000, verbose=False)
    clf.fit(X_train, y_train)
    accuracy = clf.score(X_test, y_test)
    assert accuracy > 0.8, f"Accuracy too low: {accuracy}"
    
    print("All tests passed!")


if __name__ == "__main__":
    # Example usage with visualization
    from sklearn.datasets import make_blobs
    
    # Create a simple dataset
    X, y = make_blobs(n_samples=100, centers=2, n_features=2, 
                      cluster_std=1.0, random_state=42)
    
    # Initialize and train the Perceptron
    clf = Perceptron(learning_rate=0.1, max_iter=100, verbose=True)
    clf.fit(X, y)
    
    # Make predictions and calculate accuracy
    y_pred = clf.predict(X)
    accuracy = accuracy_score(y, y_pred)
    print(f"\nTraining accuracy: {accuracy:.2f}")
    
    # Plot decision boundary
    clf.plot_decision_boundary(X, y, 
                              title=f"Perceptron Decision Boundary (Accuracy: {accuracy:.2f})")
    
    # Run unit tests
    test_perceptron()