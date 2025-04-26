import numpy as np
import pytest
from perceptron.perceptron import Perceptron
from sklearn.datasets import make_classification
from sklearn.exceptions import NotFittedError

def test_perceptron_initialization():
    """Test that the perceptron initializes correctly."""
    clf = Perceptron(learning_rate=0.01, max_iter=100, random_state=42)
    assert clf.learning_rate == 0.01
    assert clf.max_iter == 100
    assert clf.random_state == 42

def test_fit_predict():
    """Test basic fit and predict functionality."""
    X = np.array([[1, 1], [2, 2], [3, 3], [10, 10], [11, 11], [12, 12]])
    y = np.array([0, 0, 0, 1, 1, 1])
    
    clf = Perceptron(max_iter=100)
    clf.fit(X, y)
    preds = clf.predict(X)
    
    assert np.array_equal(preds, y)
    assert clf.n_iter_ < clf.max_iter  # Should converge early

def test_not_fitted():
    """Test that predict fails on unfitted model."""
    clf = Perceptron()
    X = np.array([[1, 1]])
    with pytest.raises(NotFittedError):
        clf.predict(X)

def test_different_target_encodings():
    """Test that the perceptron works with both {0,1} and {-1,1} targets."""
    X = np.array([[1, 1], [1, -1], [-1, 1], [-1, -1]])
    y1 = np.array([0, 1, 1, 0])  # XOR problem (won't converge)
    y2 = np.array([-1, 1, 1, -1])
    
    clf1 = Perceptron(max_iter=10)
    clf1.fit(X, y1)
    
    clf2 = Perceptron(max_iter=10)
    clf2.fit(X, y2)
    
    assert np.array_equal(clf1.predict(X), clf2.predict(X))

def test_early_stopping():
    """Test that the perceptron stops when converged."""
    X = np.array([[1], [2], [3], [4]])
    y = np.array([0, 0, 1, 1])
    
    clf = Perceptron(max_iter=1000)
    clf.fit(X, y)
    
    assert clf.n_iter_ < clf.max_iter
    assert clf.errors_[-1] == 0

def test_decision_function():
    """Test the decision_function method."""
    X = np.array([[1, 1], [2, 2]])
    y = np.array([0, 1])
    
    clf = Perceptron()
    clf.fit(X, y)
    decisions = clf.decision_function(X)
    
    assert decisions.shape == (2,)
    assert np.all((decisions >= 0) == y)

@pytest.mark.parametrize("n_features", [1, 2, 5])
def test_dimensionality(n_features):
    """Test that the perceptron works with different feature dimensions."""
    X, y = make_classification(n_samples=100, n_features=n_features, 
                              n_redundant=0, n_informative=n_features,
                              random_state=42)
    clf = Perceptron(max_iter=100)
    clf.fit(X, y)
    preds = clf.predict(X)
    assert preds.shape == (100,)