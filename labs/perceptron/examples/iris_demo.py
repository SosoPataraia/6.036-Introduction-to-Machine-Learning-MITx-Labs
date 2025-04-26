# %% [markdown]
# # Perceptron on Iris Dataset
# This notebook demonstrates using our Perceptron implementation on the classic Iris dataset.

# %%
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from perceptron.perceptron import Perceptron

# %%
# Load Iris dataset
iris = datasets.load_iris()
X = iris.data[:, :2]  # Use only first two features for visualization
y = iris.target

# Convert to binary classification (setosa vs non-setosa)
X = X[y != 2]
y = y[y != 2]

# Split into train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# %%
# Train Perceptron
clf = Perceptron(learning_rate=0.1, max_iter=100, verbose=True)
clf.fit(X_train, y_train)

# %%
# Evaluate
train_acc = clf.score(X_train, y_train)
test_acc = clf.score(X_test, y_test)
print(f"Training accuracy: {train_acc:.2f}")
print(f"Test accuracy: {test_acc:.2f}")

# %%
# Visualize decision boundary
plt.figure(figsize=(10, 6))
clf.plot_decision_boundary(X_train, y_train, title="Perceptron on Iris Dataset")