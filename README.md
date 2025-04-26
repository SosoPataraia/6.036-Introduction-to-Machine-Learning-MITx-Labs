# Perceptron Implementation from Scratch

[![Python Version](https://img.shields.io/badge/python-3.7%2B-blue)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A complete implementation of the Perceptron algorithm from scratch using NumPy, following software engineering best practices and scikit-learn's estimator interface.

## Features

- Pure Python/NumPy implementation
- Follows scikit-learn's BaseEstimator pattern
- Comprehensive documentation and type hints
- Visualization of decision boundaries
- Unit tested with pytest
- Handles both {0,1} and {-1,1} target encodings
- Early stopping when converged
- Progress reporting during training

## Installation

```bash
git clone https://github.com/SosoPataraia/perceptron-from-scratch.git
cd perceptron-from-scratch
pip install -r requirements.txt