# 6.036 Introduction to Machine Learning MITx Labs

This repository contains implementations and labs from the MITx 6.036 Introduction to Machine Learning course.

## Structure
- `labs/perceptron/`: Perceptron implementation from scratch.
  - `perceptron/perceptron.py`: Perceptron class.
  - `examples/`: Demos (e.g., Iris dataset).
  - `tests/`: Unit tests.
- `labs/feature_engineering/`: Feature engineering lab for car data.
  - `data/auto-mpg.tsv`: Dataset.
  - `notebooks/auto_mpg_classification.ipynb`: Lab implementation.
  - `src/feature_engineering.py`: Feature processing code.

## Setup
```bash
python -m venv myenv
myenv\Scripts\activate
pip install -r labs\perceptron\requirements.txt
pip install -r labs\feature_engineering\requirements.txt

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


