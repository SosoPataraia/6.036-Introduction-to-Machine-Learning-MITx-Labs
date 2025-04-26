import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from labs.perceptron.perceptron.perceptron import Perceptron

def load_auto_mpg(file_path):
    df = pd.read_csv(file_path, sep='\t')
    return df

def preprocess_features(df, feature_configs):
    X = []
    for feature, method in feature_configs.items():
        if method == 'drop':
            continue
        elif method == 'raw':
            X.append(df[feature].values)
        elif method == 'standard':
            scaler = StandardScaler()
            X.append(scaler.fit_transform(df[feature].values.reshape(-1, 1)))
        elif method == 'one-hot':
            dummies = pd.get_dummies(df[feature], prefix=feature)
            X.extend([dummies[col].values for col in dummies.columns])
    return np.column_stack(X)

def preprocess_car_name(df):
    # Simple transformation: length of car name
    return df['car name'].str.len().values.reshape(-1, 1)

def evaluate_representation(X, y, method='test_split'):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
    model = Perceptron()
    model.train(X_train, y_train)
    if method == 'full_data':
        y_pred = model.predict(X)
        return np.mean(y_pred == y)
    elif method == 'train_test_sum':
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)
        return np.mean(y_train_pred == y_train) + np.mean(y_test_pred == y_test)
    elif method == 'test_split':
        y_test_pred = model.predict(X_test)
        return np.mean(y_test_pred == y_test)
    elif method == 'cross_val':
        scores = cross_val_score(model, X, y, cv=10, scoring='accuracy')
        return np.mean(scores)