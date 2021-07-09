import numpy as np
import pandas as pd
from collections import OrderedDict

PATH = "dataset_2/insurance.csv"


class MyLinearRegression:
    """
    Ordinary least squares Linear Regression.
    Parameters
    ----------
    num_iterations: int, default=200
        Number of iterations.

    learning_rate = float, default=0.005
    """
    def __init__(self, num_iterations=4000, learning_rate=0.5):
        self.num_iterations = num_iterations
        self.learning_rate = learning_rate

    def fit_and_predict(self, X_train, Y_train, X_test):
        """
        Fit the model according to the given training data
        and predict values for samples in X.
        Parameters
        ----------
        X_train : matrix of shape (n_features, n_samples)
            Training vector, where n_samples is the number of samples and
            n_features is the number of features.
        Y_train : array of shape (1, n_samples)
            Target vector relative to X.
        X_test: matrix of shape (n_features, n_samples)
            Test vector, where n_samples is the number of samples and
            n_features is the number of features.
        Returns
        -------
        prediction: OrderedDict
            Ordered dictionary with predicted values for train and test sets.
        """
        def init_with_zeros(dim):
            """
            Creates an array of zeros of shape (dim, 1) for w and initializes b to 0.
            Parameters
            ----------
            dim: int
                size of the weights array
            Returns
            -------
            weights: array of shape (dim, 1)
            bias: float
            """
            weights = np.zeros([dim, 1])
            bias = 0
            return weights, bias

        def predict(weights, bias, X):
            """
            Predict values for samples in X
            Parameters
            ----------
            weights: array of shape (n_features, 1)
            bias: float
            X: matrix of shape (n_features, n_samples)
                Vector, where n_samples is the number of samples and
                n_features is the number of features.
            Returns
            -------
            Predicted values for X
            """
            H = np.dot(weights.T, X) + bias
            return H

        w, b = init_with_zeros(X_train.shape[0])

        w, b = self.optimize(w, b, X_train, Y_train)

        Y_prediction_test = predict(w, b, X_test)
        Y_prediction_train = predict(w, b, X_train)

        prediction = OrderedDict({"Training set": Y_prediction_train, "Test set": Y_prediction_test})

        return prediction

    def optimize(self, w, b, X, Y):
        """
        Update the weight and bias parameters using gradient descent.
        Parameters
        ----------
        w: array of shape (n_features, 1)
        b: float
        X: matrix of shape (n_features, n_samples)
        Y: array of shape (1, n_samples)

        Returns
        -------
        w: array of shape (n_features, 1)
        b: float
            Optimized parameters
        """

        def propagate(weights, bias, X_set, Y_set):
            """
            Calculates the cost function and its gradient.
            Parameters
            ----------
            weights: array of shape (n_features, 1)
            bias: float
            X_set: matrix of shape (n_features, n_samples)
                Training vector, where n_samples is the number of samples and
                n_features is the number of features.
            Y_set: array of shape (1, n_samples)
                Target vector relative to X.
            Returns
            -------
            dw_value: array of shape (n_features, 1)
                gradient of the loss with respect to weights
            db_value: float
                gradient of the loss with respect to bias
            cost_function: cost function for linear regression
            """
            m = X_set.shape[1]
            H = np.dot(weights.T, X_set) + bias
            cost_function = np.sum((H - Y_set) ** 2) / (2 * m)

            dw_value = (1 / m) * (np.dot(X_set, (H - Y_set).T))
            db_value = (1 / m) * np.sum(H - Y_set)
            return dw_value, db_value, cost_function

        for i in range(self.num_iterations):
            dw, db, cost = propagate(w, b, X, Y)

            w = w - self.learning_rate * dw
            b = b - self.learning_rate * db
            return w, b


def prepare_data(path):
    """
    Dataset preparation and train-test split.
    Parameters
    ----------
    path : str
        Dataset path.
    Returns
    -------
    Train-test split of dataset.

    """
    insurance = pd.read_csv(path)
    insurance.drop_duplicates(inplace=True)
    insurance['sex'] = [1 if x == "male" else 0 for x in insurance['sex']]
    insurance['smoker'] = [1 if x == "yes" else 0 for x in insurance['smoker']]
    insurance = pd.get_dummies(insurance)
    insurance.astype(float)

    shuffled_insurance = insurance.sample(frac=1)

    train_size = int(0.8 * len(insurance))

    train_set = shuffled_insurance[:train_size].reset_index(drop=True)
    test_set = shuffled_insurance[train_size:].reset_index(drop=True)

    train_X = train_set.drop(['charges'], axis=1).to_numpy().T
    train_y = train_set['charges'].to_frame().to_numpy().T
    test_X = test_set.drop(['charges'], axis=1).to_numpy().T
    test_y = test_set['charges'].to_frame().to_numpy().T

    all_set_X = np.concatenate([train_X, test_X], axis=1)
    max_value = all_set_X.max(axis=1, keepdims=True)
    train_X = train_X / max_value
    test_X = test_X / max_value

    return train_X, train_y, test_X, test_y


def mse(Y_pred, Y_true):
    """
    Mean squared error regression loss.
    Parameters
    ----------
    Y_true : array of shape (1, n_samples)
        Ground truth (correct) target values.
    Y_pred : array of shape (1, n_samples)
        Estimated target values.

    Returns
    -------
    Calculated regression loss.
    """
    return np.mean((Y_pred - Y_true) ** 2)


train_set_X, train_set_y, test_set_X, test_set_y = prepare_data(PATH)

my_model = MyLinearRegression()
predictions = my_model.fit_and_predict(train_set_X, train_set_y, test_set_X)

train_pred_y = predictions['Training set']
test_pred_y = predictions['Test set']

mse_metrics = OrderedDict({'Training set mse': mse(train_pred_y, train_set_y),
                           'Test set mse': mse(test_pred_y, test_set_y)})

print(mse_metrics)
