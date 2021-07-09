import numpy as np
import pandas as pd
from collections import OrderedDict

PATH = "dataset_1/processed.cleveland.data"


class MyLogisticRegression:
    """
    Logistic Regression classifier.
    Parameters
    ----------
    num_iterations: int, default=200
        Number of iterations.

    learning_rate = float, default=0.005
    """
    def __init__(self, num_iterations=2000, learning_rate=0.005):
        self.num_iterations = num_iterations
        self.learning_rate = learning_rate

    def fit_and_predict(self, X_train, Y_train, X_test):
        """
        Fit the model according to the given training data
        and predict class labels for samples in X.
        Parameters
        ----------
        X_train: matrix of shape (n_features, n_samples)
            Training vector, where n_samples is the number of samples and
            n_features is the number of features.
        Y_train: array of shape (1, n_samples)
            Target vector relative to X.
        X_test: matrix of shape (n_features, n_samples)
            Test vector, where n_samples is the number of samples and
            n_features is the number of features.

        Returns
        -------
        prediction: OrderedDict
            Ordered dictionary with predicted class labels for train and test sets.
        """

        def init_with_zeros(dim):
            """
            Creates an array of zeros of shape (dim, 1) for w and initializes b to 0.
            Parameters
            ----------
            dim: int
                size of the weights aaray
            Returns
            -------
            weights: array of shape (dim, 1)
            bias: float
            """
            weights = np.zeros([dim, 1])
            bias = 0
            return weights, bias

        def predict_y(weights, bias, X):
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
            m = X.shape[1]
            Y_prediction = np.zeros((1, m))

            A = self.sigmoid(np.dot(weights.T, X) + bias)

            for i in range(A.shape[1]):
                if A[0, i] > 0.5:
                    Y_prediction[0, i] = 1
                else:
                    Y_prediction[0, i] = 0

            return Y_prediction

        w, b = init_with_zeros(X_train.shape[0])

        w, b = self.optimize(w, b, X_train, Y_train)

        Y_prediction_test = predict_y(w, b, X_test)
        Y_prediction_train = predict_y(w, b, X_train)

        prediction = OrderedDict({"Training set": Y_prediction_train, "Test set": Y_prediction_test})

        return prediction

    def sigmoid(self, z):
        """Calculates the value of sigmoid function"""
        return 1 / (1 + np.exp(-z))

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
            cost_function: cost function for logistic regression
            """
            m = X_set.shape[1]

            A = self.sigmoid(np.dot(weights.T, X_set) + bias)
            cost_function = (-1 / m) * np.sum(Y_set * np.log(A) + (1 - Y_set) * (np.log(1 - A)))

            dw_value = (1 / m) * np.dot(X_set, (A - Y_set).T)
            db_value = (1 / m) * np.sum(A - Y_set)

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
    health = pd.read_csv(path)
    health.columns = ['age', 'sex', 'cp', 'rbp', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca',
                      'thal', 'target']
    health = health.replace('?', np.nan).dropna().reset_index(drop=True)
    health['target'] = health['target'].replace([2, 3, 4], 1)
    health = health.astype(float)

    health['cp'][health['cp'] == 1] = 'typical angina'
    health['cp'][health['cp'] == 2] = 'atypical angina'
    health['cp'][health['cp'] == 3] = 'non-anginal pain'
    health['cp'][health['cp'] == 4] = 'asymptomatic'

    health['restecg'][health['restecg'] == 0] = 'normal'
    health['restecg'][health['restecg'] == 1] = 'ST-T wave abnormality'
    health['restecg'][health['restecg'] == 2] = 'left ventricular hypertrophy'

    health['slope'][health['slope'] == 1] = 'upsloping'
    health['slope'][health['slope'] == 2] = 'flat'
    health['slope'][health['slope'] == 3] = 'downsloping'

    health['thal'][health['thal'] == 3.0] = 'normal'
    health['thal'][health['thal'] == 6.0] = 'fixed defect'
    health['thal'][health['thal'] == 7.0] = 'reversable defect'

    health = pd.get_dummies(health)

    s_health = health.sample(frac=1)

    train_size = int(0.8 * len(health))

    train_set = s_health[:train_size].reset_index(drop=True)
    test_set = s_health[train_size:].reset_index(drop=True)

    train_X = train_set.drop(['target'], axis=1).to_numpy().T
    train_y = train_set['target'].to_frame().to_numpy().T
    test_X = test_set.drop(['target'], axis=1).to_numpy().T
    test_y = test_set['target'].to_frame().to_numpy().T

    all_set_X = np.concatenate([train_X, test_X], axis=1)
    max_value = all_set_X.max(axis=1, keepdims=True)
    train_X = train_X / max_value
    test_X = test_X / max_value

    return train_X, train_y, test_X, test_y


def accuracy(Y_pred, Y_true):
    """
    Accuracy classification score.
    Parameters
    ----------
    Y_true : 1d label indicator array
        Ground truth (correct) target values.
    Y_pred : 1d label indicator array
        Estimated target values.

    Returns
    -------
    The fraction of correctly classified samples

    """
    n = Y_true.shape[1]
    return (Y_true == Y_pred).sum() / n


def precision(Y_pred, Y_true):
    """
    Compute the precision.
    The precision is the ratio ``tp / (tp + fp)`` where ``tp`` is the number of
    true positives and ``fp`` the number of false positives. The precision is
    intuitively the ability of the classifier not to label as positive a sample
    that is negative.
    Parameters
    ----------
    Y_true : 1d label indicator array
        Ground truth (correct) target values.
    Y_pred : 1d label indicator array
        Estimated target values.

    Returns
    -------
    Precision of the positive class in binary classification
    """
    tp = ((Y_pred == 1) & (Y_true == 1)).sum()
    fp = ((Y_pred == 1) & (Y_true == 0)).sum()
    return tp / (tp + fp)


train_set_X, train_set_y, test_set_X, test_set_y = prepare_data(PATH)

my_model = MyLogisticRegression()
predictions = my_model.fit_and_predict(train_set_X, train_set_y, test_set_X)

train_pred_y = predictions['Training set']
test_pred_y = predictions['Test set']

accuracy_metrics = OrderedDict({'Training set accuracy': accuracy(train_pred_y, train_set_y),
                                'Test set accuracy': accuracy(test_pred_y, test_set_y)})

precision_metrics = OrderedDict({'Training set precision': precision(train_pred_y, train_set_y),
                                'Test set precision': precision(test_pred_y, test_set_y)})

print(accuracy_metrics)
print(precision_metrics)
