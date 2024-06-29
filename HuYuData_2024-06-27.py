import numpy as np
import pandas as pd

class GradientBoosting:
    def __init__(self, n_estimators=100, learning_rate=0.1):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.models = []

    def fit(self, X, y):
        residuals = y
        for _ in range(self.n_estimators):
            model = DecisionStump()
            model.fit(X, residuals)
            self.models.append(model)
            predictions = model.predict(X)
            residuals -= self.learning_rate * predictions

    def predict(self, X):
        predictions = np.zeros(X.shape[0])
        for model in self.models:
            predictions += self.learning_rate * model.predict(X)
        return predictions

class DecisionStump:
    def __init__(self):
        self.feature_index = None
        self.threshold = None
        self.left_value = None
        self.right_value = None

    def fit(self, X, y):
        m, n = X.shape
        best_error = float('inf')

        for feature_index in range(n):
            thresholds = np.unique(X[:, feature_index])
            for threshold in thresholds:
                left_mask = X[:, feature_index] < threshold
                right_mask = X[:, feature_index] >= threshold
                left_value = y[left_mask].mean() if left_mask.any() else 0
                right_value = y[right_mask].mean() if right_mask.any() else 0
                predictions = np.where(left_mask, left_value, right_value)
                error = np.mean((y - predictions) ** 2)

                if error < best_error:
                    best_error = error
                    self.feature_index = feature_index
                    self.threshold = threshold
                    self.left_value = left_value
                    self.right_value = right_value

    def predict(self, X):
        left_mask = X[:, self.feature_index] < self.threshold
        return np.where(left_mask, self.left_value, self.right_value)

# Example dataset
data = {
    'feature1': [1, 2, 3, 4, 5],
    'feature2': [2, 3, 4, 5, 6],
    'target': [1.5, 2.5, 3.5, 4.5, 5.5]
}
df = pd.DataFrame(data)
X = df[['feature1', 'feature2']].values
y = df['target'].values

# Create and train the model
gb_model = GradientBoosting(n_estimators=10, learning_rate=0.1)
gb_model.fit(X, y)

# Make predictions
predictions = gb_model.predict(X)
print("Predictions:", predictions)