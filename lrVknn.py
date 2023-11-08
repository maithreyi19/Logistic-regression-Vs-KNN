import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier

# Generate some data
X = np.array([[1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7]])
y = np.array([0, 0, 0, 1, 1, 1])

# Train a logistic regression model
model = LogisticRegression(solver='lbfgs', C=1e5)
model.fit(X, y)

# Train a KNN model
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X, y)

# Make some predictions
X_new = np.array([[5, 5], [4, 4]])
y_pred_logistic = model.predict(X_new)
y_pred_knn = knn.predict(X_new)

print(y_pred_logistic)
print(y_pred_knn)
