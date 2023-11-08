import numpy as np

def knn(X_train, y_train, X_test, k=5):
    # Calculate the distances between the test points and all training points
    distances = np.zeros((X_test.shape[0], X_train.shape[0]))
    for i in range(X_test.shape[0]):
        for j in range(X_train.shape[0]):
            distances[i, j] = np.linalg.norm(X_test[i] - X_train[j])

    # Find the k nearest neighbors for each test point
    nearest_neighbors = np.argsort(distances, axis=1)[:, :k]

    # Predict the labels for the test points based on the labels of their nearest neighbors
    y_pred = []
    for i in range(X_test.shape[0]):
        # Get the labels of the k nearest neighbors
        knn_labels = y_train[nearest_neighbors[i]]

        # Count the occurrences of each label
        label_counts = np.bincount(knn_labels)

        # Predict the label with the most occurrences
        y_pred.append(np.argmax(label_counts))

    return y_pred
