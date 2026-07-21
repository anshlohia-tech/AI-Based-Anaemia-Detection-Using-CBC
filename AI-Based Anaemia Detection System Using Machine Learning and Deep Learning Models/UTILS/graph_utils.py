import pandas as pd
import numpy as np
import torch

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder


def build_graph(features, k=10):
    """
    Build a k-NN graph using Euclidean distance.
    """

    n = len(features)

    distances = np.sqrt(
        ((features[:, None] - features[None, :]) ** 2).sum(axis=2)
    )

    adjacency = np.zeros((n, n), dtype=np.float32)

    for i in range(n):

        neighbors = np.argsort(distances[i])[1:k + 1]

        adjacency[i, neighbors] = 1

    # Make graph undirected
    adjacency = np.maximum(adjacency, adjacency.T)

    # Add self-loops
    adjacency += np.eye(n)

    # Degree matrix
    degree = np.sum(adjacency, axis=1)

    degree_inv = np.diag(1.0 / np.sqrt(degree))

    adjacency = degree_inv @ adjacency @ degree_inv

    return adjacency.astype(np.float32)


def load_graph_data(csv_path):

    print("\nLoading Dataset...")

    df = pd.read_csv(csv_path)

    print(df.head())

    target = "Diagnosis"

    X = df.drop(columns=[target])

    y = df[target]

    scaler = StandardScaler()

    X = scaler.fit_transform(X)

    encoder = LabelEncoder()

    y = encoder.fit_transform(y)

    adjacency = build_graph(X)

    X = torch.tensor(X, dtype=torch.float32)

    y = torch.tensor(y, dtype=torch.long)

    adjacency = torch.tensor(adjacency, dtype=torch.float32)

    return X, y, adjacency, encoder