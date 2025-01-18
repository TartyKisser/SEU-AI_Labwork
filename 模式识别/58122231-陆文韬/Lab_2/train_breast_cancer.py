import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

class E_KNeighborsClassifier:
    def __init__(self, n_neighbors=3):
        self.n_neighbors = n_neighbors
        self.X_train = None
        self.y_train = None

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    def predict(self, X):
        y_pred = [self._predict(x) for x in X]
        return np.array(y_pred)

    def _predict(self, x):
        distances = np.sqrt(np.sum((self.X_train - x) ** 2, axis=1))
        k_indices = np.argsort(distances)[:self.n_neighbors]
        k_nearest_labels = self.y_train[k_indices]
        return np.argmax(np.bincount(k_nearest_labels))

class M_KNeighborsClassifier:
    def __init__(self, n_neighbors=3):
        self.n_neighbors = n_neighbors
        self.X_train = None
        self.y_train = None

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    def predict(self, X):
        y_pred = [self._predict(x) for x in X]
        return np.array(y_pred)

    def _predict(self, x):
        distances = np.sqrt(np.sum((self.X_train - x) ** 2, axis=1))
        k_indices = np.argsort(distances)[:self.n_neighbors]
        k_nearest_labels = self.y_train[k_indices]
        return np.argmax(np.bincount(k_nearest_labels))

# 加载乳腺癌数据集
data = load_breast_cancer()
X, y = data.data, data.target

# 划分数据集
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# 数据标准化
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)

my_E_accuracies = []
my_M_accuracies = []
sk_accuracies = []

for k_value in range(1, 20):
    my_E_knn = E_KNeighborsClassifier(n_neighbors=k_value)
    my_M_knn = M_KNeighborsClassifier(n_neighbors=k_value)
    sk_knn = KNeighborsClassifier(n_neighbors=k_value)

    my_E_knn.fit(X_train, y_train)
    my_M_knn.fit(X_train, y_train)
    sk_knn.fit(X_train, y_train)

    y_pred_val_E = my_E_knn.predict(X_val)
    y_pred_val_M = my_M_knn.predict(X_val)
    y_pred_val_sk = sk_knn.predict(X_val)

    my_E_accuracies.append(accuracy_score(y_val, y_pred_val_E))
    my_M_accuracies.append(accuracy_score(y_val, y_pred_val_M))
    sk_accuracies.append(accuracy_score(y_val, y_pred_val_sk))

def plot_accuracy_vs_k(k_values, my_E_accuracies, my_M_accuracies, sk_accuracies):
    plt.figure(figsize=(10, 6))
    plt.plot(k_values, my_E_accuracies, marker='o', linestyle='-', color='b', label='My KNN Accuracy (E)')
    plt.plot(k_values, my_M_accuracies, marker='^', linestyle='-.', color='g', label='My KNN Accuracy (M)')
    plt.plot(k_values, sk_accuracies, marker='s', linestyle='--', color='r', label='Scikit-Learn KNN Accuracy')
    plt.title('Accuracy vs. Number of Neighbors (k) on Breast Cancer Dataset')
    plt.xlabel('Number of Neighbors (k)')
    plt.ylabel('Accuracy')
    plt.xticks(k_values)
    plt.grid(True)
    plt.legend()
    plt.show()


k_values = np.arange(1, 20)
plot_accuracy_vs_k(k_values, my_E_accuracies, my_M_accuracies, sk_accuracies)
