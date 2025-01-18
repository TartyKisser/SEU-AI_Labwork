import pandas as pd
import numpy as np
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
        """存储训练数据"""
        self.X_train = X
        self.y_train = y

    def predict(self, X):
        """对提供的数据进行预测"""
        y_pred = [self._predict(x) for x in X]
        return np.array(y_pred)

    def _predict(self, x):
        """对单个数据点进行预测"""
        # 计算距离
        distances = np.sqrt(np.sum((self.X_train - x) ** 2, axis=1))
        # 获取最近的 k 个点的索引
        k_indices = np.argsort(distances)[:self.n_neighbors]
        # 获取这些点的类别
        k_nearest_labels = self.y_train[k_indices]
        # 使用多数投票法确定预测类别，这里我们不使用 Counter，改用 np.bincount
        label = np.argmax(np.bincount(k_nearest_labels))
        return label


class M_KNeighborsClassifier:
    def __init__(self, n_neighbors=3):
        self.n_neighbors = n_neighbors
        self.X_train = None
        self.y_train = None
        self.inv_cov_matrix = None  # 马氏距离所需的协方差矩阵的逆

    def fit(self, X, y):
        """存储训练数据并计算协方差矩阵的逆"""
        self.X_train = X
        self.y_train = y
        # 计算协方差矩阵及其逆
        cov_matrix = np.cov(X, rowvar=False)
        self.inv_cov_matrix = np.linalg.inv(cov_matrix)

    def predict(self, X):
        """对提供的数据进行预测"""
        y_pred = [self._predict(x) for x in X]
        return np.array(y_pred)

    def _predict(self, x):
        """对单个数据点使用马氏距离进行预测"""
        # 计算马氏距离
        distances = [np.sqrt((x - x_train).T @ self.inv_cov_matrix @ (x - x_train)) for x_train in self.X_train]
        # 获取最近的 k 个点的索引
        k_indices = np.argsort(distances)[:self.n_neighbors]
        # 获取这些点的类别
        k_nearest_labels = self.y_train[k_indices]
        # 使用多数投票法确定预测类别
        label = np.argmax(np.bincount(k_nearest_labels))
        return label


# 加载数据
train_data = pd.read_csv('./data/train.csv')
val_data = pd.read_csv('./data/val.csv')
test_data = pd.read_csv('./data/test_data.csv')

# 准备数据
X_train = train_data.iloc[:, :-1].values
y_train = train_data.iloc[:, -1].values
X_val = val_data.iloc[:, :-1].values
y_val = val_data.iloc[:, -1].values
X_test = test_data.iloc[:, :].values  # 假设测试数据没有标签

# 数据标准化
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)

my_E_accuracies = []
my_M_accuracies = []
sk_accuracies = []

for k_value in range(1,20):

    # 创建 KNN 模型
    my_E_knn = E_KNeighborsClassifier(n_neighbors=k_value)
    my_M_knn = M_KNeighborsClassifier(n_neighbors=k_value)
    sk_knn = KNeighborsClassifier(n_neighbors=k_value)
    # 训练模型
    my_E_knn.fit(X_train, y_train)
    my_M_knn.fit(X_train, y_train)
    sk_knn.fit(X_train, y_train)
    # 在验证集上评估模型
    y_pred_val_E = my_E_knn.predict(X_val)
    y_pred_val_M = my_M_knn.predict(X_val)
    y_pred_val_sk = sk_knn.predict(X_val)
    my_E_accuracies.append(accuracy_score(y_val, y_pred_val_E))
    my_M_accuracies.append(accuracy_score(y_val, y_pred_val_M))
    sk_accuracies.append(accuracy_score(y_val, y_pred_val_sk))


def plot_accuracy_vs_k(k_values, my_E_accuracies, my_M_accuracies, sk_accuracies):
    """
    绘制不同k值对应的三组准确率。

    参数:
    k_values : list 或 numpy array
        KNN 模型中尝试的 k 值列表。
    my_E_accuracies : list 或 numpy array
        使用自定义算法 E 实现的 KNN 模型的每个 k 值对应的准确率。
    my_M_accuracies : list 或 numpy array
        使用自定义算法 M 实现的 KNN 模型的每个 k 值对应的准确率。
    sk_accuracies : list 或 numpy array
        使用scikit-learn实现的 KNN 模型的每个 k 值对应的准确率。
    """
    plt.figure(figsize=(10, 6))  # 设置图像大小
    plt.plot(k_values, my_E_accuracies, marker='o', linestyle='-', color='b', label='My KNN Accuracy (E)')  # 绘制自定义KNN (E) 的准确率曲线
    plt.plot(k_values, my_M_accuracies, marker='^', linestyle='-.', color='g', label='My KNN Accuracy (M)')  # 绘制自定义KNN (M) 的准确率曲线
    plt.plot(k_values, sk_accuracies, marker='s', linestyle='--', color='r', label='Scikit-Learn KNN Accuracy')  # 绘制scikit-learn KNN的准确率曲线
    plt.title('Accuracy vs. Number of Neighbors (k)')  # 图像标题
    plt.xlabel('Number of Neighbors (k)')  # x轴标签
    plt.ylabel('Accuracy')  # y轴标签
    plt.xticks(k_values)  # 设置x轴的刻度为给定的 k 值
    plt.grid(True)  # 显示网格
    plt.legend()  # 显示图例
    plt.show()  # 显示图像



k_values = np.arange(1, 20)  # k 从 1 到 20
plot_accuracy_vs_k(k_values, my_E_accuracies, my_M_accuracies, sk_accuracies)
