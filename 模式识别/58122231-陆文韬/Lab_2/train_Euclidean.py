import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt


class KNeighborsClassifier:
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

accuracies = []

for k_value in range(1,10):

    # 创建 KNN 模型
    knn = KNeighborsClassifier(n_neighbors=k_value)

    # 训练模型
    knn.fit(X_train, y_train)

    # 在验证集上评估模型
    y_pred_val = knn.predict(X_val)
    accuracies.append(accuracy_score(y_val, y_pred_val))


def plot_accuracy_vs_k(k_values, accuracies):
    """
    绘制不同k值对应的准确率。

    参数:
    k_values : list 或 numpy array
        KNN 模型中尝试的 k 值列表。
    accuracies : list 或 numpy array
        每个 k 值对应的准确率。
    """
    plt.figure(figsize=(10, 6))  # 设置图像大小
    plt.plot(k_values, accuracies, marker='o', linestyle='-', color='b')  # 绘制准确率曲线
    plt.title('Accuracy vs. Number of Neighbors (k)')  # 图像标题
    plt.xlabel('Number of Neighbors (k)')  # x轴标签
    plt.ylabel('Accuracy')  # y轴标签
    plt.xticks(k_values)  # 设置x轴的刻度为给定的 k 值
    plt.grid(True)  # 显示网格
    plt.show()  # 显示图像


k_values = np.arange(1, 10)  # k 从 1 到 47
plot_accuracy_vs_k(k_values, accuracies)
