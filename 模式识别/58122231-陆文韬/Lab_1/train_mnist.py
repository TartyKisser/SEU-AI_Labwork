# 引用了来自sklearn的数据集mnist
import pandas as pd
from sklearn.datasets import load_digits
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, StandardScaler


# 自定义的LogisticRegression类
class my_LogisticRegression:
    def __init__(self, learning_rate=0.01, n_iter=1000, decay=0.0, reg_lambda=0.01):
        """
        多类逻辑回归模型初始化。
        :param lr: 学习率
        :param epochs: 迭代次数
        :param decay: 学习率衰减
        :param reg_lambda: 正则化参数
        """
        self.learning_rate = learning_rate
        self.n_iter = n_iter
        self.decay = decay
        self.reg_lambda = reg_lambda
        self.weights = None
        self.bias = None
        self.history = []

    def fit(self, X, y):
        """
        拟合多类逻辑回归模型。
        :param X: 特征数据
        :param y: 标签数据
        """
        num_samples, num_features = X.shape
        num_classes = np.max(y) + 1
        self.weights = np.zeros((num_features, num_classes))
        self.bias = np.zeros(num_classes)
        y_encoded = self._one_hot_encode(y, num_classes)

        for epoch in range(self.n_iter):
            model = np.dot(X, self.weights) + self.bias
            y_pred = self._softmax(model)
            loss = self._compute_loss(y_encoded, y_pred)
            self.history.append({'train loss': loss})

            grad_weights = np.dot(X.T, (y_pred - y_encoded)) / num_samples + self.reg_lambda * self.weights
            grad_bias = np.mean(y_pred - y_encoded, axis=0)

            self.weights -= self.learning_rate * grad_weights
            self.bias -= self.learning_rate * grad_bias

            self.learning_rate *= (1. / (1. + self.decay * epoch))

    def predict(self, X):
        """
        预测给定数据的类别标签。
        :param X: 特征数据
        """
        model = np.dot(X, self.weights) + self.bias
        y_preds = self._softmax(model)
        return np.argmax(y_preds, axis=1)

    def _softmax(self, z):
        e = np.exp(z - np.max(z, axis=1, keepdims=True))
        return e / np.sum(e, axis=1, keepdims=True)

    def _compute_loss(self, y, y_pred):
        epsilon = 1e-10
        y_pred = np.clip(y_pred, epsilon, 1.0 - epsilon)
        return -np.mean(np.sum(y * np.log(y_pred), axis=1))

    def _one_hot_encode(self, y, num_classes):
        return np.eye(num_classes)[y].astype(float)


# 自定义的accuracy函数
def my_accuracy(y_true, y_pred):
    length=len(y_true)
    return sum(y_true == y_pred) / length


# 自定义的PCA类
class my_PCA:
    def __init__(self, n_components):
        self.n_components = n_components
        self.components = None
        self.mean = None

    def fit_transform(self, X):
        # 计算数据的平均值，并中心化数据
        self.mean = np.mean(X, axis=0)
        X_centered = X - self.mean

        # 计算协方差矩阵
        covariance_matrix = np.cov(X_centered, rowvar=False)

        # 计算协方差矩阵的特征值和特征向量
        eigenvalues, eigenvectors = np.linalg.eigh(covariance_matrix)

        # 对特征值进行排序
        sorted_indices = np.argsort(eigenvalues)[::-1]
        sorted_eigenvalues = eigenvalues[sorted_indices]
        sorted_eigenvectors = eigenvectors[:, sorted_indices]

        # 如果n_components是浮点数，解释为方差的百分比
        if isinstance(self.n_components, float):
            total_variance = np.sum(sorted_eigenvalues)
            variance_ratio = sorted_eigenvalues / total_variance
            cumulative_variance = np.cumsum(variance_ratio)
            # 找到累积贡献率至少为n_components的特征数
            num_components = np.where(cumulative_variance >= self.n_components)[0][0] + 1
            self.components = sorted_eigenvectors[:, :num_components]
        else:
            # 否则，直接选取前n_components个特征向量
            self.components = sorted_eigenvectors[:, :self.n_components]

        # 将数据投影到选定的特征向量上
        X_reduced = X_centered.dot(self.components)
        return X_reduced


class my_LinearDiscriminantAnalysis:
    def __init__(self):
        self.means_ = None
        self.scalings_ = None

    def fit_transform(self, X, y):
        # 计算每个类的平均值
        class_labels = np.unique(y)
        mean_overall = np.mean(X, axis=0)
        S_W = np.zeros((X.shape[1], X.shape[1]))
        S_B = np.zeros((X.shape[1], X.shape[1]))

        for label in class_labels:
            X_c = X[y == label]
            mean_c = np.mean(X_c, axis=0)
            S_W += np.cov(X_c, rowvar=False) * (X_c.shape[0] - 1)
            mean_diff = (mean_c - mean_overall).reshape(-1, 1)
            S_B += X_c.shape[0] * (mean_diff @ mean_diff.T)

        # 添加正则化以避免奇异矩阵
        lambda_ = 1e-5  # 正则化系数
        S_W += lambda_ * np.eye(S_W.shape[0])

        # S_W为类内散度矩阵，S_B为类间散度矩阵
        S_W_inv = np.linalg.inv(S_W)
        eigvals, eigvecs = np.linalg.eigh(S_W_inv @ S_B)

        # 排序并选取特征向量
        sorted_indices = np.argsort(eigvals)[::-1]
        self.scalings_ = eigvecs[:, sorted_indices]

        # 投影数据
        X_transformed = X @ self.scalings_
        return X_transformed

# 示例使用
# lda = my_LinearDiscriminantAnalysis()
# X_transformed = lda.fit_transform(X, y)


def remove_outliers(df, n=1.5):
    """Remove outliers from a dataframe based on the IQR method."""
    for col in df.columns[:-1]:  # 假设最后一列是标签，不进行处理
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        filter = (df[col] >= (Q1 - n * IQR)) & (df[col] <= (Q3 + n * IQR))
        df = df.loc[filter]
    df.reset_index(drop=True, inplace=True)
    return df


def standardize_features(df):
    """Standardize features to have zero mean and unit variance."""
    features = df.iloc[:, :-1]
    standardized_features = (features - features.mean()) / features.std()
    df.iloc[:, :-1] = standardized_features
    return df


def train_test_split(X, y):
    """手动实现的简单的训练测试数据集分割函数。

    参数:
    X -- 特征数据集
    y -- 标签数据集

    返回:
    X_train, X_test, y_train, y_test -- 分割后的训练和测试数据集
    """
    # 确保数据和标签的长度一致
    assert len(X) == len(y), "特征和标签的长度必须相同。"

    # 计算测试集的大小（30%）
    test_size = int(len(X) * 0.3)

    # 生成随机的索引
    indices = np.arange(len(X))
    np.random.shuffle(indices)

    # 根据随机索引划分训练集和测试集
    X_test = X[indices[:test_size]]
    y_test = y[indices[:test_size]]
    X_train = X[indices[test_size:]]
    y_train = y[indices[test_size:]]

    return X_train, X_test, y_train, y_test


n_components = 0.9
data = {}
scaler = StandardScaler()

# 提取特征和标签
digits = load_digits()
X = scaler.fit_transform(digits.data)
y = digits.target


data['my PCA'] = my_PCA(n_components=n_components).fit_transform(X)
data['my LDA'] = my_LinearDiscriminantAnalysis().fit_transform(X, y)
data['sklearn PCA'] = PCA(n_components=n_components).fit_transform(X)
data['sklearn LDA'] = LinearDiscriminantAnalysis().fit_transform(X, y)


def plot_result(data):
    # 设置图表大小和子图布局
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 6))
    fig.supxlabel("Epochs")
    fig.supylabel("Train Loss")

    # 遍历数据集，分别训练使用PCA和其他方法处理的数据
    for label in data:
        # 初始化模型并划分数据

        model = my_LogisticRegression()
        X_train, X_test, y_train, y_test = train_test_split(data[label], y)
        model.fit(X_train, y_train)

        # 输出训练结果的准确率
        accuracy_val = my_accuracy(y_test, model.predict(X_test))
        print(f'Train on {label}, accuracy: {accuracy_val}')

        # 获取训练过程中的损失历史
        history = [x['train loss'] for x in model.history]

        # 选择合适的子图来绘制历史数据
        ax = ax1 if 'PCA' in label else ax2
        ax.plot(range(1, len(history) + 1), history, label=f'{label} (Acc: {accuracy_val:.2f})')

    # 设置图例和图形布局
    ax1.legend(title='PCA Methods')
    ax2.legend(title='LDA Methods')
    fig.tight_layout()
    plt.show()


plot_result(data)
