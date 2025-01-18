import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


# 自定义的LogisticRegression类
class LogisticRegression:
    def __init__(self, learning_rate=0.01, n_iter=1000):
        self.learning_rate = learning_rate
        self.n_iter = n_iter
        self.weights = None
        self.bias = None
        self.history = []

    def _sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def _compute_loss(self, y, y_pred):
        m = y.shape[0]
        return -np.sum(y * np.log(y_pred) + (1 - y) * np.log(1 - y_pred)) / m

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        for _ in range(self.n_iter):
            linear_model = np.dot(X, self.weights) + self.bias
            y_pred = self._sigmoid(linear_model)
            errors = y - y_pred
            dw = np.dot(X.T, errors) / n_samples
            db = np.sum(errors) / n_samples
            self.weights += self.learning_rate * dw
            self.bias += self.learning_rate * db

            # 计算并记录损失，以字典形式存储
            loss = self._compute_loss(y, y_pred)
            self.history.append({'train loss': loss})

    def predict_proba(self, X):
        linear_output = np.dot(X, self.weights) + self.bias
        return self._sigmoid(linear_output)

    def predict(self, X):
        probabilities = self.predict_proba(X)
        return np.where(probabilities >= 0.5, 1, 0)

# 自定义的accuracy函数
def accuracy(y_true, y_pred):
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

        # 对特征值进行排序，并选取最大的n_components个特征向量
        sorted_indices = np.argsort(eigenvalues)[::-1]
        selected_eigenvectors = eigenvectors[:, sorted_indices][:, :self.n_components]

        # 将数据投影到选定的特征向量上
        self.components = selected_eigenvectors.T
        X_reduced = X_centered.dot(self.components.T)
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

        # S_W为类内散度矩阵，S_B为类间散度矩阵
        S_W_inv = np.linalg.inv(S_W)
        eigvals, eigvecs = np.linalg.eigh(S_W_inv @ S_B)

        # 排序并选取特征向量
        sorted_indices = np.argsort(eigvals)[::-1]
        self.scalings_ = eigvecs[:, sorted_indices]

        # 投影数据
        X_transformed = X @ self.scalings_
        return X_transformed


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


n_components = 9  # 保留9个主成分
data = {}

# 对原始数据集进行合并处理，红酒和白酒数据集合并之后增加一列category，表示酒的类别，白酒为0，红酒为1
df_wine_white = pd.read_csv(r'data/winequality-white.csv', header=0,
                            sep=';').drop_duplicates().assign(category=0)
df_wine_red = pd.read_csv(r'data/winequality-red.csv', header=0,
                          sep=';').drop_duplicates().assign(category=1)
df_wine = pd.concat([df_wine_white, df_wine_red], axis=0)

# 移除异常值
df_wine = remove_outliers(df_wine)

# 标准化特征
df_wine = standardize_features(df_wine)

# 提取特征和标签
X = df_wine.iloc[:, :-1].to_numpy()
y = df_wine.iloc[:, -1].to_numpy()


data['my PCA'] = my_PCA(n_components=n_components).fit_transform(X)
data['my LDA'] = my_LinearDiscriminantAnalysis().fit_transform(X, y)
data['original_data_PCA'] = X.copy()
data['original_data_LDA'] = df_wine.to_numpy()


def plot_result(data):
    # 设置图表大小和子图布局
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 6))
    fig.supxlabel("Epochs")
    fig.supylabel("Train Loss")

    # 颜色列表
    colors = ['blue', 'green', 'red', 'orange', 'brown']

    # 遍历数据集，分别训练使用PCA和其他方法处理的数据
    color_index = 0  # 初始化颜色索引
    for label in data:
        # 初始化模型并划分数据
        model = LogisticRegression()
        X_train, X_test, y_train, y_test = train_test_split(data[label], y)
        model.fit(X_train, y_train)

        # 输出训练结果的准确率
        accuracy_val = accuracy(y_test, model.predict(X_test))
        print(f'Train on {label}, accuracy: {accuracy_val}')

        # 获取训练过程中的损失历史
        history = [x['train loss'] for x in model.history]

        # 选择合适的子图来绘制历史数据
        ax = ax1 if 'PCA' in label else ax2
        ax.plot(range(1, len(history) + 1), history, color=colors[color_index % len(colors)], label=f'{label} (Acc: {accuracy_val:.2f})')
        color_index += 1  # 更新颜色索引

    # 设置图例和图形布局
    ax1.legend(title='PCA Methods')
    ax2.legend(title='LDA Methods')
    fig.tight_layout()
    plt.show()


plot_result(data)
