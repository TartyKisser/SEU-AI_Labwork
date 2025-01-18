import numpy as np
import pandas as pd
from cvxopt import matrix, solvers

import numpy as np
from cvxopt import matrix, solvers


def train_hard_margin_svm(x_train, y_train):
    # 训练数据的样本数和维度
    n_samples, n_features = x_train.shape

    # 计算矩阵 P，其中 P_ij = y_i * y_j * x_i^T * x_j
    K = np.dot(x_train, x_train.T)  # 核函数为线性核: x_i^T * x_j
    Y = y_train.reshape(-1, 1)
    P = matrix(np.outer(y_train, y_train) * K, tc='d')  # 确保类型为双精度

    # 向量 q, 维度为 n_samples, 所有元素为 -1
    q = matrix(-np.ones(n_samples), tc='d')

    # 不等式约束 Gx <= h，这里 G 是一个 n_samples x n_samples 的负单位矩阵，h 是一个零向量
    G = matrix(-np.eye(n_samples), tc='d')
    h = matrix(np.zeros(n_samples), tc='d')

    # 等式约束 Ax = b, 这里 A 是 y_train 的转置，b 是 0
    A = matrix(y_train.reshape(1, -1).astype(np.double), tc='d')
    b = matrix(np.zeros(1), tc='d')

    # 调用 cvxopt 的二次规划求解器
    sol = solvers.qp(P, q, G, h, A, b)

    # 拉格朗日乘子
    alphas = np.array(sol['x']).flatten()

    return alphas


def predict(x_test, x_train, y_train, alphas):
    # 选取非零 alpha 对应的支持向量
    support_vector_indices = alphas > 1e-5
    support_vector_alphas = alphas[support_vector_indices]
    support_vector_x = x_train[support_vector_indices]
    support_vector_y = y_train[support_vector_indices]

    # 计算权重 w
    # 确保 alpha 和 y 的维度相匹配，并与 x 的维度兼容
    w = np.sum((support_vector_alphas * support_vector_y)[:, np.newaxis] * support_vector_x, axis=0)

    # 计算截距 b
    # 使用支持向量进行 b 的计算，确保使用 dot 时维度正确
    b = np.mean(support_vector_y - np.dot(support_vector_x, w))

    # 对测试集进行预测
    y_predict = np.dot(x_test, w) + b
    y_predict = np.sign(y_predict)  # 将结果转换为 +1 或 -1

    return y_predict


# 测试数据
x_train = pd.read_csv("data/breast_cancer_Xtrain.csv", header=None).values
y_train = pd.read_csv("data/breast_cancer_Ytrain.csv", header=None).values
y_train=y_train.reshape(-1)
x_test = pd.read_csv("data/breast_cancer_Xtest.csv", header=None).values
y_test = pd.read_csv("data/breast_cancer_Ytest.csv", header=None).values
y_test=y_test.reshape(-1)
# 训练 SVM
alphas = train_hard_margin_svm(x_train, y_train)
print("拉格朗日乘子:", alphas)

y_predict = predict(x_test=x_test,x_train=x_train,y_train=y_train,alphas=alphas)
accuracy = np.sum(np.equal(y_predict, y_test))/len(y_test)
print("accuracy: ", accuracy)
