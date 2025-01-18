import numpy as np
import copy
import pandas as pd
import random
from sklearn.metrics import accuracy_score


class SVM:
    def __init__(self, C=1.0, kernel='linear', toler=0.001, max_iter=50):
        self.C = C
        self.kernel = kernel
        self.toler = toler
        self.max_iter = max_iter
        self.alphas = None
        self.b = 0
        self.w = None

    def fit(self, X, y):
        self.alphas, self.b = self.smo(X, y, self.C, self.toler, self.max_iter)

        # Weight vector for linear SVM
        if self.kernel == 'linear':
            self.w = np.dot(self.alphas * y, X)
        else:
            self.w = None  # Non-linear kernel

    def predict(self, X):
        if self.kernel == 'linear':
            return np.sign(np.dot(X, self.w) + self.b)
        else:
            # Implement kernelized prediction for non-linear kernels
            raise NotImplementedError("Non-linear kernel prediction not implemented.")

    def smo(self, X, y, C, tol, max_iter):
        m, n = X.shape
        alphas = np.zeros(m)
        b = 0
        passes = 0

        while passes < max_iter:
            num_changed_alphas = 0
            for i in range(m):
                Ei = self.calculate_Ei(X, y, alphas, b, i)
                if (y[i] * Ei < -tol and alphas[i] < C) or (y[i] * Ei > tol and alphas[i] > 0):
                    # Randomly select j != i
                    j = np.random.choice(list(set(range(m)) - {i}))
                    Ej = self.calculate_Ei(X, y, alphas, b, j)

                    # Save old alphas
                    alpha_i_old = alphas[i]
                    alpha_j_old = alphas[j]

                    # Compute L and H
                    if y[i] != y[j]:
                        L = max(0, alphas[j] - alphas[i])
                        H = min(C, C + alphas[j] - alphas[i])
                    else:
                        L = max(0, alphas[i] + alphas[j] - C)
                        H = min(C, alphas[i] + alphas[j])

                    if L == H:
                        continue

                    # Compute eta
                    eta = 2.0 * np.dot(X[i], X[j]) - np.dot(X[i], X[i]) - np.dot(X[j], X[j])
                    if eta >= 0:
                        continue

                    # Update alpha_j
                    alphas[j] -= (y[j] * (Ei - Ej)) / eta
                    alphas[j] = min(H, alphas[j])
                    alphas[j] = max(L, alphas[j])

                    if abs(alphas[j] - alpha_j_old) < 0.00001:
                        continue

                    # Update alpha_i
                    alphas[i] += y[i] * y[j] * (alpha_j_old - alphas[j])

                    # Compute b
                    b1 = b - Ei - y[i] * (alphas[i] - alpha_i_old) * np.dot(X[i], X[i]) - y[j] * (
                                alphas[j] - alpha_j_old) * np.dot(X[i], X[j])
                    b2 = b - Ej - y[i] * (alphas[i] - alpha_i_old) * np.dot(X[i], X[j]) - y[j] * (
                                alphas[j] - alpha_j_old) * np.dot(X[j], X[j])

                    if 0 < alphas[i] < C:
                        b = b1
                    elif 0 < alphas[j] < C:
                        b = b2
                    else:
                        b = (b1 + b2) / 2.0

                    num_changed_alphas += 1

            if num_changed_alphas == 0:
                passes += 1
            else:
                passes = 0

        return alphas, b

    def calculate_Ei(self, X, y, alphas, b, i):
        # 计算预测值 f(xi) = sum(alpha_j * y_j * x_j^T * x_i) + b
        # 注意这里是使用矩阵向量乘法，将计算结果累加
        f_xi = np.dot((alphas * y), np.dot(X, X[i])) + b
        # Ei = f(xi) - yi
        Ei = f_xi - y[i]
        return Ei

    def check_KKT(self, X, y, alphas, b):
        violations = []
        for i in range(len(y)):
            yi_fxi = y[i] * (np.dot(X[i], self.w) + b)
            if alphas[i] == 0 and yi_fxi < 1 - self.toler:
                violations.append((i, 'yi*fxi < 1 - toler, alpha == 0'))
            elif 0 < alphas[i] < self.C and not (1 - self.toler <= yi_fxi <= 1 + self.toler):
                violations.append((i, '1 - toler <= yi*fxi <= 1 + toler not met, 0 < alpha < C'))
            elif alphas[i] == self.C and yi_fxi > 1 + self.toler:
                violations.append((i, 'yi*fxi > 1 + toler, alpha == C'))
        return violations


X_train = pd.read_csv('data/breast_cancer_Xtrain.csv', header=0).values
X_test = pd.read_csv('data/breast_cancer_Xtest.csv', header=0).values
y_train = pd.read_csv('data/breast_cancer_Ytrain.csv', header=0).values.reshape(-1)
y_test = pd.read_csv('data/breast_cancer_Ytest.csv', header=0).values.reshape(-1)

classifier = SVM(C=2)

classifier.fit(X_train, y_train)

violations = classifier.check_KKT(X_train, y_train, classifier.alphas, classifier.b)

print("KKT条件违反的样本及其情况：")
for idx, reason in violations:
    print(f"样本 {idx}: {reason}")


y_predict = classifier.predict(X_test)

accuracy = accuracy_score(y_test, y_predict)
print("准确率：", accuracy)