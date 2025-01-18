import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# 从 CSV 文件加载数据
X_train = pd.read_csv('data/breast_cancer_Xtrain.csv', header=0).values
X_test = pd.read_csv('data/breast_cancer_Xtest.csv', header=0).values
y_train = pd.read_csv('data/breast_cancer_Ytrain.csv', header=0).values.reshape(-1)
y_test = pd.read_csv('data/breast_cancer_Ytest.csv', header=0).values.reshape(-1)

# 线性SVM：C=1
model_linear_C1 = SVC(kernel='linear', C=1)
model_linear_C1.fit(X_train, y_train)

# 线性SVM：C=1000
model_linear_C1000 = SVC(kernel='linear', C=1000)
model_linear_C1000.fit(X_train, y_train)

# 多项式SVM：C=1, d=2
model_poly_C1 = SVC(kernel='poly', degree=2, C=1)
model_poly_C1.fit(X_train, y_train)

# 多项式SVM：C=1000, d=2
model_poly_C1000 = SVC(kernel='poly', degree=2, C=1000)
model_poly_C1000.fit(X_train, y_train)

# 测试并输出准确率, 精确率, 召回率 和 F1 分数
models = [model_linear_C1, model_linear_C1000, model_poly_C1, model_poly_C1000]
model_names = ['Linear SVM C=1', 'Linear SVM C=1000', 'Polynomial SVM C=1 d=2', 'Polynomial SVM C=1000 d=2']

for model, name in zip(models, model_names):
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    print(f"{name}:")
    print(f"  Accuracy = {accuracy:.2f}")
    print(f"  Precision = {precision:.2f}")
    print(f"  Recall = {recall:.2f}")
    print(f"  F1 Score = {f1:.2f}")
