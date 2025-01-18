import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# 从 CSV 文件加载数据
X_train = pd.read_csv('data/breast_cancer_Xtrain.csv', header=0).values
X_test = pd.read_csv('data/breast_cancer_Xtest.csv', header=0).values
y_train = pd.read_csv('data/breast_cancer_Ytrain.csv', header=0).values.reshape(-1)
y_test = pd.read_csv('data/breast_cancer_Ytest.csv', header=0).values.reshape(-1)

# 定义 SVM 模型和参数网格
model = SVC(kernel='rbf', C=1)
param_grid = {'gamma': np.logspace(-4, 2, 7)}

# 使用 GridSearchCV 进行参数搜索
grid_search = GridSearchCV(model, param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train, y_train)

# 获取最佳参数和对应的模型
best_gamma = grid_search.best_params_['gamma']
best_model = grid_search.best_estimator_

# 在测试集上评估最佳模型
y_pred = best_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print(f"Best gamma: {best_gamma}")
print("Best model performance on test set:")
print(f"  Accuracy = {accuracy:.2f}")
print(f"  Precision = {precision:.2f}")
print(f"  Recall = {recall:.2f}")
print(f"  F1 Score = {f1:.2f}")
