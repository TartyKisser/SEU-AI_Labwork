import pandas as pd
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer, accuracy_score, precision_score, recall_score, f1_score

# Load data
X_train = pd.read_csv('data/breast_cancer_Xtrain.csv').values
y_train = pd.read_csv('data/breast_cancer_Ytrain.csv').values.ravel()
X_test = pd.read_csv('data/breast_cancer_Xtest.csv').values
y_test = pd.read_csv('data/breast_cancer_Ytest.csv').values.ravel()

# Define parameter grid
list_grid = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
param_grid = [
    {'kernel': ['linear'], 'C': list_grid},
    {'kernel': ['poly'], 'C': list_grid, 'degree': [3], 'coef0': [0], 'gamma': ['scale']},
    {'kernel': ['rbf'], 'C': list_grid, 'gamma': ['scale']},
    {'kernel': ['sigmoid'], 'C': list_grid, 'gamma': ['scale'], 'coef0': [0]}
]

# Custom scorers
scoring = {
    'accuracy': make_scorer(accuracy_score),
    'precision': make_scorer(precision_score, zero_division=0),
    'recall': make_scorer(recall_score, zero_division=0),
    'f1': make_scorer(f1_score, zero_division=0)
}

# Create GridSearchCV object
svc = SVC()
clf = GridSearchCV(svc, param_grid, cv=5, scoring=scoring, refit='accuracy', return_train_score=False)
clf.fit(X_train, y_train)

# Prepare results dictionary
results = {kernel: {'C': [], 'accuracy': [], 'precision': [], 'recall': [], 'f1': []}
           for kernel in ['linear', 'poly', 'rbf', 'sigmoid']}

# Fill results dictionary
for i, params in enumerate(clf.cv_results_['params']):
    kernel = params['kernel']
    C = params['C']
    if C not in results[kernel]['C']:  # Ensure each C is added once per kernel
        results[kernel]['C'].append(C)
        # Append metric scores
        for metric in ['accuracy', 'precision', 'recall', 'f1']:
            metric_key = f'mean_test_{metric}'
            results[kernel][metric].append(clf.cv_results_[metric_key][i])

# Plotting
fig, axs = plt.subplots(2, 2, figsize=(8, 6))
axs = axs.ravel()
metrics = ['accuracy', 'precision', 'recall', 'f1']
for i, metric in enumerate(metrics):
    for kernel in ['sigmoid','linear', 'poly', 'rbf']:
        axs[i].plot(results[kernel]['C'], results[kernel][metric], label=f'{kernel} kernel')
        axs[i].set_xscale('log')
        axs[i].set_title(f'{metric.title()} ')
        axs[i].set_xlabel('C (log scale)')
        axs[i].set_ylabel(metric.title())
        axs[i].legend()

plt.tight_layout()
plt.show()
