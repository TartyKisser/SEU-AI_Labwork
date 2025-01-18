import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

# 加载数据集
df = pd.read_csv("song_data.csv")

# df是原始的DataFrame
# 离散特征列
discrete_columns = ['key', 'audio_mode', 'time_signature']

# 连续特征列
continuous_columns = [col for col in df.columns if col not in discrete_columns + ['song_name', 'song_popularity']]

# One-Hot编码离散特征

# 初始化OneHotEncoder
encoder = OneHotEncoder(sparse=False)

# 对离散特征列进行独热编码
encoded_data = encoder.fit_transform(df[discrete_columns])

# 将独热编码的数据转换为DataFrame
encoded_df = pd.DataFrame(encoded_data, columns=encoder.get_feature_names_out())

# 选择连续特征进行归一化
scaler = MinMaxScaler()
df[continuous_columns] = scaler.fit_transform(df[continuous_columns])

"""
vif = pd.DataFrame()
vif["variables"] = features_after_encoding
vif["VIF"] = [variance_inflation_factor(df_merged.values, i) for i in range(df_merged.values.shape[1])]
print("vif: ",vif)
"""

# X为输入属性，y为输出属性
X = df[continuous_columns].values
y = df['song_popularity'].values

# 使用PCA进行特征降维
pca = PCA(n_components=8)
X_pca = pca.fit_transform(X)

# 输出每个主成分的贡献率
print("输出每个主成分的贡献率： ", pca.explained_variance_ratio_)

# 输出降维后的属性矩阵
print("降维后的属性矩阵： ", X_pca.shape)

# 合并离散特征和连续特征
X_pca = pd.DataFrame(X_pca)
X_pca.columns = X_pca.columns.astype(str)
df_merged = pd.concat([X_pca, encoded_df], axis=1)

# 划分数据集
X_train, X_test, y_train, y_test = train_test_split(df_merged, y, test_size=0.2, random_state=42)

# 构建线性回归模型
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)

# 预测训练集和测试集的结果
y_train_predict = lr_model.predict(X_train)
y_test_predict = lr_model.predict(X_test)

# 计算训练误差和测试误差
train_error = mean_squared_error(y_train, y_train_predict)
test_error = mean_squared_error(y_test, y_test_predict)

# 输出训练误差和测试误差
print("训练误差", train_error)
print("测试误差", test_error)

# 将真实值和预测值转换为DataFrame
y_test_df = pd.DataFrame(y_test, columns=['true_target'])
y_test_predict_df = pd.DataFrame(y_test_predict, columns=['predicted_target'])

# 合并两个DataFrame
merged_df = pd.concat([y_test_df, y_test_predict_df], axis=1)

# 导出到CSV文件
merged_df.to_csv('y_test_and_predict.csv', index=False)

# 取测试集的真实值和预测值的前100个样本进行可视化
plt.figure(figsize=(10, 6))
plt.plot(y_test[:100], label='True Popularity', color='blue', marker='o')
plt.plot(y_test_predict[:100], label='Predicted Popularity', color='red', linestyle='--', marker='x')
plt.title('Song Popularity: True vs. Predicted')
plt.xlabel('Sample Index')
plt.ylabel('Popularity')
plt.legend()
plt.grid(True)
plt.show()
plt.savefig("result.png")
