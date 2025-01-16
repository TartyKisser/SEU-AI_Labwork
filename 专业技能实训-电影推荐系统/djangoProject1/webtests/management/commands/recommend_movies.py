import pandas as pd
from django.core.management.base import BaseCommand
from django.core.cache import cache
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import OneHotEncoder
import numpy as np

class Command(BaseCommand):
    help = '获取推荐给用户的电影'

    def handle(self, *args, **options):
        # 计算用户对电影的预测评分
        def predict_ratings(user_id, similarity_matrix, ratings_matrix):
            # 获取目标用户与所有用户的相似度
            # user_similarities = similarity_matrix.loc[user_id]
            # 初始化预测评分
            predicted_ratings = pd.Series(0.0, index=ratings_matrix.columns)
            # 计算每个电影的预测评分
            try:
                # 获取目标用户已经评分的电影
                rated_movies = ratings_matrix.loc[user_id] > 0
                user_similarities = similarity_matrix.loc[user_id]
            except:
                return predicted_ratings
            for movie_id in predicted_ratings.index:
                if rated_movies[movie_id]:
                    predicted_ratings[movie_id] = 0
                else:
                    # 获取所有用户对该电影的评分
                    movie_ratings = ratings_matrix[movie_id]
                    # 计算加权平均评分
                    predicted_ratings[movie_id] = (user_similarities * movie_ratings).sum()
            return predicted_ratings

        # 将基于物品的推荐算法和基于用户的协同过滤推荐算法结合，可控制两种算法所占权重，解决了冷启动问题
        def recommend_movie(predicted_ratings, similarity_series, num_recommendations=30):
            # 获取最高分的电影索引
            for movie_id in similarity_series.index:
                # 通过调整参数，调整两种算法所占权重
                predicted_ratings[movie_id] += 5 * similarity_series[movie_id]
            movie_indices = predicted_ratings.sort_values(ascending=False)
            top_recommendations = movie_indices.head(num_recommendations).index.tolist()
            return top_recommendations

        # 实现推荐算法并将得到的推荐电影保存到csv文件中
        def recommend_movies(user_id, user_liked_genres):
            # 加载数据
            movies_df = pd.read_csv('webtests/management/commands/douban_movies.csv')
            users_df = pd.read_csv('webtests/management/commands/douban_users.csv')
            # 获取全部类型
            movies_df['all_genres'] = movies_df[['style1', 'style2', 'style3']].apply(lambda x: '|'.join(x.dropna()),
                                                                                      axis=1)
            all_genres = movies_df['all_genres'].str.split('|').explode().unique()
            all_genres = all_genres[all_genres != '']

            def encode_movie_genres(movie_genres):
                # 将电影类型独热编码
                if pd.isna(movie_genres) or movie_genres == '':
                    return np.zeros(len(all_genres))
                else:
                    genres_list = movie_genres.split('|')
                    encoded_vector = np.zeros(len(all_genres))
                    for genre in genres_list:
                        # 使用 np.where 来找到类型在 encoder.categories 中的索引
                        genre_idx = np.where(encoder.categories_[0] == genre)[0]
                        if genre_idx.size > 0:
                            index = genre_idx[0]  # 获取第一个匹配的索引
                            encoded_vector[index] = 1
                    return encoded_vector

            def encode_user_genres(user_liked_genres):
                # 将用户喜欢的电影类型独热编码
                encoded_vector = np.zeros(len(all_genres))
                for genre in user_liked_genres:
                    # 使用 np.where 来找到类型在 encoder.categories 中的索引
                    genre_idx = np.where(encoder.categories_[0] == genre)[0]
                    if genre_idx.size > 0:
                        index = genre_idx[0]  # 获取第一个匹配的索引
                        encoded_vector[index] = 1
                return encoded_vector

            # 初始化 OneHotEncoder
            encoder = OneHotEncoder(sparse_output=False)
            # 对所有唯一的类型进行独热编码
            encoded_genres = encoder.fit_transform(all_genres.reshape(-1, 1))
            # 应用函数，为每部电影生成独热编码
            movies_df['encoded_genres'] = movies_df['all_genres'].apply(encode_movie_genres)
            # 将独热编码向量转换为 DataFrame，并为列命名
            encoded_df = pd.DataFrame(movies_df['encoded_genres'].tolist(),
                                      columns=[f'genre_{i}' for i in range(len(all_genres))])
            #  DataFrame 合并到原始的 movies_df DataFrame 中
            movies_df = pd.concat([movies_df, encoded_df], axis=1)

            # 构造每部电影的特征向量
            feature_matrix_df = movies_df.drop(
                ['Unnamed: 0', 'encoded_genres', 'name', 'english_name', 'directors', 'writer', 'actors', 'rate',
                 'style1',
                 'style2',
                 'style3', 'country', 'language', 'date', 'duration', 'introduction', 'url',
                 'pic', 'all_genres'], axis=1)
            feature_matrix_df = feature_matrix_df.dropna()
            feature_matrix_df1 = feature_matrix_df.drop(['dataID'], axis=1)
            feature_matrix = feature_matrix_df1.values

            # 构造用户特征向量
            user_vector = encode_user_genres(user_liked_genres)
            user_movie_similarity = cosine_similarity(user_vector.reshape(1, -1), feature_matrix).flatten()
            # 基于物品的用户电影相似度分数，解决了冷启动问题
            similarity_series = pd.Series(user_movie_similarity, index=feature_matrix_df['dataID'])

            # 基于用户的协同过滤

            # 去除重复的评分记录
            unique_ratings = users_df.drop_duplicates(subset=['user_id', 'movie_id'])
            # 构建评分矩阵
            ratings_matrix = unique_ratings.pivot(index='user_id', columns='movie_id', values='rating').fillna(0)
            # 计算用户之间的相似度
            user_similarity = cosine_similarity(ratings_matrix)
            # 将相似度矩阵转换回 DataFrame 方便使用
            user_similarity_df = pd.DataFrame(user_similarity, index=ratings_matrix.index, columns=ratings_matrix.index)
            # 计算用户对电影的预测评分
            predicted_ratings = predict_ratings(user_id, user_similarity_df, ratings_matrix)
            # 将两种推荐算法复合，得出推荐的电影
            recommended_movies = recommend_movie(predicted_ratings, similarity_series, 30)
            recommended_movies = movies_df[movies_df['dataID'].isin(recommended_movies)]
            # 将推荐电影保存到csv文件中
            item = ['name', 'english_name', 'directors', 'writer', 'actors', 'rate', 'style1', 'style2',
                    'style3', 'country', 'language', 'date', 'duration', 'introduction', 'dataID', 'url',
                    'pic']  # 列索引
            recommended_movies = recommended_movies[item]
            MOVIES = pd.DataFrame(data=recommended_movies, columns=item)  # 转换为DataFrame数据格式
            MOVIES.to_csv('webtests/management/commands/recommended_movies.csv', mode='w', encoding='utf-8-sig')  # 存入csv文件

        user_name = cache.get('current_user')
        import csv
        def find_genres_by_name(filename, target_name):
            genres = []
            with open(filename, 'r', encoding='utf-8') as file:
                csv_reader = csv.reader(file)
                for row in csv_reader:
                    if len(row) >= 2 and row[0] == target_name:
                        genres = row[1].split()  # 使用空格分割词语
                        break  # 找到目标行后退出循环
            return genres
        # 使用函数
        filename = 'webtests/management/commands/users_like_adjust.csv'  # 替换为你的CSV文件名
        target_name = user_name  # 替换为你要查找的特定名字
        result = find_genres_by_name(filename, target_name)
        user_liked_genres = result
        recommend_movies(user_name, user_liked_genres)