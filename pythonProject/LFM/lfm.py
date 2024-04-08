import numpy as np
import pandas as pd

# === 加载数据 ===
# 用户评分文件
ratings_columns = ['userId', 'movieId', 'rating', 'timestamp']
ratings = pd.read_csv('../Data/ml-1m/ratings.dat', delimiter='::', names=ratings_columns, engine='python')

# 电影信息文件
movies_columns = ['movieId', 'title', 'genres']
movies = pd.read_csv('../Data/ml-1m/movies.dat', delimiter='::', names=movies_columns, engine='python', encoding='ISO-8859-1')


# === LFM算法实现 ===
# 用户的评分数据、隐因子的数量、算法的迭代次数、学习率、正则化项的系数
def lfm(ratings, n_factors=10, n_iterations=10, learning_rate=0.001, lmbda=0.05):
    user_ids = ratings['userId'].unique()
    item_ids = ratings['movieId'].unique()

    # 随机初始化用户和物品的隐特征矩阵
    print("初始化用户和物品隐特征矩阵...")
    user_factors = np.random.normal(0, 0.1, (len(user_ids), n_factors))
    item_factors = np.random.normal(0, 0.1, (len(item_ids), n_factors))

    # 用户ID和物品ID到矩阵索引的映射
    user_id_to_index = {user_id: index for index, user_id in enumerate(user_ids)}
    item_id_to_index = {item_id: index for index, item_id in enumerate(item_ids)}
    print("映射表构建完成.")

    # 随机梯度下降来优化隐特征矩阵
    print("开始随机梯度下降优化过程...")
    for iteration in range(n_iterations):
        # print(f"迭代次数 {iteration + 1}/{n_iterations}")
        for row in ratings.itertuples():
            user_index = user_id_to_index[row.userId]
            item_index = item_id_to_index[row.movieId]
            error = row.rating - np.dot(user_factors[user_index], item_factors[item_index])

            # 检查是否存在无效数值
            if np.isnan(error) or np.isinf(error):
                raise ValueError("数值稳定性问题：无效的错误计算")

            # 更新隐特征矩阵
            user_factors[user_index] += learning_rate * (
                    error * item_factors[item_index] - lmbda * user_factors[user_index])
            item_factors[item_index] += learning_rate * (
                    error * user_factors[user_index] - lmbda * item_factors[item_index])

            # 添加额外的数值稳定性检查
            if np.any(np.isnan(user_factors[user_index])) or np.any(np.isinf(user_factors[user_index])):
                raise ValueError("数值稳定性问题：无效的用户因子更新")
            if np.any(np.isnan(item_factors[item_index])) or np.any(np.isinf(item_factors[item_index])):
                raise ValueError("数值稳定性问题：无效的物品因子更新")

        print(f"完成迭代次数 {iteration + 1}/{n_iterations}.")

    print("所有迭代完成，模型训练结束。")

    return user_factors, item_factors, user_id_to_index, item_id_to_index

# === 使用LFM模型进行评分预测 ===
def predict_rating(user_factors, item_factors, user_id_to_index, item_id_to_index, user_id, item_id):
    user_index = user_id_to_index[user_id]
    item_index = item_id_to_index[item_id]
    return np.dot(user_factors[user_index], item_factors[item_index])

# 调用LFM函数
print("开始LFM模型训练...")
user_factors, item_factors, user_id_to_index, item_id_to_index = lfm(ratings)

def recommend_top_movies(user_factors, item_factors, user_id_to_index, item_id_to_index, user_id, n_recommendations=5):
    # 检查用户是否在训练集中
    if user_id not in user_id_to_index:
        raise ValueError(f"用户ID {user_id} 不在训练集中。")

    user_index = user_id_to_index[user_id]
    user_ratings = ratings[ratings['userId'] == user_id]
    rated_movie_ids = set(user_ratings['movieId'].tolist())

    # 存储电影的ID和对应的预测评分
    movie_ratings = []

    # 遍历所有电影，计算未评分电影的预测评分
    for movie_id in item_id_to_index:
        if movie_id not in rated_movie_ids:
            item_index = item_id_to_index[movie_id]
            score = np.dot(user_factors[user_index], item_factors[item_index])
            movie_ratings.append((movie_id, score))

    # 对预测评分排序并获取最高的n_recommendations部电影
    movie_ratings.sort(key=lambda x: x[1], reverse=True)
    top_movie_ids_ratings = movie_ratings[:n_recommendations]

    # 获得电影编号和标题
    top_movies_info = [(movie_id, movies[movies['movieId'] == movie_id]['title'].values[0], rating) for movie_id, rating in top_movie_ids_ratings]

    return top_movies_info

# 用户ID
user_id = 1

# 获取推荐
top_movies_info = recommend_top_movies(user_factors, item_factors, user_id_to_index, item_id_to_index, user_id)
print(f"用户 {user_id} 最可能喜欢的五部电影是：")
for movie_id, movie, rating in top_movies_info:
    print(f"{movie_id} - {movie}，预测评分：{rating:.2f}")