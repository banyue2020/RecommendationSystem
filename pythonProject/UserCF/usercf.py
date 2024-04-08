import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import csr_matrix

# === 第一步：读取数据 ===
# 读取用户评分文件
file_path_ratings = '../Data/ml-1m/ratings.dat'
ratings_columns = ['userId', 'movieId', 'rating', 'timestamp']
ratings = pd.read_csv(file_path_ratings, delimiter='::', names=ratings_columns, engine='python')

# 可能还需要读取电影信息文件
file_path_movies = '../Data/ml-1m/movies.dat'
movies_columns = ['movieId', 'title', 'genres']
movies = pd.read_csv(file_path_movies, delimiter='::', names=movies_columns, engine='python', encoding='ISO-8859-1')

# === 第二步：预处理数据 ===
# 创建评分矩阵，行表示用户，列表示电影
ratings_pivot = ratings.pivot(index='userId', columns='movieId', values='rating').fillna(0)

# 转换为稀疏矩阵格式以进行更有效的计算
ratings_matrix = csr_matrix(ratings_pivot.values)

# === 第三步：计算用户间的相似度 ===
# 使用余弦相似度计算用户间的相似性
user_similarity = cosine_similarity(ratings_matrix)

# === 第四步：生成推荐 ===
def recommend_movies(user_similarity_matrix, user_id, num_recommendations=5, num_neighbors=50):
    # 获取用户索引
    user_idx = user_id - 1  # UserID starts at 1 in the dataset
    # 获取用户的相似度向量
    similarity_scores = list(enumerate(user_similarity_matrix[user_idx]))
    # 按相似度分数降序排列相似度分数列表
    similarity_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)

    # 选取最相似的num_neighbors个用户
    most_similar_users = [tup[0] for tup in similarity_scores[1:num_neighbors + 1]]  # 排除自己
    similarity_weights = [tup[1] for tup in similarity_scores[1:num_neighbors + 1]]

    # 创建空的Series来存储加权评分
    movie_ratings_weighted = pd.Series(0, index=ratings_pivot.columns)
    # 累计邻居的加权评分
    for i, user in enumerate(most_similar_users):
        movie_ratings_weighted += ratings_pivot.iloc[user] * similarity_weights[i]
    # 计算加权平均评分
    final_movie_ratings = movie_ratings_weighted / sum(similarity_weights)

    # 排除已经评分过的电影
    movies_unseen = final_movie_ratings[ratings_pivot.iloc[user_idx] == 0]
    # 获取推荐电影
    recommended_movies = movies_unseen.sort_values(ascending=False).head(num_recommendations)

    # 将用户ID以及电影ID转换为电影名称，并为预测评分创建一个包含用户ID、电影ID、电影名称和评分的列表
    recommended_movies_info = [
        (user_id, movie_id, movies.loc[movies['movieId'] == movie_id, 'title'].iloc[0], final_movie_ratings[movie_id])
        for movie_id in recommended_movies.index
    ]
    return recommended_movies_info

# 使用示例
recommended_movies_info = recommend_movies(user_similarity, user_id=1, num_neighbors=50)
for user_id, movie_id, movie_title, predicted_rating in recommended_movies_info:
    print(f"用户编号：{user_id}，电影编号：{movie_id}，推荐电影：'{movie_title}'，预测用户可能评分：{predicted_rating:.2f}")