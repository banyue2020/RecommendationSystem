from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import numpy as np

# === 第一步：读取数据 ===
# 读取电影信息文件
file_path_movies = '../Data/ml-1m/movies.dat'
movies_columns = ['movieId', 'title', 'genres']
movies_df = pd.read_csv(file_path_movies, delimiter='::', names=movies_columns, engine='python', encoding='ISO-8859-1')

# 读取评分信息文件
file_path_ratings = '../Data/ml-1m/ratings.dat'
ratings_columns = ['userId', 'movieId', 'rating', 'timestamp']
ratings_df = pd.read_csv(file_path_ratings, delimiter='::', names=ratings_columns, engine='python', encoding='ISO-8859-1')

# 电影的年份提取到新列
movies_df['Year'] = movies_df['title'].str.extract(r'(\\d{4})')

# 去除标题中的年份信息并去除多余空格
movies_df['title'] = movies_df['title'].str.replace(r'\\(\\d{4}\\)', '').str.strip()

# 将电影题材字符串转换成单一字符串（每个题材作为一个词）
movies_df['genres'] = movies_df['genres'].apply(lambda x: x.replace('|', ' '))

# === 第二步：特征提取 ===
# 初始化TFIDF向量器，使用电影题材作为特征
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(movies_df['genres'])

# === 第三步：用户画像创建 ===
# 使用平均加权评分构建用户画像
def create_user_profile(userId, ratings_df, tfidf_matrix):
    # 获取用户的所有评分
    user_ratings = ratings_df[ratings_df['userId'] == userId]
    # 获取用户评分的电影索引
    movies_idx = user_ratings['movieId'].apply(lambda x: movies_df[movies_df['movieId'] == x].index.values)
    if not movies_idx.empty:
        movies_idx = np.hstack(movies_idx.values) # Flatten list of lists
    # 获取用户评分的TFIDF向量
    user_tfidf = tfidf_matrix[movies_idx.astype(int)]
    # 计算加权平均用户画像
    user_profile = np.dot(user_ratings['rating'].values, user_tfidf.toarray())
    user_profile_norm = user_profile / user_ratings['rating'].sum()
    return user_profile_norm

# 示例用户
some_user_id = 1
user_profile = create_user_profile(some_user_id, ratings_df, tfidf_matrix)

# === 第四步：相似度计算 ===
user_profile = np.array(user_profile).reshape(1, -1)
cosine_similarities = cosine_similarity(user_profile, tfidf_matrix)

# === 第五步：生成推荐 ===
similarities_scores = cosine_similarities.flatten()
top_N_indices = similarities_scores.argsort()[::-1]

# === 第六步：输出推荐电影及其预测评分和用户ID ===
# 获取所有电影的预测评分（这里我们用相似度分数作为预测评分的估算）
predicted_scores = cosine_similarities.flatten()

# 生成包含预测评分的DataFrame
recommendations_df = movies_df.copy()
recommendations_df['predicted_rating'] = predicted_scores*5

# 获取用户已评分的电影ID
rated_movie_ids = ratings_df[ratings_df['userId'] == some_user_id]['movieId'].tolist()

# 剔除用户已评分的电影
recommendations_df = recommendations_df[~recommendations_df['movieId'].isin(rated_movie_ids)]

# 按预测评分降序排序并获取前5部电影
top_n_recommendations = recommendations_df.nlargest(5, 'predicted_rating')

# 打印推荐
for index, row in top_n_recommendations.iterrows():
    print("用户编号：{}，电影编号：{}，推荐电影：'{}'，预测用户可能评分：{:.2f}".format(
        some_user_id, row['movieId'], row['title'], row['predicted_rating']))