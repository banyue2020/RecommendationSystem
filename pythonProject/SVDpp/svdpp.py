import numpy as np
import pandas as pd

# === 第一步：读取数据 ===
movies_columns = ['movieId', 'title', 'genres']
movies_df = pd.read_csv('../Data/ml-1m/movies.dat', delimiter='::', names=movies_columns, engine='python', encoding='ISO-8859-1')

ratings_columns = ['userId', 'movieId', 'rating', 'timestamp']
ratings_df = pd.read_csv('../Data/ml-1m/ratings.dat', delimiter='::', names=ratings_columns, engine='python', encoding='ISO-8859-1')

users_columns = ['userId', 'gender', 'age', 'occupation', 'zip_code']
users_df = pd.read_csv('../Data/ml-1m/users.dat', delimiter='::', names=users_columns, engine='python', encoding='ISO-8859-1')

# 计算全局平均评分
mu = ratings_df['rating'].mean()

# 初始化参数
M, N = len(np.unique(ratings_df['userId'])), len(np.unique(ratings_df['movieId']))
K = 10  # 隐含因子数量，可以调整


def init_matrix(M, N, K):
    np.random.seed(0)
    return np.random.rand(M, K), np.random.rand(N, K)

print("开始初始化参数...")
P, Q = init_matrix(M, N, K)
bu = np.zeros(M)
bi = np.zeros(N)

# 新建DataFrame，用来做用户ID和物品ID的映射
user_info = pd.DataFrame(np.unique(ratings_df['userId']), columns=['raw_uid'])
item_info = pd.DataFrame(np.unique(ratings_df['movieId']), columns=['raw_iid'])
# 创建字典，用于映射原始ID到内部索引
print("开始创建字典映射...")
user_dict = user_info.reset_index().set_index('raw_uid').to_dict()['index']
item_dict = item_info.reset_index().set_index('raw_iid').to_dict()['index']


def train(ratings, P, Q, bu, bi, mu, K, steps=20, gamma=0.04, lambda_=0.15):
    # 随机梯度下降法训练模型参数
    for step in range(steps):
        print(f"开始训练第 {step + 1} 轮...")
        for uid, iid, r_ui in ratings_df[['userId', 'movieId', 'rating']].values:
            # 获取用户和物品的内部索引
            uid = user_dict[uid]
            iid = item_dict[iid]

            # 计算预测评分的误差
            pred_r_ui = mu + bu[uid] + bi[iid] + np.dot(P[uid], Q[iid].T)
            e_ui = r_ui - pred_r_ui

            # 更新偏置项和隐含因子矩阵
            bu[uid] += gamma * (e_ui - lambda_ * bu[uid])
            bi[iid] += gamma * (e_ui - lambda_ * bi[iid])
            P[uid, :] += gamma * (e_ui * Q[iid, :] - lambda_ * P[uid, :])
            Q[iid, :] += gamma * (e_ui * P[uid, :] - lambda_ * Q[iid, :])

        gamma *= 0.9  # 学习率递减

    return bu, bi, P, Q

# 训练模型
print("开始训练模型...")
bu, bi, P, Q = train(ratings_df[['userId', 'movieId', 'rating']].values, P, Q, bu, bi, mu, K)

def top_n_recommendations(user_id, n=5):
    # 获取用户和所有电影的内部索引
    uid = user_dict[user_id]
    iid_list = item_dict.values()

    # 计算用户对所有电影的评分
    rating_list = [mu + bu[uid] + bi[iid] + np.dot(P[uid, :], Q[iid, :].T) for iid in iid_list]

    # 生成评分的数据框
    rating_df = pd.DataFrame(data={'movieId': list(item_dict.keys()), 'pred_rating': rating_list})

    # 和电影信息进行合并
    result_df = pd.merge(rating_df, movies_df, on='movieId')

    # 根据预测评分排序并取出前n部
    top_n_movies = result_df.sort_values(by='pred_rating', ascending=False).iloc[:n, :]

    return top_n_movies

user_id = 1
top_n_movies = top_n_recommendations(user_id, 5)
for _, row in top_n_movies.iterrows():
    movie_id = row['movieId']
    movie_name = row['title']
    pred_rating = row['pred_rating']
    print(f"用户编号：{user_id}，电影编号：{movie_id}，推荐电影：'{movie_name}'，预测用户可能评分：{pred_rating:.2f}")