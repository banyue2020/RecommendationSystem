import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from scipy.stats import zscore

class Svdpp:
    def __init__(self, K=10, steps=3, gamma=0.04, lambdas=0.15):
        self.K = K
        self.steps = steps
        self.gamma = gamma
        self.lambdas = lambdas

    def load_data(self, path='../Data/ml-1m/'):
        self.path = path
        movies_columns = ['movieId', 'title', 'genres']
        self.movies_df = pd.read_csv(path + 'movies.dat', sep='::', names=movies_columns, engine='python',encoding='ISO-8859-1')
        ratings_columns = ['userId', 'movieId', 'rating', 'timestamp']
        self.ratings_df = pd.read_csv(path + 'ratings.dat', sep='::', names=ratings_columns, engine='python',encoding='ISO-8859-1')

        self.mu = self.ratings_df['rating'].mean()
        self.M, self.N = len(np.unique(self.ratings_df['userId'])), len(np.unique(self.ratings_df['movieId']))

        np.random.seed(0)
        self.P, self.Q = np.random.rand(self.M, self.K), np.random.rand(self.N, self.K)
        self.bu, self.bi = np.zeros(self.M), np.zeros(self.N)

        # 创建字典，用于映射原始ID到内部索引
        self.user_dict = pd.DataFrame(np.unique(self.ratings_df['userId']), columns=['raw_uid']).reset_index().set_index('raw_uid').to_dict()['index']
        self.item_dict = pd.DataFrame(np.unique(self.ratings_df['movieId']), columns=['raw_iid']).reset_index().set_index('raw_iid').to_dict()['index']

    def train(self):
        # 随机梯度下降法训练模型参数
        for step in range(self.steps):
            print("Training step ", step + 1, "...")
            for uid, iid, r_ui in self.ratings_df[['userId', 'movieId', 'rating']].values:
                # 获取用户和物品的内部索引
                uid = self.user_dict[uid]
                iid = self.item_dict[iid]

                # 计算预测评分的误差
                pred_r_ui = self.mu + self.bu[uid] + self.bi[iid] + np.dot(self.P[uid], self.Q[iid].T)
                e_ui = r_ui - pred_r_ui

                # 更新偏置项和隐含因子矩阵
                self.bu[uid] += self.gamma * (e_ui - self.lambdas * self.bu[uid])
                self.bi[iid] += self.gamma * (e_ui - self.lambdas * self.bi[iid])
                self.P[uid, :] += self.gamma * (e_ui * self.Q[iid, :] - self.lambdas * self.P[uid, :])
                self.Q[iid, :] += self.gamma * (e_ui * self.P[uid, :] - self.lambdas * self.Q[iid, :])

            self.gamma *= 0.9  # 学习率递减

    def get_recommendations(self, user_id, n=None):
        uid = self.user_dict[user_id]
        iid_list = self.item_dict.values()

        # 计算用户对所有电影的评分
        rating_list = [self.mu + self.bu[uid] + self.bi[iid] + np.dot(self.P[uid, :], self.Q[iid, :].T) for iid in iid_list]

        # 生成评分的数据框
        rating_df = pd.DataFrame(data={'movieId': list(self.item_dict.keys()), 'predicted_rating': rating_list})

        # 和电影信息进行合并
        result_df = pd.merge(rating_df, self.movies_df, on='movieId')

        # 根据预测评分排序并取出前n部
        return result_df.sort_values(by='predicted_rating', ascending=False).iloc[:n, :]


class ContentBased:
    def __init__(self):
        pass

    def load_data(self, path='../Data/ml-1m/'):
        self.path = path
        movies_columns = ['movieId', 'title', 'genres']
        self.movies_df = pd.read_csv(path + 'movies.dat', sep='::', names=movies_columns, engine='python',encoding='ISO-8859-1')

        ratings_columns = ['userId', 'movieId', 'rating', 'timestamp']
        self.ratings_df = pd.read_csv(path + 'ratings.dat', sep='::', names=ratings_columns, engine='python',encoding='ISO-8859-1')

        self.movies_df['Year'] = self.movies_df['title'].str.extract(r'(d{4})')
        self.movies_df['title'] = self.movies_df['title'].str.replace(r'(d{4})', '').str.strip()
        self.movies_df['genres'] = self.movies_df['genres'].apply(lambda x: x.replace('|', ' '))

        self.tfidf = TfidfVectorizer(stop_words='english')
        self.tfidf_matrix = self.tfidf.fit_transform(self.movies_df['genres'])

    def create_user_profile(self, user_id):
        user_ratings = self.ratings_df[self.ratings_df['userId'] == user_id]
        movies_idx = user_ratings['movieId'].apply(lambda x: self.movies_df[self.movies_df['movieId'] == x].index.values)
        if not movies_idx.empty:
            movies_idx = np.hstack(movies_idx.values)
        user_tfidf = self.tfidf_matrix[movies_idx.astype(int)]
        self.user_profile = np.dot(user_ratings['rating'].values, user_tfidf.toarray())
        self.user_profile /= user_ratings['rating'].sum()

    def generate_recommendations(self, user_id):
        user_profile = np.array(self.user_profile).reshape(1, -1)
        cosine_similarities = cosine_similarity(user_profile, self.tfidf_matrix)
        predicted_scores = cosine_similarities.flatten()
        recommendations_df = self.movies_df.copy()
        recommendations_df['predicted_rating'] = predicted_scores * 5
        rated_movie_ids = self.ratings_df[self.ratings_df['userId'] == user_id]['movieId'].tolist()
        recommendations_df = recommendations_df[~recommendations_df['movieId'].isin(rated_movie_ids)]
        self.top_recommendations_df = recommendations_df

    def get_recommendations(self, user_id, n=None):
        # 默认参数n被设为None，不再限制返回的推荐数量
        if hasattr(self, 'top_recommendations_df'):
            if n:
                return self.top_recommendations_df.nlargest(n, 'predicted_rating')
            else:
                return self.top_recommendations_df
        else:
            self.generate_recommendations(user_id)
            return self.top_recommendations_df


class HybridRecommender:
    def __init__(self, model1, model2, weight1, weight2):
        self.model1 = model1
        self.model2 = model2
        self.weight1 = weight1
        self.weight2 = weight2

    def predict(self, user_id):
        # 获取两个模型的推荐电影
        predictions1 = self.model1.get_recommendations(user_id)
        predictions2 = self.model2.get_recommendations(user_id)

        # 标记每个预测的来源
        predictions1['source'] = 'model1'
        predictions2['source'] = 'model2'

        # 合并两个预测结果
        combined = pd.concat([predictions1, predictions2])

        # 正规化评分，转化为z分数并调整到0-5范围
        combined['predicted_rating'] = normalize_to_5(combined.groupby('source')['predicted_rating'].transform(zscore))
        combined['original_rating'] = combined['predicted_rating']

        # 计算有差异性的加权混合评分
        def adjust_rating(row):
            if row['source'] == 'model1':
                return self.weight1 * row['predicted_rating']
            else:
                return self.weight2 * row['predicted_rating']

        combined['adjusted_rating'] = combined.apply(adjust_rating, axis=1)

        # 分组以得到重叠和独特的电影记录
        grouped = combined.groupby('movieId')

        # 遍历每个电影的组
        final_recommendations = []
        for movie_id, group in grouped:
            title = group['title'].iloc[0]
            # 如果电影来自两个模型
            if len(group) > 1:
                # 取平均值作为混合评分
                hybrid_rating = group['adjusted_rating'].mean()
            else:
                # 取单个模型的评分，并降低它的权重
                hybrid_rating = group['adjusted_rating'].iloc[0] * 0.7

            final_recommendations.append({
                'movieId': movie_id,
                'title': title,
                'hybrid_rating': hybrid_rating,
                'original_ratings': group['original_rating'].tolist()  # 保存原始评分以供参考
            })

        # 将推荐转换为DataFrame并排序
        final_recommendations_df = pd.DataFrame(final_recommendations)
        final_recommendations_df = final_recommendations_df.sort_values(by='hybrid_rating', ascending=False)

        # 只返回前五部电影
        return final_recommendations_df.head(5)


def normalize_to_5(scores):  # 定义一个正规化函数，将 z 分数转化为 0 到 5 的分数
    min_score = scores.min()
    range_score = scores.max() - min_score

    # 将所有的分数转换为 0-1
    normalized_scores = (scores - min_score) / range_score

    # 将所有的分数转换为 0-5
    normalized_scores = normalized_scores * 5

    return normalized_scores


if __name__ == "__main__":

    user_id = 1

    # 创建推荐器实例
    recommender_s = Svdpp()
    recommender_s.load_data(path='../Data/ml-1m/')
    recommender_s.train()
    all_recommendations_s = recommender_s.get_recommendations(user_id)
    # 只输出前五个结果
    top_recommendations_s = all_recommendations_s.nlargest(5, 'predicted_rating')
    print(f"SVDPP推荐的前五结果:")
    for _, row in top_recommendations_s.iterrows():
        print(
            f"用户编号：{user_id}，电影编号：{row['movieId']}，推荐电影：'{row['title']}'，预测用户可能评分：{row['predicted_rating']:.2f}")

    recommender_c = ContentBased()
    recommender_c.load_data(path='../Data/ml-1m/')
    recommender_c.create_user_profile(user_id)
    all_recommendations_c = recommender_c.get_recommendations(user_id)
    # 只输出前五个结果
    top_recommendations = all_recommendations_c.nlargest(5, 'predicted_rating')
    print(f"CB推荐的前五结果:")
    for _, row in top_recommendations.iterrows():
        print(
            f"用户编号：{user_id}，电影编号：{row['movieId']}，推荐电影：'{row['title']}'，预测用户可能评分：{row['predicted_rating']:.2f}")

    # 初始化一个加权融合推荐器的实例
    weight1 = 0.5
    weight2 = 0.5
    hybrid_recommender = HybridRecommender(recommender_s, recommender_c, weight1, weight2)
    # 使用加权融合推荐器预测
    hybrid_recommendations = hybrid_recommender.predict(user_id)
    # 打印预测评分
    print(f"融合推荐结果")
    for _, row in hybrid_recommendations.iterrows():
        print(
            f"用户编号：{user_id}，电影编号：{row['movieId']}，推荐电影：'{row['title']}'，加权融合评分：{row['hybrid_rating']:.2f}")