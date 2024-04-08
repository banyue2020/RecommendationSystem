import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class Svdpp:
    def __init__(self, K=10, steps=3, gamma=0.04, lambda_=0.15):
        self.K = K
        self.steps = steps
        self.gamma = gamma
        self.lambda_ = lambda_

    def load_data(self, path='../Data/ml-1m/'):
        self.path = path
        movies_columns = ['movieId', 'title', 'genres']
        self.movies_df = pd.read_csv(path + 'movies.dat', sep='::', names=movies_columns, engine='python', encoding='ISO-8859-1')
        ratings_columns = ['userId', 'movieId', 'rating', 'timestamp']
        self.ratings_df = pd.read_csv(path + 'ratings.dat', sep='::', names=ratings_columns, engine='python', encoding='ISO-8859-1')

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
                self.bu[uid] += self.gamma * (e_ui - self.lambda_ * self.bu[uid])
                self.bi[iid] += self.gamma * (e_ui - self.lambda_ * self.bi[iid])
                self.P[uid, :] += self.gamma * (e_ui * self.Q[iid, :] - self.lambda_ * self.P[uid, :])
                self.Q[iid, :] += self.gamma * (e_ui * self.P[uid, :] - self.lambda_ * self.Q[iid, :])

            self.gamma *= 0.9  # 学习率递减

    def top_n_recommendations(self, user_id, n=None):
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

    def get_top_n_recommendations(self, user_id, n=None):
        # 获取并返回包含电影标题的前n个推荐
        recommendations_df = self.top_n_recommendations(user_id, n)
        return recommendations_df

class MovieRecommender:
    def __init__(self):
        pass

    def load_data(self, path='../Data/ml-1m/'):
        self.path = path
        movies_columns = ['movieId', 'title', 'genres']
        self.movies_df = pd.read_csv(path + 'movies.dat', sep='::', names=movies_columns, engine='python', encoding='ISO-8859-1')

        ratings_columns = ['userId', 'movieId', 'rating', 'timestamp']
        self.ratings_df = pd.read_csv(path + 'ratings.dat', sep='::', names=ratings_columns, engine='python', encoding='ISO-8859-1')

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
        similarities_scores = cosine_similarities.flatten()
        predicted_scores = cosine_similarities.flatten()
        recommendations_df = self.movies_df.copy()
        recommendations_df['predicted_rating'] = predicted_scores*5
        rated_movie_ids = self.ratings_df[self.ratings_df['userId'] == user_id]['movieId'].tolist()
        recommendations_df = recommendations_df[~recommendations_df['movieId'].isin(rated_movie_ids)]
        self.top_n_recommendations_df = recommendations_df.nlargest(5, 'predicted_rating')

    def get_top_n_recommendations(self, user_id, n=None):
        # 默认参数n被设为None，不再限制返回的推荐数量
        if hasattr(self, 'top_n_recommendations_df'):
            if n:
                return self.top_n_recommendations_df.nlargest(n, 'predicted_rating')
            else:
                return self.top_n_recommendations_df
        else:
            self.generate_recommendations(user_id)
            return self.top_n_recommendations_df


class HybridRecommender:
    def __init__(self, model1, model2, weight1, weight2):
        self.model1 = model1
        self.model2 = model2
        self.weight1 = weight1
        self.weight2 = weight2

    def predict(self, user_id):
        # 获取两个模型的推荐电影
        predictions1 = self.model1.get_top_n_recommendations(user_id)
        predictions2 = self.model2.get_top_n_recommendations(user_id)

        # 标记每个预测的来源
        predictions1['source'] = 'model1'
        predictions2['source'] = 'model2'

        # 合并两个预测结果
        combined = pd.concat([predictions1, predictions2])


        # # 计算混合得分
        # combined['hybrid_rating'] = (
        #         self.weight1 * combined['predicted_rating'] +
        #         self.weight2 * combined['predicted_rating']
        # )
        #
        # # 找出只在一个表的电影
        # unique_movies = combined.groupby('movieId').filter(lambda group: len(group) == 1)
        #
        # # 找出在两个表中都出现的电影
        # intersect_movies = combined.groupby('movieId').filter(lambda group: len(group) > 1)
        #
        # # 对交集电影进行得分排序
        # intersect_movies = intersect_movies.sort_values('hybrid_rating', ascending=False)
        #
        # # 先推荐交集电影，然后推荐单一表电影
        # ordered_recommendations = pd.concat([intersect_movies, unique_movies])
        #
        # return ordered_recommendations[:5]
        #
        # # 将合并后的列表按照电影名字进行分组
        # grouped = combined.groupby('title')
        #
        # # 创建一个空列表，用来保存混合后的结果
        # hybrid_list = []
        #
        # # 对每个组进行混合操作
        # for name, group in grouped:
        #     # 根据来源获取每个模型的预测评分
        #     pred1 = group[group['source'] == 'model1']['predicted_rating'].mean()
        #     pred2 = group[group['source'] == 'model2']['predicted_rating'].mean()
        #
        #     # 计算混合评分
        #     hybrid_rating = self.weight1 * pred1 + self.weight2 * pred2
        #
        #     # 添加到结果列表中
        #     hybrid_list.append({
        #         'title': name,
        #         'movieId': group['movieId'].values[0],  # 添加这一行
        #         'hybrid_rating': hybrid_rating
        #     })
        #
        # # 创建一个新的DataFrame，从结果列表中
        # hybrid_ratings = pd.DataFrame(hybrid_list)
        #
        # # 按照混合评分降序排序并返回前 5 部电影
        # return hybrid_ratings.sort_values('hybrid_rating', ascending=False).head(5)

        # 为了处理数据，我们先创建一个标记列，用于标识每个电影的预测是来自单一模型还是两个模型都有
        combined['is_unique'] = combined.groupby('movieId')['source'].transform('nunique') == 1

        # 计算混合得分，只对同时出现在两个模型中电影进行计算
        combined['hybrid_rating'] = combined.apply(
            lambda x: self.weight1 * x['predicted_rating'] + self.weight2 * x['predicted_rating']
            if not x['is_unique'] else x['predicted_rating'],
            axis=1
        )

        # 先处理在两个模型中都出现的电影
        intersect_movies = combined[~combined['is_unique']].copy()
        intersect_movies = intersect_movies.groupby('movieId').agg({
            'title': 'first',  # 假设同一 movieId 的所有 title 都是相同的
            'hybrid_rating': 'sum'  # 对同一个电影的混合评分进行求和
        }).reset_index()
        intersect_movies = intersect_movies.sort_values(by='hybrid_rating', ascending=False)

        # 然后处理只在一个模型中出现的电影
        unique_movies = combined[combined['is_unique']].copy()
        unique_movies = unique_movies.sort_values(by=['hybrid_rating', 'movieId'], ascending=[False, True])

        # 将交集电影和单一表电影按顺序合并
        final_recommendations = pd.concat([intersect_movies, unique_movies])

        # 我们只返回合并列表中的前5部电影
        return final_recommendations.head(5)

if __name__ == "__main__":
    # 创建推荐器实例
    svdpp = Svdpp()
    svdpp.load_data(path='../Data/ml-1m/')
    svdpp.train()


    # 获取推荐
    user_id = 1
    # top_n_movies = svdpp.top_n_recommendations(user_id, n=5)
    all_recommendations = svdpp.top_n_recommendations(user_id)
    # 只输出前五个结果
    top_five_recommendations = all_recommendations.nlargest(5, 'predicted_rating')
    print(f"SVDPP推荐的前五结果:")
    for _, row in top_five_recommendations.iterrows():
        print(f"用户编号：{user_id}，电影编号：{row['movieId']}，推荐电影：'{row['title']}'，预测用户可能评分：{row['predicted_rating']:.2f}")


    recommender = MovieRecommender()
    recommender.load_data(path='../Data/ml-1m/')
    recommender.create_user_profile(user_id)
    all_recommendations = recommender.get_top_n_recommendations(user_id)
    # 只输出前五个结果
    top_five_recommendations = all_recommendations.nlargest(5, 'predicted_rating')
    print(f"CB推荐的前五结果:")
    for _, row in top_five_recommendations.iterrows():
        print(f"用户编号：{user_id}，电影编号：{row['movieId']}，推荐电影：'{row['title']}'，预测用户可能评分：{row['predicted_rating']:.2f}")


    # 初始化一个加权融合推荐器的实例
    weight1 = 0.8
    weight2 = 0.2
    hybrid_recommender = HybridRecommender(svdpp, recommender, weight1, weight2)
    # 使用加权融合推荐器预测
    hybrid_recommendations = hybrid_recommender.predict(user_id)
    # 打印预测评分
    print(f"融合推荐结果")
    for _, row in hybrid_recommendations.iterrows():
        print(
            f"用户编号：{user_id}，电影编号：{row['movieId']}，推荐电影：'{row['title']}'，加权融合评分：{row['hybrid_rating']:.2f}")