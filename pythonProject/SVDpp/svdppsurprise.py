import pandas as pd
from surprise import SVDpp, Dataset, Reader, accuracy
from surprise.model_selection import cross_validate, train_test_split

# === 第一步：读取数据 ===
# 读取电影信息文件
movies_columns = ['movieId', 'title', 'genres']
movies_df = pd.read_csv('../Data/ml-1m/movies.dat', delimiter='::', names=movies_columns, engine='python', encoding='ISO-8859-1')

# 读取评分信息文件
ratings_columns = ['userId', 'movieId', 'rating', 'timestamp']
ratings_df = pd.read_csv('../Data/ml-1m/ratings.dat', delimiter='::', names=ratings_columns, engine='python', encoding='ISO-8859-1')

users_columns = ['userId', 'gender', 'age', 'occupation', 'zip_code']
users_df = pd.read_csv('../Data/ml-1m/users.dat', delimiter='::', engine='python', names=users_columns, encoding='ISO-8859-1')

# 合并电影和评分数据集
movies_ratings_df = ratings_df.merge(movies_df, on='movieId')

# 利用Surprise的读取器和数据集定义数据
reader = Reader(rating_scale=(0, 5))  # 评分范围从0到5
data = Dataset.load_from_df(movies_ratings_df[['userId', 'movieId', 'rating']], reader)

# 创建数据集拆分：训练集和测试集
trainset, testset = train_test_split(data, test_size=.25)

# 定义SVD++模型
svdpp = SVDpp()

# 训练模型
svdpp.fit(trainset)

# === 预测 ===
user_id = 1
movie_id = 364

# 通过SVDpp模型进行预测
pred = svdpp.predict(user_id, movie_id)

# 获取电影名称
movie_name = movies_df[movies_df['movieId'] == movie_id]['title'].values[0]

print(f"用户编号：{user_id}，电影编号：{movie_id}，推荐电影：'{movie_name}'，预测用户可能评分：{pred.est:.2f}")