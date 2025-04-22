from surprise import SVD, Dataset, Reader
from surprise.model_selection import train_test_split
from surprise import accuracy

class CollaborativeFilteringRecommender:
    def __init__(self, n_factors=50, rating_scale=(5, 10), test_size=0.2, random_state=69):
        self.n_factors = n_factors
        self.rating_scale = rating_scale
        self.test_size = test_size
        self.random_state = random_state
        self.model = SVD(n_factors=self.n_factors, random_state=self.random_state)
        self.trainset = None
        self.testset = None
        self.predictions = None

    def prepare_data(self, ratings_df):
        reader = Reader(rating_scale=self.rating_scale)
        data = Dataset.load_from_df(ratings_df[['user_id', 'anime_id', 'rating']], reader)
        self.trainset, self.testset = train_test_split(data, test_size=self.test_size, random_state=self.random_state)

    def train(self):
        self.model.fit(self.trainset)

    def test(self):
        self.predictions = self.model.test(self.testset)
        return self.predictions

    def evaluate_rmse(self):
        return accuracy.rmse(self.predictions, verbose=True)