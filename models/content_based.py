from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import numpy as np

def build_tfidf_matrix(anime_df):
    anime_df['content'] = anime_df['genres']
    tfidf = TfidfVectorizer(stop_words='english', max_features=10000)
    tfidf_matrix = tfidf.fit_transform(anime_df['content'])
    return tfidf_matrix, tfidf

def build_feature_matrix(anime_df):
    # Select features
    num_features = ['average_rating', 'popularity']

    # Fill missing values
    anime_df[num_features] = anime_df[num_features].fillna(0)

    # Preprocess
    numeric = StandardScaler()

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric, num_features),
        ]
    )

    feature_matrix = preprocessor.fit_transform(anime_df)
    return feature_matrix

def compute_similarity_matrix(matrix):
    cosine_sim = linear_kernel(matrix, matrix)
    return cosine_sim