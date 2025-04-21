from data_preprocessing.load_clean_data import load_anime_data
from evaluation.evaluate_model import evaluate_model
from models.content_based import *
from recommenders.content_recommender import create_anime_index_map
from scipy.sparse import hstack
from scipy import sparse
from sklearn.model_selection import train_test_split
from utils import handle, main
import pandas as pd

def train_test_split_per_user(df, test_size=0.2, min_ratings=5):
    train_list = []
    test_list = []

    for user_id, group in df.groupby('user_id'):
        if len(group) < min_ratings:
            train_list.append(group)
            continue

        train, test = train_test_split(group, test_size=test_size, random_state=10)
        train_list.append(train)
        test_list.append(test)

    train_df = pd.concat(train_list).reset_index(drop=True)
    test_df = pd.concat(test_list).reset_index(drop=True)

    return train_df, test_df

@handle("content_based")
def content_based():
    anime_df, ratings_df = load_anime_data()
    print(f"anime_df size: {anime_df.shape[0]}, ratings_df size: {ratings_df.shape[0]}")
    train_ratings, test_ratings = train_test_split_per_user(ratings_df)
    print(f"train ratings size: {train_ratings.shape[0]}, test ratings size: {test_ratings.shape[0]}")
    # missing_anime = set(ratings_df['anime_id']) - (set(train_ratings['anime_id']).union(set(test_ratings['anime_id'])))
    # print(f"missing anime: {missing_anime}")

    sample_users = test_ratings['user_id'].drop_duplicates().sample(100, random_state=42)
    test_subset = test_ratings[test_ratings['user_id'].isin(sample_users)]
    train_subset = train_ratings[train_ratings['user_id'].isin(sample_users)]

    tfidf_matrix, _ = build_tfidf_matrix(anime_df)
    feature_matrix = build_feature_matrix(anime_df)
    if not sparse.issparse(feature_matrix):
        feature_matrix = sparse.csr_matrix(feature_matrix)

    # Combine
    combined_matrix = hstack([tfidf_matrix, feature_matrix])
    cosine_sim = compute_similarity_matrix(combined_matrix)
    anime_indices = create_anime_index_map(anime_df)
    print(f"combined_matrix shape: {combined_matrix.shape}, cosime sim shape: {cosine_sim.shape}, anime indices shape: {anime_indices.shape}")

    results = evaluate_model(train_subset, test_subset, anime_df, cosine_sim, anime_indices, top_k=10)
    print(results)

if __name__ == '__main__':
    main()