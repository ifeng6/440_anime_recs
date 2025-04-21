from data_preprocessing.load_clean_data import load_anime_data
from models.content_based import *
from recommenders.content_recommender import recommend_for_user, create_anime_index_map
from scipy.sparse import hstack
from scipy import sparse

def main(user_id):
    anime_df, ratings_df = load_anime_data()
    tfidf_matrix, _ = build_tfidf_matrix(anime_df)
    feature_matrix = build_feature_matrix(anime_df)
    if not sparse.issparse(feature_matrix):
        feature_matrix = sparse.csr_matrix(feature_matrix)

    # Combine
    combined_matrix = hstack([tfidf_matrix, feature_matrix])
    cosine_sim = compute_similarity_matrix(combined_matrix)
    anime_indices = create_anime_index_map(anime_df)

    recommendations = recommend_for_user(user_id, ratings_df, anime_df, cosine_sim, anime_indices, top_n=10)
    print(recommendations)

if __name__ == '__main__':
    user_id_to_test = 357
    main(user_id_to_test)