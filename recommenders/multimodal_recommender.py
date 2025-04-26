import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

def recommend_mm_for_user(user_id, train_df, anime_embeddings_df, top_k=10, threshold=7):
    liked_anime_ids = train_df[
        (train_df['user_id'] == user_id) & (train_df['rating'] >= threshold)
    ]['anime_id'].tolist()

    if not liked_anime_ids:
        return []

    liked_embeddings = anime_embeddings_df[
        anime_embeddings_df['anime_id'].isin(liked_anime_ids)
    ]['embedding'].tolist()

    liked_embeddings = np.array(liked_embeddings)
    user_embed = np.mean(liked_embeddings, axis=0).reshape(1, -1)

    seen_ids = set(liked_anime_ids)
    candidates = anime_embeddings_df[~anime_embeddings_df['anime_id'].isin(seen_ids)]

    candidate_embeddings = np.vstack(candidates['embedding'].tolist())
    similarities = cosine_similarity(user_embed, candidate_embeddings)[0]

    candidates = candidates.copy()
    candidates['score'] = similarities

    top_k_recs = candidates.sort_values(by='score', ascending=False).head(top_k)
    return top_k_recs['anime_id'].tolist()