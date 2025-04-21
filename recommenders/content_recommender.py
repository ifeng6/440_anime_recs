import pandas as pd

def create_anime_index_map(anime_df):
    return pd.Series(anime_df.index, index=anime_df['anime_id']).drop_duplicates()

def get_similar_animes(anime_id, cosine_sim, anime_df, anime_indices, top_n=10):
    idx = anime_indices[anime_id]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:top_n+1]
    anime_indices_top = [i[0] for i in sim_scores]
    return anime_df.iloc[anime_indices_top][['anime_id', 'name', 'genres']]

def recommend_for_user(user_id, ratings_df, anime_df, cosine_sim, anime_indices, top_n=10):
    liked_animes = ratings_df[(ratings_df['user_id'] == user_id) & (ratings_df['rating'] >= 7)]

    if liked_animes.empty:
        print("No highly rated animes for this user.")
        return pd.DataFrame()

    recs = pd.DataFrame()

    for anime_id in liked_animes['anime_id']:
        if anime_id in anime_indices:
            similar = get_similar_animes(anime_id, cosine_sim, anime_df, anime_indices, top_n=5)
            recs = pd.concat([recs, similar], ignore_index=True)

    recs = recs[~recs['anime_id'].isin(liked_animes['anime_id'])]
    recs = recs.groupby(['anime_id', 'name', 'genres']).size().reset_index(name='score')
    recs = recs.sort_values('score', ascending=False).head(top_n)
    return recs