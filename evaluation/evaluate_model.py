import numpy as np
from recommenders.content_recommender import recommend_for_user

def precision_at_k(recommended, relevant, k):
    recommended_k = recommended[:k]
    hits = sum([1 for item in recommended_k if item in relevant])
    return hits / k

def recall_at_k(recommended, relevant, k):
    recommended_k = recommended[:k]
    hits = sum([1 for item in recommended_k if item in relevant])
    return hits / len(relevant) if relevant else 0

def ndcg_at_k(recommended, relevant, k):
    dcg = 0.0
    for i, item in enumerate(recommended[:k]):
        if item in relevant:
            dcg += 1 / np.log2(i + 2)  # log2(i+2) because rank is 1-based
    # Ideal DCG = assume all relevant items are ranked at the top
    ideal_dcg = sum([1 / np.log2(i + 2) for i in range(min(len(relevant), k))])
    return dcg / ideal_dcg if ideal_dcg > 0 else 0

def evaluate_model(train_df, test_df, anime_df, cosine_sim, anime_indices, top_k=10):
    users = test_df['user_id'].unique()
    precision_scores, recall_scores, ndcg_scores = [], [], []

    for uid in users:
        print("starting user:", uid)
        recommended = recommend_for_user(user_id=uid,
                                         ratings_df=train_df,
                                         anime_df=anime_df,
                                         cosine_sim=cosine_sim,
                                         anime_indices=anime_indices,
                                         top_n=top_k)
        if recommended.empty:
            continue

        recommended_ids = recommended['anime_id'].tolist()
        relevant_ids = test_df[(test_df['user_id'] == uid) & (test_df['rating'] >= 7)]['anime_id'].tolist()

        if not relevant_ids:
            continue

        prec = precision_at_k(recommended_ids, relevant_ids, top_k)
        rec = recall_at_k(recommended_ids, relevant_ids, top_k)
        ndcg = ndcg_at_k(recommended_ids, relevant_ids, top_k)

        precision_scores.append(prec)
        recall_scores.append(rec)
        ndcg_scores.append(ndcg)

    return {
        f'Precision@{top_k}': np.mean(precision_scores),
        f'Recall@{top_k}': np.mean(recall_scores),
        f'NDCG@{top_k}': np.mean(ndcg_scores),
        'Total evaluated users': len(precision_scores)
    }