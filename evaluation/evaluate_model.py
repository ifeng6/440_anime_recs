from collections import defaultdict
import numpy as np
from tqdm import tqdm
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

def evaluate_cb_model(train_df, test_df, anime_df, cosine_sim, anime_indices, top_k=10):
    users = test_df['user_id'].unique()
    precision_scores, recall_scores, ndcg_scores = [], [], []

    for uid in tqdm(users, desc="Evaluating users"):
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

# From Surprise official documentation (http://surprise.readthedocs.io/en/stable/FAQ.html)
def evaluate_cf_model(predictions, k=10, threshold=7):
    """Return precision and recall at k metrics for each user"""

    # First map the predictions to each user.
    user_est_true = defaultdict(list)
    for uid, iid, true_r, est, _ in predictions:
        user_est_true[uid].append((iid, est, true_r))

    precisions = dict()
    recalls = dict()
    ndcgs = dict()

    for uid, user_ratings in tqdm(user_est_true.items(), desc="Evaluating users"):
        # Sort by estimated rating
        user_ratings.sort(key=lambda x: x[1], reverse=True)

        # Extract top-k recommended item IDs
        top_k_items = [iid for iid, _, _ in user_ratings[:k]]

        # Relevant items (true rating >= threshold)
        relevant_items = [iid for iid, _, true_r in user_ratings if true_r >= threshold]

        if not relevant_items:
            continue

        # Precision@K
        n_rec_k = sum(iid in relevant_items for iid in top_k_items)
        precisions[uid] = n_rec_k / k

        # Recall@K
        recalls[uid] = n_rec_k / len(relevant_items)

        # NDCG@K
        ndcgs[uid] = ndcg_at_k(top_k_items, relevant_items, k)

    return precisions, recalls, ndcgs