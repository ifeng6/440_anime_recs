from collections import defaultdict
import numpy as np
from tqdm import tqdm
from recommenders.content_recommender import recommend_for_user
from recommenders.multimodal_recommender import recommend_mm_for_user
from recommenders.ncf_recommender import recommend_ncf_for_user

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

def evaluate_multimodal_model(train_df, test_df, anime_embeddings_df, top_k=10, threshold=7):
    users = test_df['user_id'].unique()
    precision_scores = []
    recall_scores = []
    ndcg_scores = []

    for uid in tqdm(users, desc="Evaluating users"):
        recommended = recommend_mm_for_user(uid, train_df, anime_embeddings_df, top_k=top_k, threshold=threshold)
        if not recommended:
            continue

        relevant = test_df[(test_df['user_id'] == uid) & (test_df['rating'] >= threshold)]['anime_id'].tolist()
        if not relevant:
            continue

        prec = precision_at_k(recommended, relevant, top_k)
        rec = recall_at_k(recommended, relevant, top_k)
        ndcg = ndcg_at_k(recommended, relevant, top_k)

        precision_scores.append(prec)
        recall_scores.append(rec)
        ndcg_scores.append(ndcg)

    return {
        f'Precision@{top_k}': np.mean(precision_scores),
        f'Recall@{top_k}': np.mean(recall_scores),
        f'NDCG@{top_k}': np.mean(ndcg_scores),
        'Total evaluated users': len(precision_scores)
    }

def evaluate_ncf(model, train_df, test_df, anime_df, user2idx, item2idx, device, top_k=10):
    model.eval()

    train_user_items = train_df.groupby('user_id')['anime_id'].apply(list).to_dict()
    all_anime_ids = anime_df['anime_id'].tolist()
    precisions, recalls, ndcgs = [], [], []

    for user_id in tqdm(test_df['user_id'].unique(), desc="Evaluating users"):
        recommended_ids = recommend_ncf_for_user(
            model, user_id,
            train_user_items,
            all_anime_ids,
            user2idx=user2idx,
            item2idx=item2idx,
            device=device,
            top_k=top_k
        )

        relevant_ids = test_df[
            (test_df['user_id'] == user_id) &
            (test_df['rating'] >= 7)
        ]['anime_id'].tolist()

        if not relevant_ids:
            continue

        prec = precision_at_k(recommended_ids, relevant_ids, top_k)
        rec = recall_at_k(recommended_ids, relevant_ids, top_k)
        ndcg = ndcg_at_k(recommended_ids, relevant_ids, top_k)

        precisions.append(prec)
        recalls.append(rec)
        ndcgs.append(ndcg)

    results = {
        f'Precision@{top_k}': np.mean(precisions),
        f'Recall@{top_k}': np.mean(recalls),
        f'NDCG@{top_k}': np.mean(ndcgs),
        'Users evaluated': len(precisions)
    }
    return results

def leave_one_out_evaluate(model, anime_df, train_df, test_df, top_k=10):
    recalls = []
    precisions = []
    ndcgs = []
    users_evaluated = 0

    for idx, test_row in tqdm(test_df.iterrows(), total=len(test_df), desc="Evaluating"):
        user_id = test_row['user_id']
        true_anime_id = test_row['anime_id']

        try:
            test_anime = anime_df[anime_df['anime_id'] == true_anime_id].iloc[0]
            pseudo_prompt = test_anime['overview']
        except:
            continue
        prompt_emb = model.encode_prompt(pseudo_prompt)

        user_seen_animes = train_df[train_df['user_id'] == user_id]['anime_id'].tolist()

        candidate_anime_ids = anime_df[
            ~anime_df['anime_id'].isin(user_seen_animes)
        ]['anime_id'].tolist()

        try:
            recommendation_list = model.recommend_hybrid(
                user_id, 
                prompt_emb, 
                top_k=top_k,
                unseen_anime_ids=candidate_anime_ids
            )
        except:
            continue

        if len(recommendation_list) == 0:
            continue

        if true_anime_id in recommendation_list:
            recalls.append(1)
            precisions.append(1/top_k)  # one hit out of top_k
            rank = recommendation_list.index(true_anime_id) + 1
            ndcgs.append(1 / np.log2(rank + 1))
        else:
            recalls.append(0)
            precisions.append(0)
            ndcgs.append(0)

        users_evaluated += 1

    if users_evaluated == 0:
        print("No users evaluated.")
        return {}

    results = {
        f'Precision@{top_k}': np.mean(precisions),
        f'Recall@{top_k}': np.mean(recalls),
        f'NDCG@{top_k}': np.mean(ndcgs),
        'Users evaluated': users_evaluated
    }
    return results