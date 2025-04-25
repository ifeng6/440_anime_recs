from data_preprocessing.load_clean_data import load_anime_data
from evaluation.evaluate_model import evaluate_cb_model, evaluate_cf_model
from models.collaborative_filtering import CollaborativeFilteringRecommender
from models.content_based import *
from recommenders.content_recommender import create_anime_index_map
from scipy.sparse import hstack
from scipy import sparse
from sklearn.model_selection import train_test_split
from utils import handle, main
import pandas as pd
import numpy as np

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

def build_genre_vocab(df):
    all_genres = set(g for genres in df['genres'] for g in genres)
    genre2idx = {g: i + 1 for i, g in enumerate(sorted(all_genres))}
    genre2idx['<PAD>'] = 0 # assign padding so tensors are same length
    return genre2idx

def custom_collate_fn(batch):
    from torch.nn.utils.rnn import pad_sequence
    import torch
    batch_dict = {key: [d[key] for d in batch] for key in batch[0]}
    batch_dict['image'] = torch.stack(batch_dict['image'])

    # genre_ids: list of variable-length tensors (e.g. [3], [4], ...)
    batch_dict['genre_ids'] = pad_sequence(batch_dict['genre_ids'], batch_first=True, padding_value=0)

    batch_dict['text'] = batch_dict['text']

    return batch_dict

# ------------------------------------------------------------------------------------------------------------------------- #

@handle("content_based")
def content_based():
    anime_df, ratings_df = load_anime_data()
    print(f"anime_df size: {anime_df.shape[0]}, ratings_df size: {ratings_df.shape[0]}")
    train_ratings, test_ratings = train_test_split_per_user(ratings_df)
    print(f"train ratings size: {train_ratings.shape[0]}, test ratings size: {test_ratings.shape[0]}")
    # missing_anime = set(ratings_df['anime_id']) - (set(train_ratings['anime_id']).union(set(test_ratings['anime_id'])))
    # print(f"missing anime: {missing_anime}")

    sample_users = test_ratings['user_id'].drop_duplicates().sample(10, random_state=42)
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

    results = evaluate_cb_model(train_subset, test_subset, anime_df, cosine_sim, anime_indices, top_k=25)
    print(results)

@handle("collaborative_filtering")
def collaborative_filtering():
    _, ratings_df = load_anime_data()
    model = CollaborativeFilteringRecommender()

    model.prepare_data(ratings_df)
    model.train()

    predictions = model.test()
    model.evaluate_rmse()
    k=25
    precisions, recalls, ndcgs = evaluate_cf_model(predictions, k=k)
    print(f"Precision@{k}: {np.mean(list(precisions.values())):.4f}")
    print(f"Recall@{k}:    {np.mean(list(recalls.values())):.4f}")
    print(f"NDCG@{k}:      {np.mean(list(ndcgs.values())):.4f}")

@handle("train_multimodal")
def multimodal():
    import torch
    from torch.utils.data import DataLoader
    from models.multimodal import AnimeDataset, AnimeMultimodalEncoder
    from tqdm import tqdm

    anime_df, _ = load_anime_data()

    # Extra preprocess for genre2idx
    anime_df['genres'] = anime_df['genres'].fillna('').apply(lambda x: x.strip().split())
    genre2idx = build_genre_vocab(anime_df)

    dataset = AnimeDataset(anime_df, genre2idx)
    dataloader = DataLoader(dataset, batch_size=16, collate_fn=custom_collate_fn)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = AnimeMultimodalEncoder(genre_vocab_size=len(genre2idx)).to(device)
    model.eval()

    all_embeddings = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Encoding anime"):
            text = batch["text"]
            image = batch["image"].to(device)
            genre_ids = batch["genre_ids"].to(device)

            emb = model(
                input_texts=text,
                input_images=image,
                genre_ids=genre_ids
            )
            all_embeddings.append(emb.cpu())

    final_embeddings = torch.cat(all_embeddings, dim=0).numpy()
    anime_df['embedding'] = list(final_embeddings)

    anime_df.to_pickle("anime_multimodal_embeddings.pkl")
    print("Saved anime_multimodal_embeddings.pkl")

@handle("evaluate-multimodal")
def eval_multimodal():
    _, rating_df = load_anime_data()
    anime_embeddings_df = pd.read_pickle("anime_multimodal_embeddings.pkl")


if __name__ == '__main__':
    main()