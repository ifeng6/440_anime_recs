from data_preprocessing.load_clean_data import load_anime_data
from evaluation.evaluate_model import evaluate_cb_model, evaluate_cf_model, evaluate_multimodal_model, evaluate_ncf, leave_one_out_evaluate
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

def leave_one_out_split(rating_df, leave_out=1, threshold=7):
    train_rows = []
    test_rows = []

    for user_id, user_data in rating_df.groupby('user_id'):
        user_ratings = rating_df[rating_df['user_id'] == user_id]
        high_rated = user_ratings[user_ratings['rating'] >= threshold]

        if len(high_rated) < leave_out:
            continue  # skip small profile users

        # Randomly select one highly rated anime
        test_interaction = high_rated.sample(leave_out)
        train_interactions = user_data.drop(test_interaction.index)

        train_rows.append(train_interactions)
        test_rows.append(test_interaction)

    train_df = pd.concat(train_rows)
    test_df = pd.concat(test_rows)

    return train_df.reset_index(drop=True), test_df.reset_index(drop=True)

# ------------------------------------------------------------------------------------------------------------------------- #

@handle("cb")
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

@handle("cf")
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
    anime_df['genres'] = anime_df['genres'].fillna('').apply(lambda x: x.strip().split(","))
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
    train_ratings, test_ratings = train_test_split_per_user(rating_df)

    results = evaluate_multimodal_model(train_ratings, test_ratings, anime_embeddings_df, top_k=25)
    for metric, value in results.items():
        print(f"{metric}: {value:.4f}")


@handle("ncf")
def ncf():
    import torch
    from torch.utils.data import DataLoader
    from torch import nn, optim
    from tqdm import tqdm
    from models.NCF import NCF, NCFDataset, ncf_collate_batch
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    anime_df, rating_df = load_anime_data()

    # Extra preprocess for genre2idx
    anime_df['genres'] = anime_df['genres'].fillna('').apply(lambda x: x.strip().split(","))
    rating_df['genres'] = rating_df['genres'].fillna('').apply(lambda x: x.strip().split(","))

    user_ids = rating_df['user_id'].unique()
    anime_ids = rating_df['anime_id'].unique()

    user2idx = {u: idx for idx, u in enumerate(user_ids)}
    item2idx = {i: idx for idx, i in enumerate(anime_ids)}

    num_users = len(user2idx)
    num_items = len(item2idx)

    print(f"Users: {num_users}, Items: {num_items}")
    train_df, test_df = train_test_split_per_user(rating_df, test_size=0.2)

    train_dataset = NCFDataset(train_df, user2idx, item2idx)
    train_loader = DataLoader(train_dataset, batch_size=512, shuffle=True, collate_fn=ncf_collate_batch)

    model = NCF(
        num_users=num_users,
        num_items=num_items,
        user_emb_dim=64,
        item_emb_dim=64,
        hidden_dims=[128, 64]
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()

    num_epochs = 15

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            user_ids = batch['user_id'].to(device)
            item_ids = batch['item_id'].to(device)
            ratings = batch['rating'].to(device)

            preds = model(user_ids, item_ids)
            loss = loss_fn(preds, ratings)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * len(user_ids)

        avg_loss = total_loss / len(train_loader.dataset)
        print(f"Epoch {epoch+1} - Avg MSE Loss: {avg_loss:.4f}")

    torch.save(model.state_dict(), "ncf.pth")
    print("Model saved!")
    results = evaluate_ncf(
        model,
        train_df,
        test_df,
        anime_df,
        user2idx,
        item2idx,
        device,
        top_k=25
    )

    print("Evaluation Results:")
    for metric, value in results.items():
        print(f"{metric}: {value:.4f}")

@handle("prompt")
def prompt():
    from models.prompt import PromptModel
    
    anime_df, rating_df = load_anime_data()
    leave_out = 1
    print(rating_df.shape)
    train_df, test_df = leave_one_out_split(rating_df, leave_out=leave_out)
    print(train_df.shape, test_df.shape)
    model = PromptModel(anime_df, train_df)
    model.encode_anime()
    model.train_cf_model()
    results = leave_one_out_evaluate(model, anime_df, train_df, test_df, top_k=25)

    print("\nEvaluation Results:")
    for metric, value in results.items():
        print(f"{metric}: {value:.4f}")


if __name__ == '__main__':
    main()