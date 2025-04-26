import torch

def recommend_ncf_for_user(model, user_id, train_user_items, all_anime_ids, item_genres_df,
                           user2idx, item2idx, genre2idx, device, top_k=10):
    model.eval()

    with torch.no_grad():
        seen_items = set(train_user_items.get(user_id, []))  # Anime seen during train
        candidates = [aid for aid in all_anime_ids if (aid not in seen_items and aid in item2idx)]

        if not candidates:
            return []

        user_tensor = torch.tensor([user2idx[user_id]] * len(candidates), dtype=torch.long).to(device)
        item_tensor = torch.tensor([item2idx[aid] for aid in candidates], dtype=torch.long).to(device)

        genre_lists = []
        for aid in candidates:
            genres = item_genres_df.get(aid, [])
            genre_ids = [genre2idx.get(g, 0) for g in genres]
            genre_lists.append(torch.tensor(genre_ids, dtype=torch.long))
        
        # Pad genre lists
        from torch.nn.utils.rnn import pad_sequence
        genre_padded = pad_sequence(genre_lists, batch_first=True, padding_value=0).to(device)

        preds = model(user_tensor, item_tensor, genre_padded)
        preds = preds.cpu().numpy()

        # Rank by predicted score
        top_idx = preds.argsort()[::-1][:top_k]
        top_anime_ids = [candidates[i] for i in top_idx]

    return top_anime_ids