import torch

def recommend_ncf_for_user(model, user_id, train_user_items, all_anime_ids,
                           user2idx, item2idx, device, top_k=10):
    model.eval()

    with torch.no_grad():
        seen_items = set(train_user_items.get(user_id, []))  # Anime seen during train
        candidates = [aid for aid in all_anime_ids if (aid not in seen_items and aid in item2idx)]

        if not candidates:
            return []

        user_tensor = torch.tensor([user2idx[user_id]] * len(candidates), dtype=torch.long).to(device)
        item_tensor = torch.tensor([item2idx[aid] for aid in candidates], dtype=torch.long).to(device)
        
        preds = model(user_tensor, item_tensor)
        preds = preds.cpu().numpy()

        # Rank by predicted score
        top_idx = preds.argsort()[::-1][:top_k]
        top_anime_ids = [candidates[i] for i in top_idx]

    return top_anime_ids