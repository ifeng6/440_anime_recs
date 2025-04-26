import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence

def ncf_collate_batch(batch):
    user_ids = torch.stack([b['user_id'] for b in batch])
    item_ids = torch.stack([b['item_id'] for b in batch])
    ratings = torch.stack([b['rating'] for b in batch])
    genres = [b['genres'] for b in batch]
    genres_padded = pad_sequence(genres, batch_first=True, padding_value=0)
    return {
        'user_id': user_ids,
        'item_id': item_ids,
        'genres': genres_padded,
        'rating': ratings
    }

class NCFDataset(Dataset):
    def __init__(self, df, user2idx, item2idx, genre_map):
        self.user_ids = df['user_id'].map(user2idx).values
        self.item_ids = df['anime_id'].map(item2idx).values
        self.genre_lists = df['genres'].map(lambda genres: [genre_map[g] for g in genres])
        self.ratings = df['rating'].values

    def __len__(self):
        return len(self.user_ids)
    
    def __getitem__(self, idx):
        return {
            'user_id': torch.tensor(self.user_ids[idx], dtype=torch.long),
            'item_id': torch.tensor(self.item_ids[idx], dtype=torch.long),
            'genres': torch.tensor(self.genre_lists[idx], dtype=torch.long),
            'rating': torch.tensor(self.ratings[idx], dtype=torch.float32)
        }

class NCF(nn.Module):
    def __init__(self, num_users, num_items, num_genres, user_emb_dim=64, item_emb_dim=64, genre_emb_dim=32, hidden_dims=[128, 64]):
        super().__init__()
        self.user_embedding = nn.Embedding(num_users, user_emb_dim)
        self.item_embedding = nn.Embedding(num_items, item_emb_dim)
        self.genre_embedding = nn.Embedding(num_genres, genre_emb_dim)

        input_dim = user_emb_dim + item_emb_dim + genre_emb_dim
        layers = []
        for hidden in hidden_dims:
            layers.append(nn.Linear(input_dim, hidden))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.2))
            input_dim = hidden
        layers.append(nn.Linear(input_dim, 1))
        self.mlp = nn.Sequential(*layers)

    def forward(self, user_ids, item_ids, genre_ids):
        user_emb = self.user_embedding(user_ids)
        item_emb = self.item_embedding(item_ids)
        genre_emb = self.genre_embedding(genre_ids)

        # Pool genre embeddings (mean)
        mask = (genre_ids != 0).unsqueeze(-1)
        genre_emb = (genre_emb * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)

        features = torch.cat([user_emb, item_emb, genre_emb], dim=1)
        output = self.mlp(features)
        return output.squeeze(1)