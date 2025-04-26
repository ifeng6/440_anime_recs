import torch
import torch.nn as nn
from torch.utils.data import Dataset

def ncf_collate_batch(batch):
    user_ids = torch.stack([b['user_id'] for b in batch])
    item_ids = torch.stack([b['item_id'] for b in batch])
    ratings = torch.stack([b['rating'] for b in batch])
    return {
        'user_id': user_ids,
        'item_id': item_ids,
        'rating': ratings
    }

class NCFDataset(Dataset):
    def __init__(self, df, user2idx, item2idx):
        self.user_ids = df['user_id'].map(user2idx).values
        self.item_ids = df['anime_id'].map(item2idx).values
        self.ratings = df['rating'].values

    def __len__(self):
        return len(self.user_ids)
    
    def __getitem__(self, idx):
        return {
            'user_id': torch.tensor(self.user_ids[idx], dtype=torch.long),
            'item_id': torch.tensor(self.item_ids[idx], dtype=torch.long),
            'rating': torch.tensor(self.ratings[idx], dtype=torch.float32)
        }

class NCF(nn.Module):
    def __init__(self, num_users, num_items, user_emb_dim=64, item_emb_dim=64, hidden_dims=[128, 64]):
        super().__init__()
        self.user_embedding = nn.Embedding(num_users, user_emb_dim)
        self.item_embedding = nn.Embedding(num_items, item_emb_dim)

        input_dim = user_emb_dim + item_emb_dim
        layers = []
        for hidden in hidden_dims:
            layers.append(nn.Linear(input_dim, hidden))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.2))
            input_dim = hidden
        layers.append(nn.Linear(input_dim, 1))
        self.mlp = nn.Sequential(*layers)

    def forward(self, user_ids, item_ids):
        user_emb = self.user_embedding(user_ids)
        item_emb = self.item_embedding(item_ids)

        features = torch.cat([user_emb, item_emb], dim=1)
        output = self.mlp(features)
        return output.squeeze(1)