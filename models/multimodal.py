import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer
from torch.utils.data import Dataset
from torchvision import transforms, models
from PIL import Image
import requests
from io import BytesIO

# Standardize dataset for encoder
class AnimeDataset(Dataset):
    def __init__(self, anime_df, genre2idx, transform=None):
        self.df = anime_df.reset_index(drop=True)
        self.genre2idx = genre2idx

        self.transform = transform or transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.5]*3, [0.5]*3)
        ])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # --- Text ---
        text = str(row['overview'])

        # --- Image (from URL) ---
        url = row['image url']
        image = self.download_image(url)
        image = self.transform(image)

        # --- Genres ---
        genre_list = row['genres']
        genre_ids = [self.genre2idx[g] for g in genre_list if g in self.genre2idx]
        genre_ids = torch.tensor(genre_ids, dtype=torch.long)

        return {
            'text': text,
            'image': image,
            'genre_ids': genre_ids,
        }

    def download_image(self, url):
        try:
            response = requests.get(url, timeout=5)
            image = Image.open(BytesIO(response.content)).convert('RGB')
        except Exception as e:
            # Handle bad/missing image
            image = Image.new('RGB', (224, 224), color=(0, 0, 0))
        return image

# 1. synopsis (overview) --> BERT
# 2. anime cover art --> ResNet18
# 3. genres --> embedding
class AnimeMultimodalEncoder(nn.Module):
    def __init__(self, genre_vocab_size, output_dim=256):
        super().__init__()

        # Synopsis (BERT)
        self.text_model = AutoModel.from_pretrained("distilbert-base-uncased")
        self.text_tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
        self.text_proj = nn.Linear(self.text_model.config.hidden_size, output_dim)

        # Image Embedding (ResNet18)
        resnet = models.resnet18(pretrained=True)
        self.image_encoder = nn.Sequential(
            *list(resnet.children())[:-1],  # remove final FC
            nn.Flatten(),
            nn.Linear(512, output_dim)
        )

        # Genre Embedding
        self.genre_embedding = nn.Embedding(genre_vocab_size, output_dim)

        # MLP layer to combine all embeddings
        self.fusion = nn.Sequential(
            nn.Linear(output_dim * 3, output_dim),
            nn.ReLU(),
            nn.Linear(output_dim, output_dim)
        )

    def forward(self, input_texts, input_images, genre_ids):
        device = next(self.parameters()).device

        # Text encoding
        text_inputs = self.text_tokenizer(
            input_texts,
            padding=True,
            truncation=True,
            max_length=128,
            return_tensors='pt'
        )
        text_inputs = {k: v.to(device) for k, v in text_inputs.items()}
        text_out = self.text_model(**text_inputs)
        text_embed = self.text_proj(text_out.last_hidden_state[:, 0])

        image_embed = self.image_encoder(input_images)

        genre_embed = self.genre_embedding(genre_ids)
        mask = (genre_ids != 0).unsqueeze(-1)
        masked_embed = genre_embed * mask
        embed_sum = masked_embed.sum(dim=1)
        count = mask.sum(dim=1).clamp(min=1)
        genre_embed = embed_sum / count

        # Combine
        combined = torch.cat([text_embed, image_embed, genre_embed], dim=1)
        final_embedding = self.fusion(combined)

        return final_embedding