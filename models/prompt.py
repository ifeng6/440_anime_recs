# models.py

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from surprise import SVD, Dataset, Reader
import torch
from tqdm import tqdm

class PromptModel:
    def __init__(self, anime_df, rating_df, n_factors=50):
        self.anime_df = anime_df
        self.rating_df = rating_df
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.bert_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.synopsis_embeddings = {}
        self.genre_embeddings = {}
        self.svd_model = None
        self.n_factors = n_factors

    def encode_anime(self):
        for idx, row in tqdm(self.anime_df.iterrows(), total=len(self.anime_df), desc="Encoding anime"):
            synopsis = row['overview']
            genres_string = row['genres']
            genres_list = [genre.strip() for genre in genres_string.split(',') if genre.strip() != '']

            self.synopsis_embeddings[row['anime_id']] = self.bert_model.encode(synopsis)

            genre_embeds = []
            for genre in genres_list:
                genre_emb = self.bert_model.encode(genre)
                genre_embeds.append(genre_emb)
            
            if genre_embeds:
                genre_embedding = np.mean(genre_embeds, axis=0)
            else:
                genre_embedding = np.zeros(self.bert_model.get_sentence_embedding_dimension())
            
            self.genre_embeddings[row['anime_id']] = genre_embedding

    def train_cf_model(self):
        reader = Reader(rating_scale=(self.rating_df['rating'].min(), self.rating_df['rating'].max()))
        data = Dataset.load_from_df(self.rating_df[['user_id', 'anime_id', 'rating']], reader)
        trainset = data.build_full_trainset()
        self.svd_model = SVD(n_factors=self.n_factors)
        self.svd_model.fit(trainset)

    def encode_prompt(self, prompt):
        return self.bert_model.encode(prompt)

    def recommend_hybrid(self, user_id, prompt_embedding, 
                        top_k=10, alpha=0.5, unseen_anime_ids=None):
        from sklearn.metrics.pairwise import cosine_similarity

        # If unseen_anime_ids are provided manually (evaluation)
        if unseen_anime_ids is None:
            user_history = self.rating_df[self.rating_df['user_id'] == user_id]['anime_id'].tolist()
            unseen_anime_df = self.anime_df[~self.anime_df['anime_id'].isin(user_history)]
        else:
            unseen_anime_df = self.anime_df[self.anime_df['anime_id'].isin(unseen_anime_ids)]

        cf_preds = []
        for _, row in unseen_anime_df.iterrows():
            anime_id = row['anime_id']
            est = self.svd_model.predict(user_id, anime_id).est
            cf_preds.append((anime_id, est))

        cf_df = pd.DataFrame(cf_preds, columns=['anime_id', 'cf_score'])

        anime_ids = unseen_anime_df['anime_id'].tolist()
        synopsis_embeds = np.array([self.synopsis_embeddings[aid] for aid in anime_ids])

        sim_scores = cosine_similarity([prompt_embedding], synopsis_embeds).flatten()
        sim_df = pd.DataFrame({'anime_id': anime_ids, 'prompt_similarity': sim_scores})

        merged = cf_df.merge(sim_df, on='anime_id', how='inner')
        merged['cf_score'] = (merged['cf_score'] - merged['cf_score'].min()) / (merged['cf_score'].max() - merged['cf_score'].min() + 1e-8)
        merged['prompt_similarity'] = (merged['prompt_similarity'] - merged['prompt_similarity'].min()) / (merged['prompt_similarity'].max() - merged['prompt_similarity'].min() + 1e-8)
        merged['final_score'] = alpha * merged['cf_score'] + (1 - alpha) * merged['prompt_similarity']

        top_recs = merged.sort_values('final_score', ascending=False).head(top_k)
        return top_recs['anime_id'].tolist()
