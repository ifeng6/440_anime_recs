from datasets import load_dataset
import numpy as np

def load_anime_data():
    # Load datasets from Hugging Face
    anime_meta = load_dataset("krishnaveni76/Animes")
    anime_ratings = load_dataset("krishnaveni76/Anime_UserRatings")

    # Data is in "train"
    anime_df = anime_meta["train"].to_pandas()
    ratings_df = anime_ratings["train"].to_pandas()

    # Clean columns
    # - Remove anime with no genre info
    anime_df = anime_df.dropna(subset=['genres'])
    ratings_df = ratings_df.dropna(subset=['genres'])

    # Some ratings are "UNKNOWN" --> replace with mean
    anime_df['average_rating'] = anime_df['average_rating'].replace('UNKNOWN', np.nan)
    anime_df['average_rating'] = anime_df['average_rating'].astype(float)
    mean_rating = anime_df['average_rating'].mean()
    anime_df['average_rating'] = anime_df['average_rating'].fillna(mean_rating)

    anime_df['overview'] = anime_df['overview'].fillna('').astype(str)
    anime_df['image url'] = anime_df['image url'].fillna('')

    # Since we dropped entries, need to reset index for cosine similarity
    anime_df = anime_df.reset_index(drop=True)
    ratings_df = ratings_df.reset_index(drop=True)

    return anime_df, ratings_df