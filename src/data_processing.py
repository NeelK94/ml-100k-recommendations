"""
Data processing utilities.

This module loads raw MovieLens CSVs, encodes user/movie IDs, builds
per-movie feature matrices from genres (one-hot) and tags (TF-IDF), and
creates train/validation splits compatible with the PyTorch Dataset.
"""
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

from config import (
    ROOTS,
    RATINGS_CSV,
    MOVIES_CSV,
    TAGS_CSV,
    ENCODERS_DIR,
    DATA_SOURCE,
    TFIDF_MAX_FEATURES,
)


def set_data_root(data_source: str) -> Path:
    root = ROOTS.get(data_source)
    if root is None:
        raise ValueError(f"data_source must be one of {list(ROOTS.keys())}")
    return Path(root)


def load_raw_files(root: Path):
    ratings_path = root / RATINGS_CSV
    movies_path = root / MOVIES_CSV
    tags_path = root / TAGS_CSV

    ratings_df = pd.read_csv(ratings_path)
    movies_df = pd.read_csv(movies_path)
    tags_df = pd.read_csv(tags_path)
    
    return ratings_df, movies_df, tags_df


def encode_ids(ratings_df: pd.DataFrame, movies_df: pd.DataFrame, save_encoders: bool = True):
    user_encoder = preprocessing.LabelEncoder()
    movie_encoder = preprocessing.LabelEncoder()

    ratings_df = ratings_df.copy()
    movies_df = movies_df.copy()

    ratings_df["user_idx"] = user_encoder.fit_transform(ratings_df["userId"].values)
    # Fit movie encoder on the superset of movieIds present in movies_df, then transform ratings
    movie_encoder.fit(movies_df["movieId"].values)
    ratings_df["movie_idx"] = movie_encoder.transform(ratings_df["movieId"].values)

    # Map movies_df to the same movie_idx space
    movies_df["movie_idx"] = movie_encoder.transform(movies_df["movieId"].values)

    if save_encoders:
        ENCODERS_DIR.mkdir(parents=True, exist_ok=True)
        joblib.dump(user_encoder, ENCODERS_DIR / "user_encoder.joblib")
        joblib.dump(movie_encoder, ENCODERS_DIR / "movie_encoder.joblib")

    return ratings_df, movies_df, user_encoder, movie_encoder


def build_genre_features(movies_df: pd.DataFrame):
    movies_df = movies_df.copy()
    movies_df["genres_list"] = movies_df["genres"].fillna("(no genres)").apply(lambda x: x.split("|"))
    all_genres = sorted({g for sublist in movies_df["genres_list"] for g in sublist})
    for genre in all_genres:
        movies_df[f"genre__{genre}"] = movies_df["genres_list"].apply(lambda xs: int(genre in xs))
    genre_cols = [c for c in movies_df.columns if c.startswith("genre__")]
    return movies_df, genre_cols


def build_tag_features(movies_df: pd.DataFrame, tags_df: pd.DataFrame, tfidf_max_features: int):
    if tags_df is None or tags_df.empty:
        # Return empty features aligned by movie_idx
        empty = pd.DataFrame(index=movies_df.set_index("movie_idx").index)
        return empty, []

    tags_df = tags_df[["movieId", "tag"]].dropna(subset=["tag"]).copy()
    tags_df["tag"] = tags_df["tag"].astype(str).str.lower().str.strip()
    tags_df = tags_df.drop_duplicates()

    # Aggregate tags per movieId
    movie_tags = (
        tags_df.groupby("movieId")["tag"].unique().reset_index()
    )
    movie_tags["tags"] = movie_tags["tag"].apply(lambda arr: " ".join(arr))
    movie_tags = movie_tags.drop(columns=["tag"])  # keep only joined string

    # Attach movie_idx for alignment
    movies_map = movies_df[["movieId", "movie_idx"]]
    movie_tags = movie_tags.merge(movies_map, on="movieId", how="inner")

    # TF-IDF
    tfidf = TfidfVectorizer(max_features=tfidf_max_features, stop_words="english")
    tfidf_matrix = tfidf.fit_transform(movie_tags["tags"])
    tfidf_features = pd.DataFrame(
        tfidf_matrix.toarray(),
        index=movie_tags["movie_idx"].values,
        columns=[f"tag__{t}" for t in tfidf.get_feature_names_out()],
    ).sort_index()

    tag_cols = list(tfidf_features.columns)
    return tfidf_features, tag_cols


def build_movie_features(movies_df: pd.DataFrame, tags_df: pd.DataFrame, tfidf_max_features: int):
    movies_df_with_genres, genre_cols = build_genre_features(movies_df)
    tfidf_features, tag_cols = build_tag_features(movies_df_with_genres, tags_df, tfidf_max_features)

    # Align indices on movie_idx
    base = movies_df_with_genres.set_index("movie_idx")
    parts = []
    if genre_cols:
        parts.append(base[genre_cols])
    if not tfidf_features.empty:
        # Ensure same index type and order
        tfidf_features = tfidf_features.reindex(base.index).fillna(0.0)
        parts.append(tfidf_features)

    if parts:
        movies_features = pd.concat(parts, axis=1)
    else:
        # No features available; create zero columns
        movies_features = pd.DataFrame(index=base.index)

    num_movie_features = movies_features.shape[1]
    return movies_features, num_movie_features, genre_cols, tag_cols


def load_data_splits(data_source: str = None, test_size: float = 0.2, random_state: int = 42):
    if data_source is None:
        data_source = DATA_SOURCE
    root = set_data_root(data_source)
    ratings_df, movies_df, tags_df = load_raw_files(root)
    ratings_df, movies_df, _, _ = encode_ids(ratings_df, movies_df, save_encoders=True)

    # Stratify by rating values to mimic notebook
    train_df, val_df = train_test_split(
        ratings_df, test_size=test_size, stratify=ratings_df["rating"].values, random_state=random_state
    )
    return train_df.reset_index(drop=True), val_df.reset_index(drop=True), movies_df, tags_df


def load_movie_features(data_source: str = None, tfidf_max_features: int = None):
    if data_source is None:
        data_source = DATA_SOURCE
    if tfidf_max_features is None:
        tfidf_max_features = TFIDF_MAX_FEATURES

    root = set_data_root(data_source)
    ratings_df, movies_df, tags_df = load_raw_files(root)
    ratings_df, movies_df, _, _ = encode_ids(ratings_df, movies_df, save_encoders=True)
    movies_features, num_movie_features, genre_cols, tag_cols = build_movie_features(
        movies_df, tags_df, tfidf_max_features
    )
    num_users = int(ratings_df["user_idx"].nunique())
    num_movies = int(ratings_df["movie_idx"].nunique())
    return movies_features, num_movie_features, num_users, num_movies