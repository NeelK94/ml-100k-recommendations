import torch
import sys
from torch import nn, optim
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from config import (
    BATCH_SIZE,
    LR,
    EPOCHS,
    DATA_SOURCE,
    DROPOUT,
    EMBEDDING_SIZE,
    NUM_WORKERS,
    HIDDEN_DIM,
)
from dataset import MovieLensDataset
from model import RecommendationSystemModel
from data_processing import load_data_splits, load_movie_features


def get_device() -> torch.device:
    if hasattr(torch, "accelerator") and torch.accelerator.is_available():
        acc = torch.accelerator.current_accelerator().type
        return torch.device(acc)
    if torch.cuda.is_available():
        return torch.device("cuda")
    try:
        if torch.backends.mps.is_available():
            return torch.device("mps")
    except Exception:
        pass
    return torch.device("cpu")


def train():
    device = get_device()
    print(f"Using {device} device")
    print(f"Loading {DATA_SOURCE} dataset...")

    # Load data splits and movie features
    train_df, val_df, movies_df, tags_df = load_data_splits(DATA_SOURCE)
    movies_features, num_movie_features, num_users, num_movies = load_movie_features(DATA_SOURCE)

    # Datasets and loaders
    train_dataset = MovieLensDataset(
        users=train_df["user_idx"].values,
        movies=train_df["movie_idx"].values,
        ratings=train_df["rating"].values,
        movie_features=movies_features,
    )
    val_dataset = MovieLensDataset(
        users=val_df["user_idx"].values,
        movies=val_df["movie_idx"].values,
        ratings=val_df["rating"].values,
        movie_features=movies_features,
    )

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

    # Model
    model = RecommendationSystemModel(
        num_users=num_users,
        num_movies=num_movies,
        num_movie_features=num_movie_features,
        embedding_size=EMBEDDING_SIZE,
        hidden_dim=HIDDEN_DIM,
        dropout_rate=DROPOUT,
    ).to(device)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3)

    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        for batch in tqdm(
            train_loader,
            desc=f"Train {epoch+1}/{EPOCHS}",
            leave=True,
            dynamic_ncols=True,
            disable=not sys.stdout.isatty(),
        ):
            users = batch["users"].to(device)
            movies = batch["movies"].to(device)
            ratings = batch["ratings"].to(device)
            movie_features_batch = batch.get("movie_features")
            if movie_features_batch is not None:
                movie_features_batch = movie_features_batch.to(device)

            optimizer.zero_grad()
            preds = model(users, movies, movie_features_batch)
            loss = criterion(preds.squeeze(), ratings)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * users.size(0)

        train_loss = running_loss / len(train_loader.dataset)

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                users = batch["users"].to(device)
                movies = batch["movies"].to(device)
                ratings = batch["ratings"].to(device)
                movie_features_batch = batch.get("movie_features")
                if movie_features_batch is not None:
                    movie_features_batch = movie_features_batch.to(device)

                preds = model(users, movies, movie_features_batch)
                loss = criterion(preds.squeeze(), ratings)
                val_loss += loss.item() * users.size(0)

        val_loss = val_loss / len(val_loader.dataset)
        scheduler.step(val_loss)

        print(f"Epoch [{epoch+1}/{EPOCHS}] | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")


if __name__ == "__main__":
    train()
