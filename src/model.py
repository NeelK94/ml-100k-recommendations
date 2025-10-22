import torch
import torch.nn as nn


class RecommendationSystemModel(nn.Module):
    def __init__(self, num_users: int, num_movies: int, num_movie_features: int,
                 embedding_size: int = 64, hidden_dim: int = 128, dropout_rate: float = 0.1):
        super().__init__()

        self.user_embedding = nn.Embedding(num_users, embedding_size)
        self.movie_embedding = nn.Embedding(num_movies, embedding_size)

        # Project dense movie features (genres + TF-IDF tags) into embedding space
        self.movie_feat_fc = nn.Linear(num_movie_features, embedding_size)

        self.fc1 = nn.Linear(embedding_size * 3, hidden_dim)
        #self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.fc3 = nn.Linear(hidden_dim, 1)

        self.dropout = nn.Dropout(dropout_rate)
        self.relu = nn.ReLU()

    def forward(self, users: torch.Tensor, movies: torch.Tensor, movie_features: torch.Tensor):
        u = self.user_embedding(users)
        m = self.movie_embedding(movies)
        f = self.relu(self.movie_feat_fc(movie_features))
        x = torch.cat([u, m, f], dim=1)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        #x = self.relu(self.fc2(x))
        #x = self.dropout(x)
        out = self.fc3(x)
        return out
