import numpy as np
import torch
from torch.utils.data import Dataset




class MovieLensDataset(Dataset):
    """
    Memory-safe Dataset that stores per-movie features as a compact matrix.
    It expects `movie_features` to be a DataFrame (indexed by movie_idx) or a numpy array
    of shape [num_movies, num_features].
    """


    def __init__(self, users, movies, ratings, movie_features):
        self.users = np.array(users)
        self.movies = np.array(movies)
        self.ratings = np.array(ratings, dtype=np.float32)


        # movie_features: if DataFrame is passed, we convert to numpy once
        if movie_features is None:
            self.movie_features = None
        else:
            # assume movie_features is a pandas DataFrame indexed by movie_idx
            try:
                self.movie_features = movie_features.values
            except Exception:
                # if it's already numpy
                self.movie_features = np.asarray(movie_features)


    def __len__(self):
        return len(self.users)


    def __getitem__(self, i):
        user = torch.tensor(self.users[i], dtype=torch.long)
        movie = torch.tensor(self.movies[i], dtype=torch.long)
        rating = torch.tensor(self.ratings[i], dtype=torch.float)


        if self.movie_features is not None:
            # movie index is an integer referencing row in movie_features
            mv_idx = int(self.movies[i])
            mv_feat = torch.tensor(self.movie_features[mv_idx], dtype=torch.float)
        else:
            mv_feat = None


        sample = {"users": user, "movies": movie, "ratings": rating}
        if mv_feat is not None:
            sample["movie_features"] = mv_feat
        return sample