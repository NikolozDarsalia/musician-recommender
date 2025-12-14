import os
import joblib
import numpy as np
from scipy.sparse import csr_matrix, coo_matrix
from typing import Tuple, Optional, Union
from ..interfaces.base_recommender import BaseRecommender
from lightfm import LightFM

class LightFMRecommender(BaseRecommender):
    def __init__(
        self,
        lightfm_params=None,
        epochs=20,
        num_threads=1,
        model_path="lightfm_basemodel.pkl",
        artist_mapping=None,
    ):
        # Default LightFM parameters
        default_params = {
            'no_components': 10,
            'k': 5,
            'n': 10,
            'learning_schedule': 'adagrad',
            'loss': 'logistic',
            'learning_rate': 0.05,
            'rho': 0.95,
            'epsilon': 1e-06,
            'item_alpha': 0.0,
            'user_alpha': 0.0,
            'max_sampled': 10,
            'random_state': None
        }
        
        # Update with user-provided parameters
        if lightfm_params is None:
            lightfm_params = {}
        
        self.lightfm_params = {**default_params, **lightfm_params}
        self.epochs = epochs
        self.num_threads = num_threads
        self.model_path = model_path
        self.artist_mapping = artist_mapping  # Dictionary mapping artist names to {'global_id': id, 'genre': genre}
        
        # Initialize LightFM model with parameters
        self.model = LightFM(**self.lightfm_params)

    def set_artist_mapping(self, artist_mapping):
        """
        Set or update the artist mapping dictionary.
        
        Parameters
        ----------
        artist_mapping : dict
            Dictionary mapping artist names to {'global_id': id, 'genre': genre}.
            Example: {'The Beatles': {'global_id': 123, 'genre': 'Rock'}}
        """
        self.artist_mapping = artist_mapping
        return self

    def fit(self, interactions, user_features=None, item_features=None, sample_weight=None):
        """
        Fit the LightFM model to the provided interactions and features.
        """
        # Store interactions for similarity calculations
        self.interactions = interactions
        self.sample_weight = sample_weight
        
        self.model.fit(
            interactions,
            user_features=user_features,
            item_features=item_features,
            sample_weight=sample_weight,
            epochs=self.epochs,
            num_threads=self.num_threads,
        )
        return self

    def _find_similar_users(self, artists=None, genres=None, k=10):
        """
        Hidden method to find users similar to given artists or genres.
        
        Parameters
        ----------
        artists : list of str, optional
            List of artist names (max 5)
        genres : list of str, optional  
            List of genre names (max 5)
        k : int, default=10
            Number of similar users to return
            
        Returns
        -------
        user_ids : np.ndarray
            Array of user IDs most similar to given artists/genres
        """
        if not hasattr(self, 'interactions'):
            raise ValueError("Model must be fitted first to find similar users")
            
        if artists is None and genres is None:
            raise ValueError("At least one of 'artists' or 'genres' must be provided")
            
        # Validate input lengths
        if artists is not None and len(artists) > 5:
            raise ValueError("Maximum 5 artists allowed")
        if genres is not None and len(genres) > 5:
            raise ValueError("Maximum 5 genres allowed")
            
        interactions_csr = self.interactions.tocsr()
        weights = self.sample_weight.tocsr() if self.sample_weight is not None else interactions_csr
        
        target_items = set()
        
        # Convert artist names to item IDs if provided
        if artists is not None and self.artist_mapping is not None:
            for artist in artists:
                if artist in self.artist_mapping:
                    # Extract global_id from the mapping structure
                    artist_info = self.artist_mapping[artist]
                    if isinstance(artist_info, dict) and 'global_id' in artist_info:
                        target_items.add(artist_info['global_id'])
                    else:
                        # Backward compatibility for simple id mapping
                        target_items.add(artist_info)
                    
        # Add genre-based item mapping
        if genres is not None and self.artist_mapping is not None:
            for genre in genres:
                # Find all artists with matching genre
                for artist_name, artist_info in self.artist_mapping.items():
                    if isinstance(artist_info, dict):
                        artist_genre = artist_info.get('genre', '').lower()
                        if artist_genre == genre.lower():
                            target_items.add(artist_info['global_id'])
                    # Skip backward compatibility case for genre search
        
        if not target_items:
            raise ValueError("No matching items found for provided artists/genres")
            
        # Find users whose highest weighted item is in target_items
        similar_users = []
        
        for user_idx in range(interactions_csr.shape[0]):
            user_weights = weights[user_idx].toarray().flatten()
            if np.any(user_weights > 0):  # User has interactions
                top_item = np.argmax(user_weights)
                if top_item in target_items:
                    # Store (user_id, max_weight) for ranking
                    similar_users.append((user_idx, user_weights[top_item]))
        
        # Sort by weight and return top k users
        similar_users.sort(key=lambda x: x[1], reverse=True)
        return np.array([user_id for user_id, _ in similar_users[:k]])

    def predict(self, user_ids, item_ids, user_features=None, item_features=None):
        return self.model.predict(user_ids, item_ids, user_features, item_features)

    def recommend(self, user_id=None, artists=None, genres=None, k=10, similar_users_k=10):
        """
        Generate recommendations based on user ID or similar users to given artists/genres.
        
        Parameters
        ----------
        user_id : int, optional
            Specific user ID to get recommendations for
        artists : list of str, optional
            List of artist names to find similar users (max 5)
        genres : list of str, optional
            List of genre names to find similar users (max 5)
        k : int, default=10
            Number of recommendations to return
        similar_users_k : int, default=10
            Number of similar users to consider when using artists/genres
            
        Returns
        -------
        recommendations : np.ndarray
            Array of recommended item IDs
        """
        n_items = self.model.item_embeddings.shape[0]
        all_items = np.arange(n_items)
        
        if user_id is not None:
            # Traditional single-user recommendation
            scores = self.model.predict(user_id, all_items)
            top_k = np.argsort(-scores)[:k]
            return top_k
        
        elif artists is not None or genres is not None:
            # Find similar users and aggregate their predictions
            similar_user_ids = self._find_similar_users(artists=artists, genres=genres, k=similar_users_k)
            
            if len(similar_user_ids) == 0:
                raise ValueError("No similar users found for given artists/genres")
                
            # Aggregate scores from similar users
            aggregated_scores = np.zeros(n_items)
            
            for similar_user in similar_user_ids:
                user_scores = self.model.predict(similar_user, all_items)
                aggregated_scores += user_scores
                
            # Average the scores
            aggregated_scores /= len(similar_user_ids)
            
            top_k = np.argsort(-aggregated_scores)[:k]
            return top_k
            
        else:
            raise ValueError("Either 'user_id' or at least one of 'artists'/'genres' must be provided")

    def save(self):
        """
        Save the trained LightFM model and parameters to disk using joblib.
        """
        joblib.dump(
            {
                "model": self.model,
                "lightfm_params": self.lightfm_params,
                "epochs": self.epochs,
                "num_threads": self.num_threads,
                "artist_mapping": self.artist_mapping,
            },
            self.model_path,
        )

    def load(self):
        """
        Load the LightFM model and parameters from disk using joblib.
        """
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"No saved model found at {self.model_path}")

        data = joblib.load(self.model_path)
        self.model = data["model"]
        self.lightfm_params = data["lightfm_params"]
        self.epochs = data["epochs"]
        self.num_threads = data["num_threads"]
        self.artist_mapping = data.get("artist_mapping", None)  # Handle backward compatibility
        return self
