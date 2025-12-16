import numpy as np
import pandas as pd
import pickle
from typing import List, Dict

SURVEY_ARTISTS = {
    # artistID: (artist_name, genre/description)
    227: ("The Beatles", "Classic Rock"),
    295: ("Beyonce", "Pop"),
    278: ("2PAC", "Hip-Hop"),
    610: ("Miles Davis", "Jazz"),
    15313: ("Daft Punk", "Electronic"),
    707: ("Metallica", "Metal"),
    1934: ("Adele", "Soul/Pop"),
    250: ("Bob Marley", "Reggae"),
    1121: ("Mozart", "Classical"),
    1042: ("Bruno Mars", "Singer-Songwriter"),
}


class RecommendationService:
    """
    Service for generating recommendations for new users based on survey responses.
    """

    def __init__(
        self,
        model_path: str,
        matrix_builder_path: str,
        item_features_path: str,
        item_features_builder_path: str,
        artist_metadata_path: str,
        rank_to_weight: float,
    ):
        """
        Initialize service by loading all pre-trained components.

        Args:
            model_path: Path to saved LightFM model (.pkl)
            matrix_builder_path: Path to saved InteractionMatrixBuilder (.pkl)
            item_features_path: Path to saved item features sparse matrix (.pkl)
            item_features_builder_path: Path to saved item features builder (.pkl)
            artist_metadata_path: Path to artist metadata CSV (artistID, artist_name)
        """
        print("Loading recommendation system components...")

        # Load model
        with open(model_path, "rb") as f:
            self.model = pickle.load(f)
        print(f"✓ Loaded model from {model_path}")

        # Load matrix builder
        with open(matrix_builder_path, "rb") as f:
            self.matrix_builder = pickle.load(f)
        print(f"✓ Loaded matrix builder from {matrix_builder_path}")

        # Load item features
        with open(item_features_path, "rb") as f:
            self.item_features = pickle.load(f)
        print(f"✓ Loaded item features from {item_features_path}")

        # Load item features builder (for metadata)
        with open(item_features_builder_path, "rb") as f:
            self.item_features_builder = pickle.load(f)
        print(f"✓ Loaded item features builder from {item_features_builder_path}")

        # Load artist metadata (artistID -> artist_name mapping)
        self.artist_metadata = pd.read_parquet(artist_metadata_path)
        self.artist_id_to_name = dict(
            zip(self.artist_metadata["artistID"], self.artist_metadata["artist_name"])
        )
        print(f"✓ Loaded artist metadata from {artist_metadata_path}")
        print(f"  Total artists in system: {len(self.artist_id_to_name)}")

        print("\n✓ Recommendation system ready!\n")

        self.RANK_TO_WEIGHT = rank_to_weight

    def get_survey_artists(self) -> Dict[int, tuple]:
        """Get the fixed survey artists for display."""
        return SURVEY_ARTISTS

    def create_user_vector_from_survey(
        self, survey_responses: Dict[int, int]
    ) -> np.ndarray:
        """
        Create user preference vector from survey rankings.

        Args:
            survey_responses: Dict mapping artistID -> rank (1-10)
                             Example: {1: 1, 2: 5, 3: 2, ...}
                             (Beatles ranked 1st, Taylor Swift 5th, etc.)

        Returns:
            Dense user vector of shape (n_artists,)
        """
        # Convert rankings to weights
        survey_interactions = []
        for artist_id, rank in survey_responses.items():
            if artist_id in SURVEY_ARTISTS:
                weight = self.RANK_TO_WEIGHT.get(rank, 0.5)
                survey_interactions.append({"artistID": artist_id, "weight": weight})

        survey_df = pd.DataFrame(survey_interactions)

        # Use InteractionMatrixBuilder to create user vector
        user_vector = self.matrix_builder.add_new_user_interactions(
            userID="new_user",  # Placeholder ID
            survey_interactions=survey_df,
        )

        return user_vector

    def get_recommendations(
        self, user_vector: np.ndarray, k: int = 3, exclude_survey_artists: bool = True
    ) -> List[Dict]:
        # Access the underlying LightFM model
        if hasattr(self.model, "model"):
            lightfm_model = self.model.model
        else:
            lightfm_model = self.model

        # Get item representations using LightFM's official API
        item_biases, item_embeddings = lightfm_model.get_item_representations(
            features=self.item_features
        )

        nonzero_indices = np.where(user_vector > 0)[0]

        if len(nonzero_indices) == 0:
            raise ValueError("User vector has no preferences!")

        # Weighted average of item embeddings based on user preferences
        user_embedding = np.average(
            item_embeddings[nonzero_indices],
            weights=user_vector[nonzero_indices],
            axis=0,
        )

        scores = np.dot(item_embeddings, user_embedding) + item_biases

        # Exclude survey artists from recommendations
        if exclude_survey_artists:
            survey_artist_ids = list(SURVEY_ARTISTS.keys())
            for artist_id in survey_artist_ids:
                # Use artist_id_map (correct attribute name)
                if artist_id in self.matrix_builder.artist_id_map:
                    idx = self.matrix_builder.artist_id_map[artist_id]
                    scores[idx] = -np.inf

        # Get top-K recommendations
        top_k_indices = np.argsort(scores)[::-1][:k]  # Sort descending, take top K

        # Map indices back to artistIDs and names
        recommendations = []
        for rank, idx in enumerate(top_k_indices, 1):
            # Use reverse_artist_map (correct attribute name)
            artist_id = self.matrix_builder.reverse_artist_map.get(idx)

            if artist_id is not None:
                artist_name = self.artist_id_to_name.get(artist_id, "Unknown Artist")

                recommendations.append(
                    {
                        "artist_id": int(artist_id),
                        "artist_name": artist_name,
                        "score": float(scores[idx]),
                        "rank": rank,
                    }
                )

        return recommendations

    def recommend_from_survey(
        self, survey_responses: Dict[int, int], k: int = 3
    ) -> List[Dict]:
        """
        Complete pipeline: survey → recommendations.

        Args:
            survey_responses: Dict mapping artistID → rank (1-10)
            k: Number of recommendations

        Returns:
            List of recommendation dicts
        """
        # Step 1: Create user vector
        user_vector = self.create_user_vector_from_survey(survey_responses)

        # Step 2: Get recommendations
        recommendations = self.get_recommendations(user_vector, k=k)

        return recommendations
