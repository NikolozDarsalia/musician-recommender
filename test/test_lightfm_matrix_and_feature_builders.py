import pandas as pd
import numpy as np

from recommender_pipeline.interaction_matrix_builder import InteractionMatrixBuilder
from recommender_pipeline.artist_feature_builder import ArtistFeaturesBuilder


def test_interaction_matrix_builder():
    """Test 1: Basic matrix building and cold-start user handling"""
    print("TEST 1: InteractionMatrixBuilder - Basic functionality")

    # Create sample data
    train_data = pd.DataFrame(
        {
            "userID": ["u1", "u1", "u2", "u2"],
            "artistID": ["a1", "a2", "a1", "a3"],
            "weight": [0.8, 0.6, 0.9, 0.7],
        }
    )

    test_data = pd.DataFrame(
        {
            "userID": ["u1", "u2"],
            "artistID": ["a3", "a2"],
            "weight": [0.5, 0.4],
        }
    )

    val_data = pd.DataFrame({"userID": ["u1"], "artistID": ["a2"], "weight": [0.3]})

    # Build matrices
    builder = InteractionMatrixBuilder()
    builder.fit(train_data, test_data, val_data)

    train_matrix = builder.transform(train_data)

    # Assertions
    assert train_matrix.shape == (2, 3), f"Expected (2, 3), got {train_matrix.shape}"
    assert builder.n_users == 2, f"Expected 2 users, got {builder.n_users}"
    assert builder.n_artists == 3, f"Expected 3 artists, got {builder.n_artists}"
    assert train_matrix[0, 0] == 0.8, f"Expected 0.8, got {train_matrix[0, 0]}"

    print("✓ Matrix building works correctly")
    print(f"  Train matrix shape: {train_matrix.shape}")
    print(
        f"  Matrix sparsity: {train_matrix.nnz}/{train_matrix.shape[0] * train_matrix.shape[1]}"
    )
    print()


def test_cold_start_user():
    """Test 2: Cold-start user interaction creation"""
    print("TEST 2: InteractionMatrixBuilder - Cold-start user handling")

    # Setup
    train_data = pd.DataFrame(
        {
            "userID": ["u1", "u1"],
            "artistID": ["a1", "a2"],
            "weight": [0.8, 0.6],
        }
    )

    builder = InteractionMatrixBuilder()
    builder.fit(train_data, train_data, train_data)

    # Simulate new user survey
    new_user_survey = pd.DataFrame({"artistID": ["a1", "a2"], "weight": [0.9, 0.7]})

    # Create interaction vector for new user
    new_user_vector = builder.add_new_user_interactions("u_new", new_user_survey)

    # Assertions
    assert new_user_vector.shape == (2,), f"Expected (2,), got {new_user_vector.shape}"
    assert new_user_vector[0] == 0.9, f"Expected 0.9, got {new_user_vector[0]}"
    assert new_user_vector[1] == 0.7, f"Expected 0.7, got {new_user_vector[1]}"

    print("✓ Cold-start user vector created successfully")
    print(f"  New user vector: {new_user_vector}")
    print(f"  Non-zero interactions: {np.count_nonzero(new_user_vector)}")
    print()


def test_artist_features_builder():
    """Test 3: Artist features matrix building"""
    print("TEST 3: ArtistFeaturesBuilder - Feature matrix creation")

    # Create sample artist metadata
    spotify_new = pd.DataFrame(
        {
            "feature1": [0.5, 0.8, 0.3],
            "feature2": [0.7, 0.4, 0.9],
            "feature3": [0.2, 0.6, 0.1],
        },
        index=["a1", "a2", "a3"],
    )

    artist_id_map = {"a1": 0, "a2": 1, "a3": 2}

    # Build features
    builder = ArtistFeaturesBuilder()
    features_matrix = builder.fit_transform(spotify_new, artist_id_map)

    # Assertions
    assert features_matrix.shape == (3, 3), (
        f"Expected (3, 3), got {features_matrix.shape}"
    )
    assert builder.feature_names == ["feature1", "feature2", "feature3"]
    assert features_matrix[0, 0] == 0.5, f"Expected 0.5, got {features_matrix[0, 0]}"

    print("✓ Artist features matrix built successfully")
    print(f"  Features matrix shape: {features_matrix.shape}")
    print(f"  Feature names: {builder.feature_names}")
    print()


def test_artist_features_alignment():
    """Test 4: Artist features alignment with interaction mappings"""
    print("TEST 4: ArtistFeaturesBuilder - Alignment with mappings")

    # Artist metadata with extra artist not in interactions
    spotify_new = pd.DataFrame(
        {"feature1": [0.5, 0.8], "feature2": [0.7, 0.4]}, index=["a1", "a2"]
    )

    # Interaction mapping includes a3, but not in metadata
    artist_id_map = {"a1": 0, "a2": 1, "a3": 2}

    builder = ArtistFeaturesBuilder()
    features_matrix = builder.fit_transform(spotify_new, artist_id_map)

    # Should create matrix with all 3 artists, but a3 will have zero features
    assert features_matrix.shape == (3, 2), (
        f"Expected (3, 2), got {features_matrix.shape}"
    )
    assert np.sum(features_matrix[2, :]) == 0, "Artist a3 should have zero features"

    print("✓ Feature alignment handles missing artists correctly")
    print(
        f"  Artist 'a3' (not in metadata) has zero features: {features_matrix[2, :].toarray()}"
    )
    print()
