"""
Cross validation utilities for recommender models.
"""

from scipy.sparse import csr_matrix
from typing import Tuple, Optional
import numpy as np
from lightfm.cross_validation import random_train_test_split as lightfm_random_train_test_split


def random_train_test_split(
    interactions: csr_matrix,
    weights: csr_matrix,
    item_features: csr_matrix,
    test_percentage: float = 0.2,
    random_state: Optional[int] = None
) -> Tuple[csr_matrix, csr_matrix, csr_matrix, csr_matrix, csr_matrix, csr_matrix]:
    """
    Randomly split user-item interactions into train and test sets using LightFM's built-in function.
    
    Parameters
    ----------
    interactions : scipy.sparse.csr_matrix
        User-item interaction matrix to split.
    weights : scipy.sparse.csr_matrix
        Weight matrix corresponding to interactions. Will be split the same way.
    item_features : scipy.sparse.csr_matrix
        Item features matrix. Will be filtered based on items in train/test splits.
    test_percentage : float, default=0.2
        Percentage of interactions to hold out for testing.
    random_state : int, optional
        Random seed for reproducible splits.
        
    Returns
    -------
    train_interactions : scipy.sparse.csr_matrix
        Training interaction matrix.
    test_interactions : scipy.sparse.csr_matrix
        Test interaction matrix.
    train_weights : scipy.sparse.csr_matrix
        Training weight matrix.
    test_weights : scipy.sparse.csr_matrix
        Test weight matrix.
    train_item_features : scipy.sparse.csr_matrix
        Item features for items in training set.
    test_item_features : scipy.sparse.csr_matrix
        Item features for items in test set.
    """
    # Split interactions
    train, test = lightfm_random_train_test_split(
        interactions=interactions,
        test_percentage=test_percentage,
        random_state=random_state
    )
    
    # Split weights using same random state
    train_weights, test_weights = lightfm_random_train_test_split(
        interactions=weights,
        test_percentage=test_percentage,
        random_state=random_state
    )
    
    # Get items present in train and test sets
    train_items = np.unique(train.nonzero()[1])
    test_items = np.unique(test.nonzero()[1])
    
    # Filter item features for train and test items
    train_item_features = item_features[train_items, :]
    test_item_features = item_features[test_items, :]
    
    return train, test, train_weights, test_weights, train_item_features, test_item_features