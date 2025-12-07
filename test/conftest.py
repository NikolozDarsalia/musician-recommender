"""
Pytest configuration and shared fixtures.
"""

import pytest
import matplotlib
import matplotlib.pyplot as plt
import warnings

# Use non-interactive backend for matplotlib during testing
matplotlib.use('Agg')

# Configure matplotlib to avoid memory warnings
plt.rcParams['figure.max_open_warning'] = 0

# Suppress warnings during testing
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=RuntimeWarning)


@pytest.fixture(autouse=True)
def clean_matplotlib():
    """Automatically clean up matplotlib figures after each test."""
    # Close any existing figures before test
    plt.close('all')
    yield
    # Close any figures created during test
    plt.close('all')