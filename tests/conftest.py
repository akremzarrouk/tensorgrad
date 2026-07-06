"""Shared pytest fixtures/configuration."""

import numpy as np
import pytest


@pytest.fixture(autouse=True)
def _seed_rng():
    """Make every test deterministic."""
    np.random.seed(0)
