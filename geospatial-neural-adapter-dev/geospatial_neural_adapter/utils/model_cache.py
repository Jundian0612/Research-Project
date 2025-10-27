"""
Model caching utilities for hyperparameter optimization.

This module provides caching mechanisms for model weights to enable
efficient warm-starting between trials during hyperparameter optimization.
"""

import math
from typing import Dict, Tuple

from torch import nn


class ModelCache:
    """
    Cache for model weights to enable warm-starting between trials.

    This is particularly useful for hyperparameter optimization where
    similar parameter combinations can benefit from warm-starting with
    previously trained model weights.
    """

    def __init__(self):
        self.cache: Dict[Tuple[float, float], Tuple[Dict, Dict]] = {}

    def store(
        self, tau1: float, tau2: float, trend_state: Dict, basis_state: Dict
    ) -> None:
        """
        Store model weights for given tau1, tau2 parameters.

        Args:
            tau1: First regularization parameter
            tau2: Second regularization parameter
            trend_state: Trend model state dict
            basis_state: Basis model state dict
        """
        self.cache[(tau1, tau2)] = (trend_state, basis_state)

    def load_nearest(
        self, trend: nn.Module, basis: nn.Module, tau1: float, tau2: float
    ) -> None:
        """
        Load nearest cached weights for warm-starting.

        Finds the closest tau1, tau2 combination in the cache and loads
        the corresponding model weights. Uses log-space distance for
        better matching across different scales.

        Args:
            trend: Trend model to load weights into
            basis: Basis model to load weights into
            tau1: First regularization parameter
            tau2: Second regularization parameter
        """
        if not self.cache:
            return

        logt1, logt2 = math.log10(tau1 + 1e-12), math.log10(tau2 + 1e-12)
        key_best = min(
            self.cache,
            key=lambda k: (math.log10(k[0] + 1e-12) - logt1) ** 2
            + (math.log10(k[1] + 1e-12) - logt2) ** 2,
        )
        sd_t, sd_b = self.cache[key_best]
        trend.load_state_dict(sd_t, strict=False)
        basis.load_state_dict(sd_b, strict=False)

    def clear(self) -> None:
        """Clear the cache to free memory."""
        self.cache.clear()

    def size(self) -> int:
        """Get the number of cached models."""
        return len(self.cache)

    def keys(self) -> list:
        """Get all cached parameter combinations."""
        return list(self.cache.keys())
