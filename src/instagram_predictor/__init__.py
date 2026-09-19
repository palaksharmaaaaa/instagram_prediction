"""
Reach Forecaster.

Only real, observed data is ever used or shown. Forecasts come from a model trained on real
post-level data supplied by the user, are compared against a simple baseline on held-out creators,
carry conformal intervals whose measured coverage is reported, and are refused outside the range
of the training data.
"""

__version__ = "3.0.0"
