import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

def compute_composite_macro_factor(macro_df):
    """Compute composite macro factor from all macro variables."""
    if len(macro_df) < 2:
        return np.ones(len(macro_df)) * 0.5
    scaler = StandardScaler()
    macro_scaled = scaler.fit_transform(macro_df)
    pca = PCA(n_components=1)
    factor = pca.fit_transform(macro_scaled).flatten()
    factor = (factor - factor.min()) / (factor.max() - factor.min() + 1e-8)
    return factor

def double_well_potential(x, a=1.0, b=1.0):
    """Double well potential: V(x) = a * (x^2 - b^2)^2"""
    return a * (x**2 - b**2)**2

def instanton_score(returns, macro_df, n_minima=3, barrier_scale=0.5):
    """
    Compute per-ETF instanton transition score.
    Score = distance from current state to barrier peak (normalised by width).
    Higher score = closer to barrier = more likely to tunnel.
    """
    if len(returns) < 5 or macro_df is None or len(macro_df) < 5:
        return 0.0
    # Align lengths
    min_len = min(len(returns), len(macro_df))
    returns = returns[:min_len]
    macro_df = macro_df.iloc[:min_len]
    # Remove NaN
    mask = ~(np.isnan(returns) | np.isnan(macro_df).any(axis=1))
    returns = returns[mask]
    macro_df = macro_df[mask]
    if len(returns) < 5:
        return 0.0
    # Compute macro factor
    macro_factor = compute_composite_macro_factor(macro_df)
    # Use macro factor to adjust barrier scale
    adjusted_barrier = barrier_scale * (1 + macro_factor[-1] * 0.5)
    # Standardise returns
    returns_std = np.std(returns)
    if returns_std < 1e-8:
        return 0.0
    returns_norm = returns / returns_std
    # Current position (last return)
    current_x = returns_norm[-1]
    # Barrier position (at x=0 for double well)
    barrier_x = 0.0
    # Distance from current position to barrier (normalised)
    distance_to_barrier = abs(current_x - barrier_x)
    # Width of the well (where the minima are)
    well_width = 1.0  # normalised
    # Score: how close to barrier relative to well width
    # Higher score = closer to barrier = more likely to tunnel
    score = 1.0 - min(1.0, distance_to_barrier / well_width)
    # Add a small random variation based on ticker to break ties
    # (we'll do this outside, in train.py)
    return float(score)
