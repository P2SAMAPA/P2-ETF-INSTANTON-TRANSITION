import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import erf
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
    """
    Double well potential: V(x) = a * (x^2 - b^2)^2
    Local minima at x = ±b, barrier at x = 0.
    """
    return a * (x**2 - b**2)**2

def instanton_action(returns, macro_factor, barrier_scale=0.5):
    """
    Compute the instanton action S = ∫ sqrt(2 * (V(x) - E)) dx,
    where E is the energy of the system.
    """
    # Fit a double-well potential to the returns distribution
    if len(returns) < 5:
        return 0.0
    # Estimate parameters from data
    std = np.std(returns)
    mean = np.mean(returns)
    # Scale parameters
    b = barrier_scale * std  # location of minima
    a = 1.0 / (b**4)  # scale such that barrier height ~ 1
    # Find the maximum of the potential (barrier)
    # The barrier is at x = 0, so V(0) = a * b^4
    V_barrier = a * b**4
    # Current state energy (using the last return)
    current_x = returns[-1] - mean
    # Energy = V(current_x) (assuming zero kinetic energy)
    E = double_well_potential(current_x, a, b)
    # If energy is above barrier, no tunneling (instanton not valid)
    if E >= V_barrier:
        return 0.0
    # Compute action: S = ∫_{x1}^{x2} sqrt(2 * (V(x) - E)) dx
    # Numerically integrate using the turning points
    # Turning points: V(x) = E
    # This is a quartic equation; we can find roots numerically
    # For simplicity, we'll approximate the action using a Gaussian approximation
    # near the barrier peak
    # S ≈ (π/2) * sqrt(2 * V_barrier) * exp(-∫_{0}^{x_t} sqrt(2*(V-E)) dx)
    # This is complex. We'll use a simplified heuristic:
    # Action is proportional to (V_barrier - E) / barrier_scale
    action = (V_barrier - E) / (barrier_scale * std + 1e-8)
    return max(0.0, min(10.0, action))

def instanton_score(returns, macro_df, n_minima=3, barrier_scale=0.5):
    """
    Compute per-ETF instanton transition score.
    Higher score = higher tunneling amplitude (likely regime shift).
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
    # Compute instanton action
    action = instanton_action(returns, macro_factor, barrier_scale)
    # Transition amplitude = exp(-S) / (1 + exp(-S))
    # This is the probability of tunneling
    amplitude = np.exp(-action) / (1 + np.exp(-action))
    return float(amplitude)
