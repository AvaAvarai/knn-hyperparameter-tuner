"""
Distance metrics for k-NN, defined by their mathematical formulae.

Each callable metric has signature (u, v) -> float for 1D arrays.
Built-in sklearn metrics are referenced by string name.
"""
import numpy as np

# Optional scipy for extra metrics
try:
    from scipy.spatial.distance import canberra, braycurtis, sqeuclidean
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False


# ---------------------------------------------------------------------------
# Custom metrics (formulae implemented here)
# ---------------------------------------------------------------------------

def squared_euclidean(u, v):
    """Squared Euclidean (L2²) distance.

    Formula: d(u,v) = Σᵢ (uᵢ - vᵢ)²
    """
    if SCIPY_AVAILABLE:
        return sqeuclidean(u, v)
    return float(np.sum((np.asarray(u) - np.asarray(v)) ** 2))


def angular(u, v):
    """Angular distance: angle between vectors in radians.

    Formula: d(u,v) = arccos( (u·v) / (‖u‖ ‖v‖) )
    Range: [0, π]. Zero vectors yield π.
    """
    u = np.asarray(u, dtype=float)
    v = np.asarray(v, dtype=float)
    nu = np.linalg.norm(u)
    nv = np.linalg.norm(v)
    if nu == 0.0 or nv == 0.0:
        return np.pi
    cos_sim = float(np.dot(u, v) / (nu * nv))
    cos_sim = np.clip(cos_sim, -1.0, 1.0)
    return float(np.arccos(cos_sim))


def levenshtein_numeric(u, v):
    """Levenshtein-style edit distance for numeric vectors (same length).

    Substitution cost = 1 if |uᵢ - vᵢ| > ε else 0.
    Insert/delete cost = 1.
    Formula: standard DP edit distance with element-wise numeric comparison.
    """
    u = np.asarray(u)
    v = np.asarray(v)
    n = len(u)
    if len(v) != n:
        raise ValueError("Vectors must have same length")
    tol = 1e-9
    dp = np.zeros((n + 1, n + 1), dtype=float)
    dp[:, 0] = np.arange(n + 1)
    dp[0, :] = np.arange(n + 1)
    for i in range(1, n + 1):
        for j in range(1, n + 1):
            sub = 0.0 if np.abs(u[i - 1] - v[j - 1]) <= tol else 1.0
            dp[i, j] = min(
                dp[i - 1, j] + 1,
                dp[i, j - 1] + 1,
                dp[i - 1, j - 1] + sub,
            )
    return float(dp[n, n])


# ---------------------------------------------------------------------------
# Metric registry: name -> (metric, params, algorithm)
# metric: string for sklearn built-in, or callable(u,v)->float
# ---------------------------------------------------------------------------

# Built-in formulae (sklearn/scipy; passed as string):
#   euclidean:    d = √(Σ(uᵢ-vᵢ)²)
#   manhattan:     d = Σ|uᵢ-vᵢ|
#   chebyshev:    d = maxᵢ|uᵢ-vᵢ|
#   hamming:      d = (1/n)Σ 1₍uᵢ≠vᵢ₎
#   cosine:       d = 1 - (u·v)/(‖u‖‖v‖)
#   correlation:  d = 1 - Pearson correlation
#   canberra:     d = Σ|uᵢ-vᵢ|/(|uᵢ|+|vᵢ|)
#   braycurtis:   d = Σ|uᵢ-vᵢ| / Σ(uᵢ+vᵢ)
#   mahalanobis:  d = √((u-v)ᵀ Σ⁻¹ (u-v))  [VI set per fold]

METRIC_NAMES = [
    "euclidean",
    "manhattan",
    "hamming",
    "cosine",
    "correlation",
    "angular",
    "chebyshev",
    "squared_euclidean",
    "canberra",
    "braycurtis",
    "levenshtein",
    "mahalanobis",
]


def get_metric_config(metric_name: str):
    """Return (metric, metric_params, algorithm) for KNeighborsClassifier.

    metric: string for sklearn built-in, or callable(u, v) -> float.
    """
    name = metric_name.lower()
    cfg = {
        "euclidean": ("euclidean", {}, "auto"),
        "manhattan": ("manhattan", {}, "auto"),
        "hamming": ("hamming", {}, "auto"),
        "cosine": ("cosine", {}, "auto"),
        "correlation": ("correlation", {}, "auto"),
        "angular": (angular, {}, "brute"),
        "chebyshev": ("chebyshev", {}, "auto"),
        "squared_euclidean": (squared_euclidean, {}, "brute"),
        "canberra": ("canberra", {}, "auto"),
        "braycurtis": ("braycurtis", {}, "auto"),
        "levenshtein": (levenshtein_numeric, {}, "brute"),
        "mahalanobis": ("mahalanobis", {}, "brute"),
    }
    return cfg.get(name, (metric_name, {}, "brute"))
