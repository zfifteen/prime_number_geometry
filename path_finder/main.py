import math
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import KDTree
from mpl_toolkits.mplot3d import Axes3D

# ----------------------------
# Core classes & functions
# ----------------------------
UNIVERSAL = math.e

class Numberspace:
    def __init__(self, B: float, C: float = UNIVERSAL):
        if B == 0:
            raise ValueError("B cannot be zero")
        self._B = B
        self._C = C

    def __call__(self, value: float) -> float:
        return value * (self._B / self._C)

def is_prime(n: int) -> bool:
    if n < 2:
        return False
    r = int(math.isqrt(n))
    for i in range(2, r + 1):
        if n % i == 0:
            return False
    return True

# ----------------------------
# Objective: measure prime-density
# ----------------------------
def prime_density_score(B: float, helix_freq: float, N_POINTS: int):
    """
    Embed 1..N_POINTS into 3D via Numberspace(B) and helix_freq,
    compute average nearest-neighbor distance among primes,
    return density score = 1 / mean_distance.
    """
    n = np.arange(1, N_POINTS + 1)
    primality = np.vectorize(is_prime)(n)

    # choose y_raw = n * (n/pi)
    y_raw = n * (n / math.pi)
    transformer = Numberspace(B=B)
    y = transformer(y_raw)

    z = np.sin(math.pi * helix_freq * n)

    # gather prime positions
    pts = np.vstack((
        n[primality],
        y[primality],
        z[primality]
    )).T
    if len(pts) < 2:
        return 0.0  # no primes or single prime

    tree = KDTree(pts)
    # k=2 => first neighbor is itself (distance=0), second is nearest other prime
    dists, _ = tree.query(pts, k=2)
    mean_nn = np.mean(dists[:,1])
    return 1.0 / mean_nn if mean_nn > 0 else 0.0

# ----------------------------
# Main optimization & plotting
# ----------------------------
def main():
    # parameters
    N_POINTS     = 2000
    N_CANDIDATES = 100   # number of (B, freq) samples
    TOP_K        = 10

    # random sample B in [e/2, 2e], helix_freq in [0.05, 0.2]
    np.random.seed(42)
    Bs   = np.random.uniform(math.e/2, 2*math.e, size=N_CANDIDATES)
    FREQ = np.random.uniform(0.05, 0.2, size=N_CANDIDATES)

    # evaluate scores
    results = []
    for B, f in zip(Bs, FREQ):
        score = prime_density_score(B, f, N_POINTS)
        results.append({'B': B, 'freq': f, 'score': score})

    # pick top K
    topk = sorted(results, key=lambda x: x['score'], reverse=True)[:TOP_K]

    # bar chart of top-K scores
    labels = [f"B={r['B']:.2f}\nf={r['freq']:.3f}" for r in topk]
    scores = [r['score'] for r in topk]

    plt.figure(figsize=(12, 5))
    plt.bar(range(TOP_K), scores, color='teal', alpha=0.7)
    plt.xticks(range(TOP_K), labels, rotation=45, ha='right')
    plt.ylabel("Prime-Density Score (1/mean NN dist)")
    plt.title("Top 10 Parameter Sets by Prime-Density Score")
    plt.tight_layout()
    plt.show()

    # detailed 3D scatter for each top parameter set
    n = np.arange(1, N_POINTS + 1)
    primality = np.vectorize(is_prime)(n)
    y_raw = n * (n / math.pi)

    for idx, params in enumerate(topk, 1):
        B, f = params['B'], params['freq']
        transformer = Numberspace(B=B)
        y = transformer(y_raw)
        z = np.sin(math.pi * f * n)

        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(n[~primality], y[~primality], z[~primality],
                   c='blue', alpha=0.2, s=8)
        ax.scatter(n[primality], y[primality], z[primality],
                   c='red', marker='*', s=50)
        ax.set_title(f"#{idx}: B={B:.2f}, freq={f:.3f}, score={params['score']:.4f}")
        ax.set_xlabel('n (Position)')
        ax.set_ylabel('Transformed y')
        ax.set_zlabel('Helix z')
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    main()
