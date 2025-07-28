import math
import numpy as np
import matplotlib.pyplot as plt
import scipy
from mpl_toolkits.mplot3d import Axes3D

# Choose your universal constant
UNIVERSAL = 360   # or math.e, or 2.71828

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
    for i in range(2, int(math.sqrt(n)) + 1):
        if n % i == 0:
            return False
    return True

# Parameters
N_POINTS = 5000


# HELIX_FREQ = scipy.constants.golden_ratio   # you can tweak this ratio
HELIX_FREQ = 17  # you can tweak this ratio

LOG_SCALE = False     # toggle for log scaling on Y-axis

# Instantiate your transformer with a fixed B
transformer = Numberspace(B=math.e)  # e.g. scale factor base = 2

# Generate data
n = np.arange(1, N_POINTS)
primality = np.vectorize(is_prime)(n)

# Y-values: choose raw, log, or polynomial
if LOG_SCALE:
    y_raw = np.log(n, where=(n>1), out=np.zeros_like(n, dtype=float))
else:
    y_raw = n * (n / math.pi)

# Apply your Numberspace transform
y = transformer(y_raw)

# Z-values for the helix
z = np.sin( math.pi * HELIX_FREQ * n )

# Split into primes vs non-primes
x_primes   = n[primality]
y_primes   = y[primality]
z_primes   = z[primality]

x_nonprimes = n[~primality]
y_nonprimes = y[~primality]
z_nonprimes = z[~primality]

# Plot
fig = plt.figure(figsize=(12, 8))
ax  = fig.add_subplot(111, projection='3d')

ax.scatter(x_nonprimes, y_nonprimes, z_nonprimes,
           c='blue', alpha=0.3, s=10, label='Non-primes')

ax.scatter(x_primes, y_primes, z_primes,
           c='red', marker='*', s=50, label='Primes')

ax.set_xlabel('n (Position)')
ax.set_ylabel('Scaled Value')
ax.set_zlabel('Helical Coord')
ax.set_title('3D Prime Geometry Visualization')
ax.legend()

# If you want a log‐scaled axis instead of pre‐transform:
# ax.set_yscale('log')

plt.tight_layout()
plt.show()
