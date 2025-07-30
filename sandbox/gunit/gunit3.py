import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from main import is_prime  # Assume from your documents

N = 2000
n = np.arange(1, N+1)
primality = np.vectorize(is_prime)(n)
primes = n[primality]

# Gaps and Z scores
gaps = np.diff(primes)
delta_max = max(gaps)
z_scores = [primes[i] * (gaps[i-1] / delta_max) if i > 0 else 0 for i in range(len(primes))]

# Helical from main.py
HELIX_FREQ = 0.1003033
x = n
y = np.log(n + 1)  # Simplified scaling
z_helical = np.sin(np.pi * HELIX_FREQ * n)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(x[~primality], y[~primality], z_helical[~primality], c='blue', alpha=0.3, label='Non-primes')
ax.scatter(primes[:-1], np.log(primes[:-1] + 1), z_scores, c='red', marker='*', s=50, label='Primes with Z height')
ax.set_xlabel('n')
ax.set_ylabel('log(n)')
ax.set_zlabel('Helical / Z Score')
ax.legend()
plt.show()