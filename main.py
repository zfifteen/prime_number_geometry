import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

# UNIVERSAL = 2.71828
UNIVERSAL = math.pi
# UNIVERSAL = 3


class Numberspace:
    def __init__(self, B: int):
        self._B = B
        self._C = UNIVERSAL

    @property
    def B(self) -> int:
        return self._B

    @property
    def C(self) -> float:
        return self._C

    def __call__(self, numberspace: float) -> float:
        if self._B == 0:
            raise ValueError("B cannot be zero")
        return numberspace * (self._B / self._C)

def is_prime(n):
    """Simple primality test for small numbers"""
    if n < 2:
        return False
    for i in range(2, int(n**0.5) + 1):
        if n % i == 0:
            return False
    return True

# Generate numbers and identify primes
numbers = range(1, 500)  # First 100 numbers
primes = [n for n in numbers if is_prime(n)]

# Create 3D lattice points
# X-axis: position, Y-axis: transformed value, Z-axis: helical component
points = []
for n in numbers:
    x = n
    y = Numberspace(n)(n * (n/math.pi) if n > 1 else 1)  # Log scaling
    # z = math.sin(math.pi * (math.pi ** 2) * n / (math.pi ** 3) ) # Helical component
    z = math.sin(math.pi * (31) * n / (26) ) # Helical component

    points.append((x, y, z))

# Separate primes and non-primes
x_all, y_all, z_all = zip(*points)
prime_indices = [i-1 for i in primes]  # Adjust for 0-based indexing

x_primes = [x_all[i] for i in prime_indices]
y_primes = [y_all[i] for i in prime_indices]
z_primes = [z_all[i] for i in prime_indices]

x_nonprimes = [x_all[i] for i in range(len(x_all)) if i not in prime_indices]
y_nonprimes = [y_all[i] for i in range(len(y_all)) if i not in prime_indices]
z_nonprimes = [z_all[i] for i in range(len(z_all)) if i not in prime_indices]

# Create 3D plot
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')

# Plot non-primes as blue dots
ax.scatter(x_nonprimes, y_nonprimes, z_nonprimes,
           c='blue', alpha=0.3, s=10, label='Non-primes')

# Plot primes as red stars
ax.scatter(x_primes, y_primes, z_primes,
           c='red', marker='*', s=50, label='Primes')

ax.set_xlabel('Position (n)')
ax.set_ylabel('Numberspace Value')
ax.set_zlabel('Helical Coordinate')
ax.set_title('3D Prime Geometry Visualization')
ax.legend()

plt.show()