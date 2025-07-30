from mpmath import mp, mpf, gamma, pi, zeta, quad, cos, log, plot

# Set lower precision to reduce CPU/memory load
mp.dps = 10  # Reduced from 30 to 10 for MacBook compatibility

def xi(s):
    """Completed Riemann zeta function."""
    return (s * (s - 1)) / 2 * pi**(-s / 2) * gamma(s / 2) * zeta(s)

def f(t):
    """Density function f(t) from zeta, optimized for performance."""
    xi_half = xi(mpf('0.5'))
    integrand = lambda x: cos(t * x) / xi(mpf('0.5') + x)
    # Smaller integration range and lower quadrature degree for faster computation
    integral = quad(integrand, [-10, 10], maxdegree=6)
    return xi_half / (2 * pi) * integral

# Fewer evaluation points to reduce computation time
ts = [mpf(t) for t in [-0.8, 0, 0.8]]
fs = [f(t) for t in ts]

# Empirical checks
print("f(t) values (should be positive):")
for t, val in zip(ts, fs):
    print(f"f({t}) = {val}")
    if val <= 0:
        print("Warning: Non-positive value detected.")

# Compute integral over [-1,1] with optimized settings
integral_f = quad(f, [-1, 1], maxdegree=6)
print(f"Integral of f(t) over [-1,1]: {integral_f} (expected ~1)")

# Check log-concavity with second differences
log_fs = [log(val) for val in fs]
h = mpf('0.2')
second_diffs = []
for i in range(1, len(log_fs)-1):
    diff = (log_fs[i-1] - 2 * log_fs[i] + log_fs[i+1]) / (h**2)
    second_diffs.append(diff)
print("Second differences of log f (should be negative for logconcavity):", second_diffs)

# Optional: Plot with fewer points to avoid overloading the system
# Comment out if plotting is not needed
plot([lambda t: f(t), lambda t: log(f(t))], [-1, 1], points=50, labels=['f(t)', 'log f(t)'])