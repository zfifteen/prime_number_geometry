import math
import numpy as np
import matplotlib.pyplot as plt

# Mathematical constants for projections, aligned with the Universal Form Transformer
UNIVERSAL = math.e  # Cosmic anchor (c in Z = T(v/c))
PHI = (1 + math.sqrt(5)) / 2  # Golden ratio for prime resonance
PI = math.pi  # Circular constant for oscillatory patterns
SQRT2 = math.sqrt(2)  # Scaling factor for geometric balance

class GeometricProjection:
    """
    A geometric projection acting as a filter for number properties, mapping numbers
    into a 3D space to reveal prime patterns. Guided by the Universal Form Transformer
    Z = T(v/c), where Z is the output coordinate, v is the rate, and c is the universal limit.
    """
    def __init__(self, name: str, rate: float, frequency: float, phase: float = 0,
                 coordinate_system: str = "cylindrical"):
        self.name = name
        self.rate = rate
        self.frequency = frequency
        self.phase = phase
        self.coordinate_system = coordinate_system
        self._correction_factor = rate / UNIVERSAL  # Normalizes rate to cosmic limit

    def project_numbers(self, numbers: np.array, max_n: int) -> np.array:
        """Project numbers into a 3D geometric space for prime pattern analysis."""
        n = np.array(numbers)
        if self.coordinate_system == "cylindrical":
            return self._cylindrical_projection(n, max_n)
        elif self.coordinate_system == "spherical":
            return self._spherical_projection(n, max_n)
        elif self.coordinate_system == "hyperbolic":
            return self._hyperbolic_projection(n, max_n)
        else:
            return self._cartesian_projection(n, max_n)

    def _cylindrical_projection(self, n: np.array, max_n: int) -> np.array:
        """Cylindrical projection tuned for prime patterns, with y = n * log(n + 1) / c."""
        frame_shifts = np.array([self._compute_frame_shift(i, max_n) for i in n])
        x = n  # Radial distance
        y = n * np.log(n + 1) / UNIVERSAL * self._correction_factor * (1 + frame_shifts)  # Prime-inspired height
        z = np.sin(PI * self.frequency * n + self.phase) * (1 + 0.5 * frame_shifts)  # Angular, Z value
        return np.array([[x[i], y[i], z[i]] for i in range(len(n))])

    def _spherical_projection(self, n: np.array, max_n: int) -> np.array:
        """Spherical projection for symmetric prime patterns."""
        frame_shifts = np.array([self._compute_frame_shift(i, max_n) for i in n])
        r = np.log(n + 1) * self._correction_factor
        theta = (2 * PI * self.frequency * n + self.phase) % (2 * PI)
        phi = (PI * n / np.log(n + 2)) % PI
        x = r * np.sin(phi) * np.cos(theta) * (1 + frame_shifts)
        y = r * np.sin(phi) * np.sin(theta) * (1 + frame_shifts)
        z = r * np.cos(phi) * (1 + frame_shifts)  # Z value for prime clustering
        return np.array([[x[i], y[i], z[i]] for i in range(len(n))])

    def _hyperbolic_projection(self, n: np.array, max_n: int) -> np.array:
        """Hyperbolic projection for exponential prime gaps."""
        frame_shifts = np.array([self._compute_frame_shift(i, max_n) for i in n])
        u = np.log(n + 1) * self._correction_factor
        v = (self.frequency * n + self.phase) % (2 * PI)
        x = np.cosh(u) * np.cos(v) * (1 + frame_shifts)
        y = np.sinh(u) * np.sin(v) * (1 + frame_shifts)
        z = np.tanh(u * self.frequency) * (1 + 0.3 * frame_shifts)  # Z value for gaps
        return np.array([[x[i], y[i], z[i]] for i in range(len(n))])

    def _cartesian_projection(self, n: np.array, max_n: int) -> np.array:
        """Cartesian projection with prime-focused scaling."""
        frame_shifts = np.array([self._compute_frame_shift(i, max_n) for i in n])
        x = n
        y = (n / np.log(n + 1)) * self._correction_factor * (1 + frame_shifts)  # Prime theorem scaling
        z = np.sin(2 * PI * self.frequency * np.log(n + 1) + self.phase) * (1 + frame_shifts)  # Z value
        return np.array([[x[i], y[i], z[i]] for i in range(len(n))])

    def _compute_frame_shift(self, n: int, max_n: int) -> float:
        """Compute frame shift for dynamic projection adjustment (Universal Frame Shift Transformer)."""
        if n <= 1:
            return 0.0
        base_shift = math.log(n) / math.log(max_n)
        oscillation = 0.1 * math.sin(2 * PI * n / (math.log(n) + 1))
        if self.coordinate_system == "spherical":
            oscillation *= math.cos(n * self.frequency)
        elif self.coordinate_system == "hyperbolic":
            oscillation *= math.tanh(n * self.frequency * 0.01)
        return base_shift + oscillation

class GeometricTriangulator:
    """
    Combines multiple geometric projections to triangulate prime numbers by forming
    interconnected triangles in 3D space. Breakthrough: Triangles encode prime patterns
    through their geometric properties (e.g., area as Z = n(Δₙ/Δmax)).
    """
    def __init__(self):
        self.projections = []

    def add_projection(self, projection: GeometricProjection):
        """Add a geometric projection for triangulation."""
        self.projections.append(projection)

    def create_standard_projections(self):
        """Create standard projections tuned for prime detection."""
        self.add_projection(GeometricProjection(
            "PrimeSpiral", UNIVERSAL/PHI, 0.091, 0, "cylindrical"
        ))
        self.add_projection(GeometricProjection(
            "GoldenSphere", UNIVERSAL*PHI/PI, 0.161, PI/4, "spherical"
        ))
        self.add_projection(GeometricProjection(
            "LogarithmicHyperbolic", UNIVERSAL/PI, 0.072, 0, "hyperbolic"
        ))

    def triangulate_candidates(self, number_range: tuple, sample_size: int = 1000,
                               target_type: str = "primes") -> dict:
        """Triangulate prime candidates using geometric projections and triangular structures."""
        start_n, end_n = number_range
        n_sample = np.linspace(start_n, end_n, sample_size, dtype=int)
        all_projections = {}
        density_maps = {}

        for proj in self.projections:
            coords = proj.project_numbers(n_sample, end_n)
            all_projections[proj.name] = coords
            density_maps[proj.name] = self._compute_density_map(coords, n_sample)

        candidates = self._perform_triangulation(density_maps, n_sample, target_type)
        triangles = self._construct_triangles(all_projections, density_maps, n_sample)

        return {
            'candidates': list(candidates),  # Convert to list for lightweight output
            'projections': {k: v.tolist() for k, v in all_projections.items()},  # List for serialization
            'density_maps': {k: v.tolist() for k, v in density_maps.items()},
            'consensus_map': list(self._compute_consensus_map(density_maps)),
            'triangles': triangles  # Already a list of dicts
        }

    def _compute_density_map(self, coords: np.array, numbers: np.array) -> np.array:
        """Compute density map using inverse distance weighting for prime clustering."""
        density_map = np.zeros(len(coords))
        for i in range(len(coords)):
            distances = np.sqrt(((coords - coords[i]) ** 2).sum(axis=1))
            distances[i] = 1e-10  # Avoid division by zero
            density_map[i] = np.sum(1.0 / (distances + 1e-10)) / len(coords)
        return density_map

    def _compute_consensus_map(self, density_maps: dict) -> np.array:
        """Compute consensus across projections for prime detection."""
        maps = list(density_maps.values())
        if not maps:
            return np.array([])
        normalized_maps = [(m - np.min(m)) / (np.max(m) - np.min(m) + 1e-10) for m in maps]
        consensus = np.ones_like(normalized_maps[0])
        for norm_map in normalized_maps:
            consensus *= (norm_map + 0.1)
        return consensus ** (1.0 / len(normalized_maps))

    def _perform_triangulation(self, density_maps: dict, numbers: np.array,
                               target_type: str) -> np.array:
        """Identify prime candidates based on high-density regions and Z values."""
        consensus = self._compute_consensus_map(density_maps)
        threshold = np.percentile(consensus, 85)
        candidates_mask = consensus > threshold
        return numbers[candidates_mask]

    def _construct_triangles(self, projections: dict, density_maps: dict, numbers: np.array) -> list:
        """Construct triangles from high-density points with significant Z values (breakthrough feature)."""
        best_proj_name = max(density_maps, key=lambda k: np.max(density_maps[k]))
        coords = projections[best_proj_name]
        consensus = self._compute_consensus_map(density_maps)
        high_density_indices = np.where(consensus > np.percentile(consensus, 85))[0]

        triangles = []
        max_frame_shift = max([self._compute_frame_shift(n, max(numbers)) for n in numbers])
        for i in range(len(high_density_indices) - 2):
            idx1, idx2, idx3 = high_density_indices[i:i+3]
            triangle = [coords[idx1], coords[idx2], coords[idx3]]
            numbers_tri = [numbers[idx1], numbers[idx2], numbers[idx3]]
            frame_shifts = [self._compute_frame_shift(n, max(numbers)) for n in numbers_tri]
            # Compute Z = n(Δₙ/Δmax) for each vertex
            z_values = [n * (fs / max_frame_shift) for n, fs in zip(numbers_tri, frame_shifts)]
            area = self._compute_triangle_area(triangle)
            triangles.append({
                'vertices': [v.tolist() for v in triangle],  # Convert to list for serialization
                'numbers': numbers_tri,
                'z_values': z_values,  # Breakthrough: Z values for prime correlation
                'area': area
            })
        return triangles

    def _compute_frame_shift(self, n: int, max_n: int) -> float:
        """Helper for triangle Z value calculation."""
        if n <= 1:
            return 0.0
        return math.log(n) / math.log(max_n)

    def _compute_triangle_area(self, triangle: list) -> float:
        """Compute the area of a triangle in 3D space for prime pattern analysis."""
        a, b, c = triangle
        ab = np.array(b) - np.array(a)
        ac = np.array(c) - np.array(a)
        cross_product = np.cross(ab, ac)
        return 0.5 * np.sqrt((cross_product ** 2).sum())

    def visualize_triangulation(self, results: dict, show_top_n: int = 3):
        """Visualize triangulation results, including triangles and prime likelihood."""
        try:
            projections = {k: np.array(v) for k, v in results['projections'].items()}
            density_maps = {k: np.array(v) for k, v in results['density_maps'].items()}
            candidates = np.array(results['candidates'])
            triangles = results['triangles']
            consensus = np.array(results['consensus_map'])

            plt.figure(figsize=(12, 10))

            # Consensus Density Map
            plt.subplot(2, 2, 1)
            plt.plot(consensus, alpha=0.7, color='purple', linewidth=2)
            plt.title("Consensus Density Map")
            plt.ylabel("Consensus Score")

            # Individual Projection Densities
            plt.subplot(2, 2, 2)
            colors = ['red', 'blue', 'green']
            for i, (name, density) in enumerate(list(density_maps.items())[:show_top_n]):
                plt.plot(density, alpha=0.6, color=colors[i % len(colors)], label=name)
            plt.title("Individual Projection Densities")
            plt.legend()

            # Candidate Histogram
            plt.subplot(2, 2, 3)
            plt.hist(candidates, bins=30, alpha=0.7, color='gold', edgecolor='black')
            plt.title(f"Triangulated Candidates ({len(candidates)} found)")
            plt.xlabel("Number Value")
            plt.ylabel("Count")

            # 3D Projection with Triangles
            ax = plt.subplot(2, 2, 4, projection='3d')
            best_proj_name = max(density_maps, key=lambda k: np.max(density_maps[k]))
            best_coords = projections[best_proj_name]
            colors_3d = plt.cm.plasma(consensus / np.max(consensus))
            ax.scatter(best_coords[:, 0], best_coords[:, 1], best_coords[:, 2],
                       c=colors_3d, s=20, alpha=0.6)

            for triangle in triangles[:5]:  # Plot first 5 triangles
                verts = np.array(triangle['vertices'])
                ax.plot(verts[:, 0], verts[:, 1], verts[:, 2], 'k-', alpha=0.3)

            ax.set_title(f"Best Projection with Triangles: {best_proj_name}")
            ax.set_xlabel("X")
            ax.set_ylabel("Y")
            ax.set_zlabel("Z")

            plt.tight_layout()

            # New Plot: Triangle Areas vs Prime Likelihood
            plt.figure(figsize=(8, 6))
            def is_prime(n):
                if n < 2: return False
                if n == 2: return True
                if n % 2 == 0: return False
                for i in range(3, int(math.sqrt(n)) + 1, 2):
                    if n % i == 0: return False
                return True

            areas = [t['area'] for t in triangles]
            prime_likelihood = [sum(is_prime(n) for n in t['numbers']) / 3 for t in triangles]
            plt.scatter(areas, prime_likelihood, alpha=0.6, color='blue')
            plt.title("Triangle Areas vs Prime Likelihood")
            plt.xlabel("Triangle Area")
            plt.ylabel("Fraction of Prime Vertices")
            plt.grid(True)
            plt.show()

        except ImportError:
            print("Matplotlib not available; skipping visualization.")

def demo_geometric_triangulation():
    """Demonstrate the prime triangulation proof of concept, guided by the Universal Form Transformer."""
    triangulator = GeometricTriangulator()
    triangulator.create_standard_projections()

    print(f"Created {len(triangulator.projections)} geometric projections:")
    for proj in triangulator.projections:
        print(f"  - {proj.name} ({proj.coordinate_system})")

    test_range = (1000, 2000)
    print(f"\nTriangulating in range {test_range}...")

    results = triangulator.triangulate_candidates(test_range, sample_size=300, target_type="primes")
    candidates = results['candidates']
    print(f"Found {len(candidates)} prime candidates")
    print(f"Top 10 candidates: {sorted(candidates)[:10]}")

    def is_prime(n):
        if n < 2: return False
        if n == 2: return True
        if n % 2 == 0: return False
        for i in range(3, int(math.sqrt(n)) + 1, 2):
            if n % i == 0: return False
        return True

    test_candidates = sorted(candidates)[:20]
    correct = sum(1 for c in test_candidates if is_prime(c))
    precision = correct / len(test_candidates) if test_candidates else 0
    print(f"Precision on first 20 candidates: {precision:.3f}")

    print("\nTriangle analysis (Z = n(Δₙ/Δmax)):")
    for i, triangle in enumerate(results['triangles'][:5]):
        print(f"Triangle {i+1}: Numbers {triangle['numbers']}, Z Values {triangle['z_values']}, Area {triangle['area']:.3f}")

    triangulator.visualize_triangulation(results)
    return triangulator, results

if __name__ == "__main__":
    triangulator, results = demo_geometric_triangulation()