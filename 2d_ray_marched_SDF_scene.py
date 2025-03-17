import numpy as np
import matplotlib.pyplot as plt

# === Step 1: Define the SDF ===
def sdf_sphere(p, center, radius):
    return np.linalg.norm(p - center, axis=-1) - radius

# === Step 2: Define Ray Marching Function ===
def ray_march(orig, direction, max_steps=100, min_dist=0.001, max_dist=2.0):
    """ Ray marching loop """
    t = 0  # Distance traveled along the ray
    for _ in range(max_steps):
        point = orig + t * direction  # Current point along the ray
        dist = sdf_sphere(point, center=np.array([0, 0.0]), radius=0.5)  # Evaluate SDF
        if dist < min_dist:  # If close enough to surface, return hit distance
            return t
        if t > max_dist:  # If ray goes too far, return no hit
            return max_dist
        t += dist  # Move forward by SDF value
    return max_dist

# === Step 3: Render the SDF Scene ===
res = 200  # Resolution of the image
screen_x = np.linspace(-1, 1, res)  # Screen X-coordinates
screen_y = np.linspace(-1, 1, res)  # Screen Y-coordinates
X, Y = np.meshgrid(screen_x, screen_y)

orig = np.array([0, -1])  # Camera origin (2D)
direction = np.stack([X, Y], axis=-1)  # Ray direction
direction /= np.linalg.norm(direction, axis=-1, keepdims=True)  # Normalize

# Compute distances using ray marching
distances = np.array([[ray_march(orig, d) for d in row] for row in direction])

# === Step 4: Display the Rendered Image ===#
plt.imshow(distances, cmap='inferno', extent=[-1, 1, -1, 1])
plt.colorbar(label="Distance")
plt.title("2D Ray Marched SDF Sphere")
plt.show()