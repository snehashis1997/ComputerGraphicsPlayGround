import numpy as np
import matplotlib.pyplot as plt

# Signed Distance Function for a Circle (Head)
def sdf_circle(p, center, radius):
    return np.linalg.norm(p - center, axis=-1) - radius

# Signed Distance Function for a Rectangle (Torso)
def sdf_rectangle(p, center, size):
    d = np.abs(p - center) - size
    return np.linalg.norm(np.maximum(d, 0), axis=-1) + np.minimum(np.max(d, axis=-1), 0)

# Smooth minimum function for blending shapes
def smooth_min(a, b, k=0.1):
    return -np.log(np.exp(-a/k) + np.exp(-b/k)) * k

# Grid for visualization
res = 200
x = np.linspace(-1, 1, res)
y = np.linspace(-1.5, 1.5, res)
X, Y = np.meshgrid(x, y)
points = np.stack([X, Y], axis=-1)

# Define Human Body Parts using SDFs
head = sdf_circle(points, np.array([0, 0.7]), 0.2)
torso = sdf_rectangle(points, np.array([0, 0.2]), np.array([0.2, 0.4]))
legs = sdf_rectangle(points, np.array([0, -0.5]), np.array([0.2, 0.3]))

# Blend them together
human_sdf = smooth_min(head, torso, 0.1)
human_sdf = smooth_min(human_sdf, legs, 0.1)

# Visualize SDF Contours
plt.contour(X, Y, human_sdf, levels=[0], colors='black')
plt.title("SDF-Based 2D Human")
plt.show()
