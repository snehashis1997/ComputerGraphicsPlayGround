import numpy as np
import matplotlib.pyplot as plt

# Signed Distance Function for a Circle
def sdf_circle(p, center, radius):
    return np.linalg.norm(p - center, axis=-1) - radius

# Signed Distance Function for a Rectangle
def sdf_rectangle(p, center, size):
    d = np.abs(p - center) - size
    return np.linalg.norm(np.maximum(d, 0), axis=-1) + np.minimum(np.max(d, axis=-1), 0)

# Smooth Min (Soft Union)
def smooth_min(a, b, k=0.1):
    return -np.log(np.exp(-a/k) + np.exp(-b/k)) * k

# Smooth Max (Soft Intersection)
def smooth_max(a, b, k=0.1):
    return -np.log(np.exp(a/k) + np.exp(b/k)) * k

# Grid for visualization
res = 200
x = np.linspace(-1, 1, res)
y = np.linspace(-1, 1, res)
X, Y = np.meshgrid(x, y)
points = np.stack([X, Y], axis=-1)

# Define two SDFs
circle_sdf = sdf_circle(points, np.array([0, 0.3]), 0.3)
rect_sdf = sdf_rectangle(points, np.array([0, -0.2]), np.array([0.3, 0.2]))

# Smooth blending
smooth_union = smooth_min(circle_sdf, rect_sdf, k=0.2)
smooth_intersection = smooth_max(circle_sdf, rect_sdf, k=0.2)

# Plot results
fig, axs = plt.subplots(1, 3, figsize=(12, 4))

# Original Shapes
axs[0].contour(X, Y, circle_sdf, levels=[0], colors='red', label='Circle')
axs[0].contour(X, Y, rect_sdf, levels=[0], colors='blue', label='Rectangle')
axs[0].set_title("Original SDFs")

# Smooth Union
axs[1].contour(X, Y, smooth_union, levels=[0], colors='black')
axs[1].set_title("Smooth Union (Soft Min)")

# Smooth Intersection
axs[2].contour(X, Y, smooth_intersection, levels=[0], colors='black')
axs[2].set_title("Smooth Intersection (Soft Max)")

plt.show()