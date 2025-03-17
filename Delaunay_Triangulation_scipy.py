import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import Delaunay

# Generate random 2D points
points = np.random.rand(20, 2)  # 20 random points in a unit square

# Compute the Delaunay triangulation
tri = Delaunay(points)

# Plot the triangulation
plt.triplot(points[:, 0], points[:, 1], tri.simplices, color='blue')
plt.scatter(points[:, 0], points[:, 1], color='red', marker='o')
plt.title("Delaunay Triangulation")
plt.show()
