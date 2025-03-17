import open3d as o3d
import numpy as np

# Generate random 3D points
points = np.random.rand(50, 3)  # 50 random points in 3D space

# Convert to Open3D point cloud
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(points)

# Compute the Delaunay triangulation using alpha shape (for 3D surfaces)
alpha = 2.0  # Adjust for desired surface smoothness
tetra_mesh, pt_map = o3d.geometry.TetraMesh.create_from_point_cloud(pcd)
mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(pcd, alpha)

# Visualize
o3d.visualization.draw_geometries([mesh])
