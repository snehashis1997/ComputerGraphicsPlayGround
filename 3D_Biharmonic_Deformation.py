import numpy as np
import tetgen
import scipy.sparse
import scipy.sparse.linalg
import open3d as o3d

# Create a simple 3D cube mesh
vertices = np.array([
    [0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],  # Bottom square
    [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]   # Top square
])
faces = np.array([
    [0, 1, 2], [0, 2, 3],  # Bottom
    [4, 5, 6], [4, 6, 7],  # Top
    [0, 1, 5], [0, 5, 4],  # Front
    [1, 2, 6], [1, 6, 5],  # Right
    [2, 3, 7], [2, 7, 6],  # Back
    [3, 0, 4], [3, 4, 7]   # Left
])

# Generate tetrahedral mesh using TetGen
tet = tetgen.TetGen()
tet.make_manifold_surface(vertices, faces)
tet.tetrahedralize(order=1)
tet_vertices = tet.node
tet_faces = tet.face
tet_tets = tet.tetrahedron

from scipy.sparse import csr_matrix

def compute_laplacian_3d(vertices, tets):
    """ Compute 3D Laplacian matrix using cotangent weights. """
    n = len(vertices)
    L = scipy.sparse.lil_matrix((n, n))

    for t in tets:
        for i in range(4):
            for j in range(i + 1, 4):
                v1, v2 = vertices[t[i]], vertices[t[j]]
                edge_vec = v2 - v1
                weight = np.dot(edge_vec, edge_vec)  # Simple edge weight

                L[t[i], t[j]] -= weight
                L[t[j], t[i]] -= weight
                L[t[i], t[i]] += weight
                L[t[j], t[j]] += weight

    return csr_matrix(L)

# Compute Laplacian and Biharmonic Matrix (Laplacian²)
L = compute_laplacian_3d(tet_vertices, tet_tets)
L_biharmonic = L @ L

# Find boundary (e.g., fix bottom vertices)
boundary_indices = np.where(tet_vertices[:, 2] == 0)[0]
displacement = np.zeros((len(tet_vertices), 3))

# Apply a deformation (e.g., moving top layer up)
boundary_top = np.where(tet_vertices[:, 2] == 1)[0]
displacement[boundary_top, 2] = np.sin(np.pi * tet_vertices[boundary_top, 0])

# Solve biharmonic equation for free vertices
free_indices = np.setdiff1d(np.arange(len(tet_vertices)), boundary_indices)
new_positions = tet_vertices.copy()
new_positions[free_indices] = scipy.sparse.linalg.spsolve(L_biharmonic[free_indices, :][:, free_indices], displacement[free_indices])

# Convert to Open3D point cloud
pcd_original = o3d.geometry.PointCloud()
pcd_original.points = o3d.utility.Vector3dVector(tet_vertices)

pcd_deformed = o3d.geometry.PointCloud()
pcd_deformed.points = o3d.utility.Vector3dVector(new_positions)

# Show before & after deformation
o3d.visualization.draw_geometries([pcd_original], window_name="Original Mesh")
o3d.visualization.draw_geometries([pcd_deformed], window_name="Deformed Mesh")