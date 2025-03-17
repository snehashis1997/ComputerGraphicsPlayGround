import numpy as np
import matplotlib.pyplot as plt

# SDF for a sphere
def sdf_sphere(p, center=np.array([0, 0, 3]), radius=1.0):
    return np.linalg.norm(p - center) - radius

# SDF for a box
def sdf_box(p, center=np.array([0, 0, 3]), size=np.array([1, 1, 1])):
    q = np.abs(p - center) - size
    return np.linalg.norm(np.maximum(q, 0)) + min(max(q[0], max(q[1], q[2])), 0)

# Scene composition (return min distance)
def sdf_scene(p):
    return min(sdf_sphere(p), sdf_box(p))

def ray_march(orig, direction, max_steps=100, min_dist=0.001, max_dist=10.0):
    """ Performs ray marching for given ray origin and direction """
    t = 0  # Distance traveled along ray
    for _ in range(max_steps):
        point = orig + t * direction  # Compute current point on ray
        dist = sdf_scene(point)  # Evaluate SDF
        if dist < min_dist:  # Hit detected
            return t, point
        if t > max_dist:  # Miss (ray escapes)
            return max_dist, None
        t += dist  # Move forward by SDF value
    return max_dist, None  # Miss

def compute_normal(p, eps=1e-3):
    """ Approximate surface normal using central differences """
    dx = np.array([eps, 0, 0])
    dy = np.array([0, eps, 0])
    dz = np.array([0, 0, eps])
    
    normal = np.array([
        sdf_scene(p + dx) - sdf_scene(p - dx),
        sdf_scene(p + dy) - sdf_scene(p - dy),
        sdf_scene(p + dz) - sdf_scene(p - dz)
    ])
    return normal / np.linalg.norm(normal)

def phong_shading(point, normal, light_pos=np.array([5, 5, 5])):
    """ Compute Phong shading (ambient + diffuse + specular) """
    light_dir = light_pos - point
    light_dir /= np.linalg.norm(light_dir)

    # Ambient term
    ambient = 0.1

    # Diffuse term (Lambertian reflection)
    diffuse = max(np.dot(normal, light_dir), 0.0)

    # Specular term (Blinn-Phong)
    view_dir = np.array([0, 0, -1])  # Camera direction
    halfway = (light_dir + view_dir) / np.linalg.norm(light_dir + view_dir)
    specular = max(np.dot(normal, halfway), 0.0) ** 32  # Shininess factor

    return ambient + 0.7 * diffuse + 0.3 * specular  # Weighted sum

res = 200  # Image resolution
aspect_ratio = 1.0  # Square image
screen_x = np.linspace(-1, 1, res) * aspect_ratio
screen_y = np.linspace(-1, 1, res)
X, Y = np.meshgrid(screen_x, screen_y)

# Camera setup
camera_origin = np.array([0, 0, -2])  # Camera position
direction = np.stack([X, Y, np.ones_like(X)], axis=-1)  # Rays direction
direction /= np.linalg.norm(direction, axis=-1, keepdims=True)  # Normalize

# Compute image
image = np.zeros((res, res))
for i in range(res):
    for j in range(res):
        t, hit_point = ray_march(camera_origin, direction[i, j])
        if hit_point is not None:
            normal = compute_normal(hit_point)
            image[i, j] = phong_shading(hit_point, normal)

# Display the Rendered Image
plt.imshow(image, cmap='inferno', extent=[-1, 1, -1, 1])
plt.colorbar(label="Intensity")
plt.title("3D Ray Marched SDF Scene with Phong Shading")
plt.show()