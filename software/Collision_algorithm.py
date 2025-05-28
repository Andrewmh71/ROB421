import numpy as np
import fcl
import trimesh
from trimesh.transformations import rotation_matrix, translation_matrix
import pyrender  # for nicer visualization

# Utility function to create a box collision object
def create_box(size, transform=np.eye(4)):
    box = fcl.Box(*size)
    return fcl.CollisionObject(box, fcl.Transform(transform[:3, :3], transform[:3, 3]))

# Sizes of the links (all 40x40mm in cross section)
link_dims = [
    [0.04, 0.02, 0.03175],  # Link 1
    [0.04, 0.02, 0.04],     # Link 2
    [0.04, 0.04, 0.11],     # Link 3
    [0.04, 0.04, 0.25]      # Link 4
]

# Mount height
base_z = 0.22  # 220mm

# Joint angles (radians)
angles = {
    'chest': np.radians(40),
    'shoulder': np.radians(45),
    'bicep': np.radians(0),
    'elbow': np.radians(0)
}

# Build transforms link-by-link
transforms = []
# Chest (rotates about X)
T = translation_matrix([0, 0, base_z]) @ rotation_matrix(angles['chest'], [1, 0, 0])
link1_T = T @ translation_matrix([0, 0, link_dims[0][2] / 2])
transforms.append(link1_T)

# Shoulder (rotates about Y)
T = T @ translation_matrix([0, 0, link_dims[0][2]]) @ rotation_matrix(angles['shoulder'], [0, 1, 0])
link2_T = T @ translation_matrix([0, 0, link_dims[1][2] / 2])
transforms.append(link2_T)

# Bicep twist (rotates about Z)
T = T @ translation_matrix([0, 0, link_dims[1][2]]) @ rotation_matrix(angles['bicep'], [0, 0, 1])
link3_T = T @ translation_matrix([0, 0, link_dims[2][2] / 2])
transforms.append(link3_T)

# Elbow pivot (rotates about Y)
T = T @ translation_matrix([0, 0, link_dims[2][2]]) @ rotation_matrix(angles['elbow'], [0, 1, 0])
link4_T = T @ translation_matrix([0, 0, link_dims[3][2] / 2])
transforms.append(link4_T)

# Create collision objects
links = [create_box(dim, tf) for dim, tf in zip(link_dims, transforms)]

# Environment
ground_plane = fcl.CollisionObject(fcl.Plane(np.array([0, 0, -1], dtype=np.float64), 0))        # Ground at Z = 0
wall_plane = fcl.CollisionObject(fcl.Plane(np.array([-1, 0, 0], dtype=np.float64), 0.06))      # Wall at X = 60mm
body_plane = fcl.CollisionObject(fcl.Plane(np.array([0, -1, 0], dtype=np.float64), 0.2))       # Wall at X = 60mm


# Check for self-collisions
for i in range(len(links)):
    for j in range(i + 1, len(links)):
        req = fcl.CollisionRequest()
        result = fcl.CollisionResult()
        fcl.collide(links[i], links[j], req, result)
        if result.is_collision:
            print(f"❌ Self-collision between Link {i+1} and Link {j+1}")

# Check for environment collisions
for idx, link in enumerate(links):
    for env, name in [(ground_plane, 'ground'), (wall_plane, 'wall'), (body_plane, 'body')]:
        req = fcl.CollisionRequest()
        result = fcl.CollisionResult()
        fcl.collide(link, env, req, result)
        if result.is_collision:
            print(f"❌ Link {idx+1} collides with {name}")
print("Link 1 = chest fraim, Link 2 = shoulder, Link 3 = bicep, Link 4= forearm")
print("✅ Collision check complete.")


# Create trimesh box meshes with the same size as your fcl boxes
meshes = []
for dim, tf in zip(link_dims, transforms):
    # Create box mesh centered at origin
    box_mesh = trimesh.creation.box(extents=dim)
    # Apply transform (convert 4x4 matrix to (4,4) numpy array if needed)
    box_mesh.apply_transform(tf)
    meshes.append(box_mesh)

# Combine meshes into a scene for visualization
scene = trimesh.Scene()
for i, mesh in enumerate(meshes):
    scene.add_geometry(mesh, node_name=f'Link_{i+1}')

wall_thickness = 0.01
wall_size = 1.0

# Red color with 70% transparency (alpha = 0.3)
red_transparent = [255, 0, 0, int(255 * 0.3)]  # RGBA in 0-255 range
blue_transparent = [0, 0, 255, int(255 * 0.3)]  # RGBA in 0-255 range
green_transparent = [0, 255, 0, int(255 * 0.3)]  # RGBA in 0-255 range

# Ground plane (transparent Blue)
ground_mesh = trimesh.creation.box(extents=[wall_size, wall_size, wall_thickness])
ground_mesh.apply_translation([0, 0, -wall_thickness/2])
ground_mesh.visual.face_colors = blue_transparent

# Wall plane (transparent red)
wall_mesh = trimesh.creation.box(extents=[wall_thickness, wall_size, wall_size])
wall_mesh.apply_translation([-0.06 + wall_thickness/2, 0, wall_size/2])
wall_mesh.visual.face_colors = red_transparent

# Body plane (transparent green)
body_mesh = trimesh.creation.box(extents=[wall_size, wall_thickness, wall_size])
body_mesh.apply_translation([0, -0.2 - wall_thickness/2, wall_size/2])
body_mesh.visual.face_colors = green_transparent

# Create scene and add robot links (assuming meshes from before)
scene = trimesh.Scene()
for i, mesh in enumerate(meshes):
    scene.add_geometry(mesh, node_name=f'Link_{i+1}')

# Add transparent red walls
scene.add_geometry(ground_mesh, node_name='Ground')
scene.add_geometry(wall_mesh, node_name='Wall')
scene.add_geometry(body_mesh, node_name='Body')

scene.show()