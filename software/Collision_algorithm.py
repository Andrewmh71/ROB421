import numpy as np
import fcl
import trimesh
from trimesh.transformations import rotation_matrix, translation_matrix
import pyrender  # for nicer visualization

#for future work contact 541 891 6043 "Daniel McVay"

# Utility function to create a box collision object
def create_box(size, transform=np.eye(4)):
    box = fcl.Box(*size)
    return fcl.CollisionObject(box, fcl.Transform(transform[:3, :3], transform[:3, 3]))

def right_arm_check(chest,shoulder,bicep,elbow):
    # Sizes of the links (all 40x40mm in cross section)
    link_dims = [
        [0.04, 0.02, 0.03175],  # Link 1
        [0.04, 0.02, 0.04],     # Link 2
        [0.04, 0.04, 0.11],     # Link 3
        [0.04, 0.04, 0.25]      # Link 4
    ]

    # Mount height
    base_z = 0.25  # 250mm

    #angle adjustments to match robot
    Chest_fix = 135
    Shoulder_fix = 0
    Bicep_fix = 115
    Elbow_fix = 0

    # Joint angles (radians)
    angles = {
        'chest': np.radians(135-Chest_fix),
        'shoulder': np.radians(85-Shoulder_fix),
        'bicep': np.radians(115-Bicep_fix),
        'elbow': np.radians(90-Elbow_fix)
    }

    # Build transforms link-by-link
    transforms = []
    # Chest (rotates about Y)
    T = translation_matrix([0, 0, base_z]) @ rotation_matrix(angles['chest'], [0, 1, 0])
    link1_T = T @ translation_matrix([0, 0, link_dims[0][2] / 2])
    transforms.append(link1_T)

    # Shoulder (rotates about X)
    T = T @ translation_matrix([0, 0, link_dims[0][2]]) @ rotation_matrix(angles['shoulder'], [1, 0, 0])
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
    ground_plane = fcl.CollisionObject(fcl.Plane(np.array([0, 0, 1], dtype=np.float64), 0))        # Ground at Z = 0
    wall_plane = fcl.CollisionObject(fcl.Plane(np.array([-1, 0, 0], dtype=np.float64), 0.06))      # Wall at X = 60mm
    body_plane = fcl.CollisionObject(fcl.Plane(np.array([0, -1, 0], dtype=np.float64), 0.2))       # Body at Y = 20mm


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
                test_value = False
            else:
                test_value = True
        # print("Link 1 = chest fraim, Link 2 = shoulder, Link 3 = bicep, Link 4= forearm")
        # print("✅ Collision check complete.")
    return (test_value)

def left_arm_check(chest,shoulder,bicep,elbow):
            
    # Sizes of the links (all 40x40mm in cross section)
    link_dims = [
        [0.04, 0.02, 0.03175],  # Link 1
        [0.04, 0.02, 0.04],     # Link 2
        [0.04, 0.04, 0.11],     # Link 3
        [0.04, 0.04, 0.25]      # Link 4
        ]

        # Mount height
    base_z = 0.25  # 250mm

        #angle adjustments to match robot
    Chest_fix = 115
    Shoulder_fix = 90
    Bicep_fix = 15
    Elbow_fix = 15

        # Joint angles (radians)
    angles = {
        'chest': np.radians(115-Chest_fix),
        'shoulder': np.radians(180-Shoulder_fix),
        'bicep': np.radians(115-Bicep_fix),
        'elbow': np.radians(105-Elbow_fix)
    }

    # Build transforms link-by-link
    transforms = []
    # Chest (rotates about Z)
    T = translation_matrix([0, 0, -base_z]) @ rotation_matrix(angles['chest'], [0, 0, 1])
        #this section creates the stat link     This rotates the peice around the vector by "Chest" angle
    link1_T = T @ translation_matrix([0, 0, link_dims[0][2] *3 / 2 ])
        #This creates the end of the link, and ^ centers the next start point
    transforms.append(link1_T)

    # Shoulder (rotates about X)
    T = T @ translation_matrix([0, 0, link_dims[0][2]]) @ rotation_matrix(angles['shoulder'], [1, 0, 0])
    #this section creates the stat link     This rotates the peice around the vector by "shoulder" angle
    link2_T = T @ translation_matrix([0, 0, link_dims[1][2] / 2])
        #This creates the end of the link, and ^ centers the next start point
    transforms.append(link2_T)

    # Bicep twist (rotates about Z)
    T = T @ translation_matrix([0, 0, link_dims[1][2]]) @ rotation_matrix(angles['bicep'], [0, 0, 1])
    #this section creates the stat link     This rotates the peice around the vector by "bicep" angle
    link3_T = T @ translation_matrix([0, 0, link_dims[2][2] / 2])
        #This creates the end of the link, and ^ centers the next start point
    transforms.append(link3_T)

    # Elbow pivot (rotates about Y)
    T = T @ translation_matrix([0, 0, link_dims[2][2]]) @ rotation_matrix(angles['elbow'], [0, 1, 0])
    #this section creates the stat link     This rotates the peice around the vector by "elbow" angle
    link4_T = T @ translation_matrix([0, 0, link_dims[3][2] / 2])
        #This creates the end of the link, and ^ centers the next start point
    transforms.append(link4_T)

    # Create collision objects
    links = [create_box(dim, tf) for dim, tf in zip(link_dims, transforms)]

    # Environment
    ground_plane = fcl.CollisionObject(fcl.Plane(np.array([0, 0, 1], dtype=np.float64), 0))        # Ground at Z = 0
    wall_plane = fcl.CollisionObject(fcl.Plane(np.array([-1, 0, 0], dtype=np.float64), 0.06))      # Wall at X = 60mm
    body_plane = fcl.CollisionObject(fcl.Plane(np.array([0, -1, 0], dtype=np.float64), -.425))       # Body at Y = 20mm

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
                test_value = False
            else:
                test_value = True

