import math
import random
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

left_shoulder = (0,0,0)
right_shoulder = (0,0,0)

# Convert (x, y, z) to spherical with elevation as phi
def cartesian_to_spherical(x, y, z):
    r = math.sqrt(x**2 + y**2 + z**2)
    theta = math.atan2(z, x)
    phi = math.asin(z / r) if r != 0 else 0
    return r, math.degrees(theta), math.degrees(phi)

# Generate a random 3D point within given bounds
def generate_random_coords(n=1):
    return [(random.uniform(0, 1), random.uniform(0, 1), random.uniform(0, 1)) for _ in range(n)]


# Generate body points
left_elbow = np.array(generate_random_coords(1)[0])
right_elbow = np.array(generate_random_coords(1)[0])
left_wrist_actual = np.array(generate_random_coords(1)[0])
right_wrist_actual= np.array(generate_random_coords(1)[0])
left_wrist = np.array(left_wrist_actual-left_elbow)  # relative to elbow
right_wrist = np.array(right_wrist_actual-right_elbow)  # relative to elbow

def elbow_angle(shoulder, elbow, wrist):
    a = shoulder - elbow
    b = wrist - elbow
    cos_theta = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
    return np.arccos(np.clip(cos_theta, -1.0, 1.0))  # radians

# Print angles
r, theta, phi = cartesian_to_spherical(*left_elbow)
print(f"Left Elbow: x={left_elbow[0]:.3f}, y={left_elbow[1]:.3f}, z={left_elbow[2]:.3f} -> Left Shoulder θ={theta:.2f}°, Left Chest φ={phi:.2f}°")

r, theta, phi = cartesian_to_spherical(*right_elbow)
print(f"Right Elbow: x={right_elbow[0]:.3f}, y={right_elbow[1]:.3f}, z={right_elbow[2]:.3f} -> Right Shoulder θ={theta:.2f}°, Right Chest φ={phi:.2f}°")

r, theta, phi = cartesian_to_spherical(*left_wrist)
print(f"Left Wrist: x={left_wrist[0]:.3f}, y={left_wrist[1]:.3f}, z={left_wrist[2]:.3f} -> Left Bicep Rotation θ={theta:.2f}°, Left Elbow φ={math.degrees(elbow_angle(left_shoulder,left_elbow,left_wrist_actual)):.2f}°")

r, theta, phi = cartesian_to_spherical(*right_wrist)
print(f"Right Wrist: x={right_wrist[0]:.3f}, y={right_wrist[1]:.3f}, z={right_wrist[2]:.3f} -> Right Bicep Rotation θ={theta:.2f}°, Right Elbow φ={math.degrees(elbow_angle(right_shoulder,right_elbow,right_wrist_actual)):.2f}°")

# Plot body segments
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Plot origin (shoulders)
ax.scatter(0, 0, 0, c='black', s=50, label='Shoulders (origin)')

# Plot and connect Left Arm
ax.plot([0, left_elbow[0]], [0, left_elbow[1]], [0, left_elbow[2]], c='blue', label='Left Upper Arm')
ax.plot([left_elbow[0], left_elbow[0] + left_wrist[0]],
        [left_elbow[1], left_elbow[1] + left_wrist[1]],
        [left_elbow[2], left_elbow[2] + left_wrist[2]],
        c='cyan', label='Left Forearm')

# Plot and connect Right Arm
ax.plot([0, right_elbow[0]], [0, right_elbow[1]], [0, right_elbow[2]], c='red', label='Right Upper Arm')
ax.plot([right_elbow[0], right_elbow[0] + right_wrist[0]],
        [right_elbow[1], right_elbow[1] + right_wrist[1]],
        [right_elbow[2], right_elbow[2] + right_wrist[2]],
        c='orange', label='Right Forearm')

# Joint markers
ax.scatter(*left_elbow, c='blue')
ax.scatter(left_elbow[0] + left_wrist[0], left_elbow[1] + left_wrist[1], left_elbow[2] + left_wrist[2], c='cyan')
ax.scatter(*right_elbow, c='red')
ax.scatter(right_elbow[0] + right_wrist[0], right_elbow[1] + right_wrist[1], right_elbow[2] + right_wrist[2], c='orange')

# Axis labels
ax.set_xlabel('X (left-right)')
ax.set_ylabel('Y (up-down)')
ax.set_zlabel('Z (forward-back)')
ax.set_title("Arm Simulation in Custom 3D Coordinate System")
ax.legend()
ax.view_init(elev=90, azim=-90)
ax.text(left_elbow[0], left_elbow[1], left_elbow[2], 
        f"{math.degrees(elbow_angle(left_shoulder,left_elbow,left_wrist_actual)):.1f}°", color='red', fontsize=10)

ax.text(right_elbow[0], right_elbow[1], right_elbow[2], 
        f"{math.degrees(elbow_angle(right_shoulder,right_elbow,right_wrist_actual)):.1f}°", color='red', fontsize=10)
plt.show()

