import numpy as np
import cv2
import matplotlib.pyplot as plt
import glob
import os

# Define a voxel grid which has the 3D locations of each voxel which can then be projected onto each image
def InitializeVoxels(xlim, ylim, zlim, voxel_size):
    voxels_number = [1, 1, 1]
    voxels_number[0] = np.abs(xlim[1]-xlim[0]) / voxel_size[0]
    voxels_number[1] = np.abs(ylim[1]-ylim[0]) / voxel_size[1]
    voxels_number[2] = np.abs(zlim[1]-zlim[0]) / voxel_size[2]
    voxels_number_act = np.array(voxels_number).astype(int) + 1
    total_number = np.prod(voxels_number_act)

    voxel = np.ones((total_number, 4))

    sx = xlim[0]
    ex = xlim[1]
    sy = ylim[0]
    ey = ylim[1]
    sz = zlim[0]
    ez = zlim[1]

    voxel3Dx, voxel3Dy, voxel3Dz = np.meshgrid(
        np.linspace(sx, ex, voxels_number_act[0]),
        np.linspace(sy, ey, voxels_number_act[1]),
        np.linspace(sz, ez, voxels_number_act[2])
    )

    l = 0
    for z in np.linspace(sz, ez, voxels_number_act[2]):
        for x in np.linspace(sx, ex, voxels_number_act[0]):
            for y in np.linspace(sy, ey, voxels_number_act[1]):
                voxel[l] = [x, y, z, 1]
                l += 1

    return voxel, voxel3Dx, voxel3Dy, voxel3Dz, voxels_number

# Look-at matrix function
def look_at_matrix(camera_position, target_position, up_vector=np.array([0, 0, 1])):
    """
    Generate a look-at matrix for a camera.
    """
    forward = target_position - camera_position
    forward = forward / np.linalg.norm(forward)  # Normalize

    right = np.cross(up_vector, forward)
    right = right / np.linalg.norm(right)  # Normalize

    up = np.cross(forward, right)

    rotation_matrix = np.eye(4)
    rotation_matrix[0, :3] = right
    rotation_matrix[1, :3] = up
    rotation_matrix[2, :3] = -forward

    translation_matrix = np.eye(4)
    translation_matrix[:3, 3] = -camera_position

    return rotation_matrix @ translation_matrix

# Orthogonal Projection Matrix
def orthogonal_projection_matrix():
    """
    Generates an orthogonal projection matrix.
    This removes the depth (Z) component from the 3D points.
    """
    return np.array([
        [1, 0, 0, 0],  # Retain X
        [0, 1, 0, 0],  # Retain Y
        [0, 0, 0, 0]   # Ignore Z
    ])

# Initialize voxels
voxel_size = [10, 10, 10]
xlim = [0, 900]
ylim = [-640, 0]
zlim = [0, 640]
voxels, voxel3Dx, voxel3Dy, voxel3Dz, voxels_number = InitializeVoxels(xlim, ylim, zlim, voxel_size)

# Camera setup
N = 8
radius = 480
center = np.array([0, 0, 0])  # Center point to look at
angles = np.linspace(0, 2 * np.pi, N, endpoint=False)

M = []
camera_positions = []

for theta in angles:
    # Compute camera position
    camera_pos = np.array([radius * np.cos(theta) + 400, radius * np.sin(theta) + 300, 1])
    camera_positions.append(camera_pos)

    # Generate look-at matrix
    M_lookat = look_at_matrix(camera_pos, center)
    M.append(M_lookat)

    print(f"Camera {len(M)}: Look-At Matrix\n", M_lookat)

# Image loading
data_dir = "images/results/"
images = glob.glob(os.path.join(data_dir, '*.png'))

imgs = []
kernel = np.ones((3, 3), np.uint8)
for i in images:
    im = cv2.imread(i)
    im = cv2.cvtColor((im * 255).astype(np.uint8), cv2.COLOR_BGR2GRAY) / 255.0
    im = cv2.erode(im, kernel, iterations=1)
    im = cv2.dilate(im, kernel, iterations=1)
    _, im = cv2.threshold(im, 0.1, 1, cv2.THRESH_BINARY)
    imgs.append(np.array(im))
imgs = np.array(imgs)

silhouettes = []
for im in imgs:
    silhouettes.append(im > 0)

silhouettes = np.array(silhouettes).transpose(1, 2, 0)

object_points3D = np.copy(voxels).T
voxels[:, 3] = 0  # Reset the fourth variable of each voxel

# Orthogonal Projection 적용
P_ortho = orthogonal_projection_matrix()

proj = []
for i in range(N):
    # Apply Look-At Transformation
    M_ = M[i]
    camera_space_points = np.matmul(M_, object_points3D)

    # Apply Orthogonal Projection
    points2D = np.matmul(P_ortho, camera_space_points)  # Project to 2D
    points2D = points2D[:2, :]  # Extract x, y
    points2D = np.floor(points2D).astype(np.int32)  # Discretize to pixel space


    # Handle boundary conditions
    img_height, img_width = silhouettes[:, :, 0].shape
    points2D[0, :] = np.clip(points2D[0, :], 0, img_width - 1)
    points2D[1, :] = np.clip(points2D[1, :], 0, img_height - 1)


    # Accumulate voxel contributions based on silhouettes
    voxels[:, 3] += silhouettes[:, :, i].T[points2D.T[:, 0], points2D.T[:, 1]]
    proj.append(points2D)


# Thresholding to filter active voxels
threshold = 3
filtered_voxels = voxels[voxels[:, 3] >= threshold]

x, y, z = filtered_voxels[:, 0], filtered_voxels[:, 1], filtered_voxels[:, 2]

# 시각화
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# 3D 산점도
ax.scatter(x, y, z, c=filtered_voxels[:, 3], cmap='viridis', s=2)

# 축 및 제목 설정
ax.set_title('Filtered Voxels with Overlap >= 3', fontsize=16)
ax.set_xlabel('X', fontsize=12)
ax.set_ylabel('Y', fontsize=12)
ax.set_zlabel('Z', fontsize=12)
ax.grid(True)

plt.show()

for i in range(N):
    camera_space_points = np.matmul(M[i], object_points3D)  # Transform to camera space
    points2D = np.matmul(P_ortho, camera_space_points)  # Orthogonal projection
    points2D = np.floor(points2D[:2, :]).astype(np.int32)  # 2D coordinates

    # Check alignment with silhouette
    plt.imshow(silhouettes[:, :, i], cmap='gray')
    plt.scatter(points2D[0], points2D[1], s=1, c='red')  # Projected points
    plt.title(f"Camera {i} Projection")
    plt.show()
