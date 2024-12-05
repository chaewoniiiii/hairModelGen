import numpy as np
import cv2
import torch.nn as nn
import torch
import matplotlib as plt


data_dir = "dinoRing/"
file_base = "dinoR"

with open(data_dir + "/" + file_base + "_par.txt", 'r') as f:
  lines = f.readlines() 

  M = []
  for l in lines[1:]: 
    tmp = np.array(l.strip().split(" ")[1:]).astype(np.float32)
    K = tmp[0:9].reshape((3, 3))
    R = tmp[9:18].reshape((3, 3))
    t = tmp[18:].reshape((3, 1))
    M.append(np.matmul(K, np.concatenate([R, t], axis=1)))

  N = len(lines) - 1

imgs = []
kernel = np.ones((3,3), np.uint8) 
for i in range(N):
  im = cv2.imread(data_dir + "/" + file_base + f'{i + 1:04}' + ".png", 0) / 255
  im = cv2.erode(im, kernel, iterations=1) 
  im = cv2.dilate(im, kernel, iterations=1) 
  _, im = cv2.threshold(im, 0.1, 1, cv2.THRESH_BINARY)
  print(im)
  imgs.append(np.array(im))
imgs = np.array(imgs)

silhouettes = []
for im in imgs:
  silhouettes.append(im > 0)
silhouettes = np.array(silhouettes).transpose(1, 2, 0)

def InitializeVoxels(x_size, y_size, z_size):

  total_number = x_size * y_size * z_size
  voxel = np.ones((np.int(total_number), 4))


  voxel3Dx, voxel3Dy, voxel3Dz = np.meshgrid(np.linspace(1,x_size, x_size), 
                                             np.linspace(1,y_size, y_size),
                                             np.linspace(1,z_size, z_size))
  
  l = 0
  for z in voxel3Dx:
    for x in voxel3Dy:
      for y in voxel3Dy:
        voxel[l] = [x, y, z, 1] 
        l=l+1

  return voxel, voxel3Dx, voxel3Dy, voxel3Dz, voxels_number


voxel_size = [0.001, 0.001, 0.001] # size of each voxel

# The dimension limits
xlim = [0.07, -0.04] # [-0.04, 0.07]
ylim = [0.022, 0.132] # [0.04, 0.15]
zlim = [0.07, -0.04] # [-0.04, 0.07]

voxels, voxel3Dx, voxel3Dy, voxel3Dz, voxels_number = InitializeVoxels(xlim, ylim, zlim, voxel_size)

object_points3D = np.copy(voxels).T
voxels[:, 3] = 0 # making the fourth variable of each voxel 0

proj = []

for i in range(N):

  # CAMERA PARAMETERS
  M_ = M[i]

  # PROJECTION TO THE IMAGE PLANE
  points2D = np.matmul(M_, object_points3D)
  points2D = np.floor(points2D / points2D[2, :]).astype(np.int32)
  points2D[np.where(points2D < 0)] = 0; # check for negative image coordinates

  img_size = (silhouettes).shape
  ind1 = np.where(points2D[1, :] >= img_size[0]) # check for out-of-bounds (width) coordinate
  points2D[:, ind1] = 0
  ind1 = np.where(points2D[0, :] >= img_size[1]) # check for out-of-bounds (height) coordinate
  points2D[:, ind1] = 0

  # ACCUMULATE THE VALUE OF EACH VOXEL IN THE CURRENT IMAGE
  voxels[:, 3] += silhouettes[:, :, i].T[points2D.T[:, 0], points2D.T[:, 1]]

  proj.append(points2D)

def ConvertVoxelList2Voxel3D(voxels_number, voxel_size, voxel):
  sx = -(voxels_number[0] / 2) * voxel_size[0]
  ex = voxels_number[0] / 2 * voxel_size[0]

  sy = -(voxels_number[1] / 2) * voxel_size[1]
  ey = voxels_number[1] / 2 * voxel_size[1]

  sz = -(voxels_number[2] / 2) * voxel_size[2] # 0;
  ez = voxels_number[2] / 2 * voxel_size[2] # voxels_number[2] * voxel_size[2]
  voxels_number = np.array(voxels_number).astype(np.int32)
  voxel3D = np.zeros((voxels_number[1] + 1, voxels_number[0] + 1, voxels_number[2] + 1));

  l = 0
  z1 = 0
  for z in np.arange(sz, ez, voxel_size[2]):
      x1 = 0
      for x in np.arange(sx, ex, voxel_size[0]):
          y1 = 0
          for y in np.arange(sy, ey, voxel_size[1]):
              voxel3D[y1, x1, z1] = voxel[l, 3]
              l = l + 1
              y1 = y1 + 1
          x1 = x1 + 1
      z1 = z1 + 1

  return voxel3D

error_amount = 5
maxv = np.max(voxels[:, 3])
iso_value = maxv-np.round(((maxv)/100)*error_amount)-0.5
print('max number of votes:' + str(maxv))
print('threshold for marching cube:' + str(iso_value))

voxel3D = ConvertVoxelList2Voxel3D(np.array(voxels_number), voxel_size, voxels)