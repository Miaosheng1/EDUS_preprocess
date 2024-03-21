import os
import open3d as o3d
import numpy as np
import torch
import pickle
import math
from typing import Tuple, List, Dict, Iterable
from plyfile import PlyData, PlyElement
from kitti_label import id2label, assigncolor
import json
from pathlib import Path
import math
from tqdm import tqdm
from PIL import Image
import cv2
from scipy.signal import convolve2d
from scipy.ndimage import maximum_filter


"""  这个脚本可以将 Lidar 点的 世界坐标投影到 相机帧上，得到 Sparse 的深度图 
    
    这种投影得到的深度图是有一些 BUG，还应该考虑观察角度，假设存在一条和Z平面平行的 直线， 其 Z值大小是相同的，因此投影到像素平面上的
    深度值Z 也是相同的。但是 因为相机观测视角的 差异，实际上 观测出这条射线上的 depth 应该不一样。
                    * * * * * *
                      \     /
                       \  / 
                        *
"""
Focal = 552.554261
CX = 682.049453
CY = 238.769549
K = np.array([
    [Focal, 0, CX, 0],
    [0, Focal, CY, 0],
    [0, 0, 1, 0],
    [0, 0, 0, 1]
])
kernel = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]])

Image_height = 376
Image_width = 1408

def load_from_json(filename: Path):
    """Load a dictionary from a JSON filename.

    Args:
        filename: The filename to load from.
    """
    assert filename.suffix == ".json"
    with open(filename, encoding="UTF-8") as file:
        return json.load(file)

def inverse_rigid_trans(Tr):
    ''' Inverse a rigid body transform matrix (3x4 as [R|t])
        [R'|-R't; 0|1]
    '''
    inv_Tr = np.zeros_like(Tr)  # 3x4
    inv_Tr[0:3, 0:3] = np.transpose(Tr[0:3, 0:3])
    inv_Tr[0:3, 3] = np.dot(-np.transpose(Tr[0:3, 0:3]), Tr[0:3, 3])
    return inv_Tr

def w2c(points,camera_pose):
    points = np.hstack((points, np.ones((points.shape[0], 1))))  ## cam 系的3D点
    camera_pose = camera_pose[:3,:]
    world2camera = inverse_rigid_trans(camera_pose)
    pts_camera = np.dot(points, np.transpose(world2camera))  ## 将cam 系的点转到 lidar
    return pts_camera[:,:3]


save_dir = f"depth_"
os.makedirs(save_dir + "/mask",exist_ok=True)
lidar_path = "output_pointcloud/40scene_lidar/295.ply"
pose_dir = "pose"
Data_type = "40scene_stereo"
id = 295
meta = load_from_json(Path(pose_dir) / Path(Data_type) / Path(f"nerfacto_{id}_40") / "transforms.json")

ply_data = PlyData.read(lidar_path)
vertex_element = ply_data['vertex'].data
## 发现 vertex 其实是采用 结构化的 numpy 去存储的
points_x = vertex_element['x']
points_y = vertex_element['y']
points_z = vertex_element['z']
r, g, b = vertex_element['red'], vertex_element['green'], vertex_element['blue']
color = np.stack([r, g, b], axis=1) / 255.0

points_data = np.stack([points_x,points_y,points_z],axis=0)
points_data = np.moveaxis(points_data, -1, 0).astype(np.float32)
points_data1 = np.stack([points_data[:,0], points_data[:,1],points_data[:,2] +0.02],axis=1)
points_data2 = np.stack([points_data[:,0], points_data[:,1],points_data[:,2] +0.04],axis=1)

points_data = np.concatenate([points_data,points_data1,points_data2],axis=0)
color = np.concatenate([color,color,color],axis=0)
point_cloud = o3d.geometry.PointCloud()
point_cloud.points = o3d.utility.Vector3dVector(points_data)
point_cloud.colors = o3d.utility.Vector3dVector(color)


poses = []
images = []

for frame in meta["frames"]:
    ## 将pose 从 opengl 系转化到 opencv 系
    pose = np.array(frame["transform_matrix"]) * np.array([1, -1, -1,1])
    poses.append(pose)
    images.append(frame["file_path"])

num_images = len(images)
for i in tqdm(range(num_images)):
## 1. 保证点云的坐标和 camera 处于同一个坐标系，将点云的坐标转换到 当前相机的 Camera 系
    point_c = w2c(points_data.copy(),poses[i])
    z_buffer = point_c[:,2]
    ## 2. 将camera 系的点通过K 投影到图像坐标系
    uv = np.matmul(K[None,:3, :3], point_c[...,None]).squeeze(-1)
    uv[:, :2] = (uv[:, :2] / uv[:, -1:])
    uv_coords = np.round(uv[:, :2]).astype(np.int32)
    depth_image = np.zeros((Image_height,Image_width),dtype=np.float32)
    color_image = np.zeros((Image_height,Image_width,3),dtype=np.uint8)

    valid_indices = np.where((uv_coords[:, 0] >= 0) & (uv_coords[:, 0] < Image_width) & (uv_coords[:, 1] >= 0) & (uv_coords[:, 1] < Image_height ) & (uv[:,-1] > 0))[0]

    color_image[uv_coords[valid_indices, 1],uv_coords[valid_indices, 0]] = color[valid_indices]*255

    color_image = Image.fromarray(color_image)
    color_image.save(f"{save_dir}/color_{i}.png")

    depth_image[uv_coords[valid_indices, 1],uv_coords[valid_indices, 0]] = z_buffer[valid_indices]
    pred_depth = cv2.applyColorMap(cv2.convertScaleAbs(((depth_image/depth_image.max()) * 255).astype(np.uint8),alpha=2), cv2.COLORMAP_JET)
    # pred_depth = np.save(f"{save_dir}/mask/{images[i].split('.')[0]}.npy",depth_image)
    cv2.imwrite(f"{save_dir}/mask/depth_{i}.png", pred_depth)








