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
import yaml
from tqdm import tqdm

""" 
 这个脚本 主要是想实现将 Lidar 3D 点云离散化成 Voxel 然后进行 后续的 3D CNN 的卷积 
 假设 离散化成 shape (3,128,128,512) 的 Voxel, 每个 Voxel 放置的 3 表示 rgb 的颜色

 点云 和 图像序列的 对应关系 在 folder2pcd 文件夹里面，会生成一个对应 yaml 文件
 Voxel_size 设置为 0.1 m
 X = [-12.8,12.8]
 Y = [-9,3.8]
 Z = [-20,31.2]

"""
VIS_CAMERA = False
X_MIN, X_MAX = -12.8, 12.8
Y_MIN, Y_MAX = -9, 3.8
Z_MIN, Z_MAX = -20, 31.2
Voxel_size = 0.1


def load_from_json(filename: Path):
    """Load a dictionary from a JSON filename.

    Args:
        filename: The filename to load from.
    """
    assert filename.suffix == ".json"
    with open(filename, encoding="UTF-8") as file:
        return json.load(file)


def visible_camera_location(seq_data):
    start_camera = np.array(seq_data['frames'][0]['transform_matrix'])[:3, 3]
    end_camera = np.array(seq_data['frames'][-1]['transform_matrix'])[:3, 3]
    camera_loction = np.stack([start_camera, end_camera], axis=0)

    camear_color = np.zeros_like(camera_loction)
    point_camera = o3d.geometry.PointCloud()
    point_camera.points = o3d.utility.Vector3dVector(camera_loction)
    point_camera.colors = o3d.utility.Vector3dVector(camear_color)
    return point_camera


def get_bbx():
    return np.array([X_MIN, Y_MIN, Z_MIN]), np.array([X_MAX, Y_MAX, Z_MAX])


def crop_pointcloud(bbx_min, bbx_max, points, semantic, color):
    mask = (points[:, 0] > bbx_min[0]) & (points[:, 0] < bbx_max[0]) & \
           (points[:, 1] > bbx_min[1]) & (points[:, 1] < bbx_max[1]) & \
           (points[:, 2] > bbx_min[2]) & (points[:, 2] < bbx_max[2])

    return points[mask], semantic[mask], color[mask] / 255.0


"""  这个脚本可以将 Kitti360 的 Lidar 根据 jiaxin 提供的 80 哥徐柳，生成3D 的 semantic voxel
"""

train_id = -1
data_root = "input_pointcloud/lidar_pointcloud"
save_dir = Path("output_pointcloud")
Data_type = "40scene_lidar"
# Data_type = "few_show"
pose_dir = "pose"
os.makedirs(save_dir, exist_ok=True)
Save_Semantics_Voxel = True
scene_dir_path = os.path.join("E:\80scene","train")

def query_pcdfile(current_seq,filename):
    """  Read the pointcloud and the corresponding pose Dict  """
    with open(os.path.join("D:\Code\Tool", f"folder2pcd\seq{current_seq}_pcd_matching.yaml"), 'r') as file:
        pointcloud_idx_list = yaml.safe_load(file)
    return pointcloud_idx_list[filename]



if __name__ == "__main__":
    pcd_dirs = [os.path.join(data_root, f) for f in sorted(os.listdir(data_root))]
    scenes_dirs = os.listdir(scene_dir_path)


    for i, scene in enumerate(tqdm(scenes_dirs)):
        current_seq = scene.split('_')[1]
        pcd_path = query_pcdfile(current_seq=current_seq,filename=scene)

        ply_data = PlyData.read(os.path.join(data_root,f"seq_{current_seq}",pcd_path))
        vertex_element = ply_data['vertex'].data
        ## 发现 vertex 其实是采用 结构化的 numpy 去存储的
        points_x = vertex_element['x']
        points_y = vertex_element['y']
        points_z = vertex_element['z']
        semantic = vertex_element['semantic']
        r, g, b = vertex_element['red'], vertex_element['green'], vertex_element['blue']
        color = np.stack([r, g, b], axis=1)
        semantic = vertex_element['semantic']

        points_data = np.stack([points_x, points_y, points_z], axis=0)
        points_data = np.moveaxis(points_data, -1, 0).astype(np.float32)

        ## 如果生成10个场景的 pointcloud， 将10个场景的 pointcloud 转化到 对应的 10 个 training sequence 各自的 坐标系
        pose_data = load_from_json(Path(os.path.join(scene_dir_path, scene, "transforms.json")))

        w2c = np.array(pose_data['inv_pose']).astype(np.float32)
        pts_camera = np.hstack((points_data, np.ones((points_data.shape[0], 1))))
        pts_camera = np.dot(pts_camera, np.transpose(w2c))  ## 得到camera 系下的点

        bbx_min, bbx_max = get_bbx()
        pts_camera, semantic, colors = crop_pointcloud(bbx_min, bbx_max, pts_camera, semantic, color)

        point_cloud = o3d.geometry.PointCloud()
        point_cloud.points = o3d.utility.Vector3dVector(pts_camera[:, :3])
        # colors = assigncolor(semantic)

        ## 将seamntic id 放入color 属性当中离散化
        if Save_Semantics_Voxel:
            colors = np.stack([semantic, semantic, semantic], axis=1)

        point_cloud.colors = o3d.utility.Vector3dVector(colors)

        ## 可视化 crop 之后的点云
        # o3d.visualization.draw_geometries([point_cloud])
        # o3d.io.write_point_cloud(f"output_pointcloud/{Data_type}/"+f"{image_idx_list[train_id]}.ply", point_cloud)
        # exit()

        """ 点云离散化成 voxel_grid 可视化 """
        voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(point_cloud, voxel_size=Voxel_size)
        # o3d.visualization.draw_geometries([voxel_grid])
        # exit()

        voxels = voxel_grid.get_voxels()

        ## 先创建一个 3D Semantic Volume
        X_Num = int((X_MAX - X_MIN) / Voxel_size)
        Y_Num = int((Y_MAX - Y_MIN) / Voxel_size)
        Z_Num = int((bbx_max[2] - bbx_min[2]) / Voxel_size)
        output_volume = np.zeros([X_Num, Y_Num, Z_Num, 3])

        if Save_Semantics_Voxel:
            for voxel in voxels:
                index = voxel.grid_index
                coords = voxel_grid.get_voxel_center_coordinate(index)
                x_index = int((coords[0] - X_MIN) / Voxel_size)
                y_index = int((coords[1] - Y_MIN) / Voxel_size)
                z_index = int((coords[2] - bbx_min[2]) / Voxel_size)
                semantic_id = np.round(voxel.color[0], 0).clip(0, 26)
                if (z_index >= output_volume.shape[2]) | (x_index >= output_volume.shape[0]) | (
                        y_index >= output_volume.shape[1]):
                    continue
                output_volume[x_index, y_index, z_index] = semantic_id

            if i % 7 == 0:
                voxel_show = output_volume != 0
                occIdx = np.where(voxel_show)
                points = np.concatenate((occIdx[0][:, None] * Voxel_size, \
                                         occIdx[1][:, None] * Voxel_size, \
                                         occIdx[2][:, None] * Voxel_size), axis=1)
                labels = output_volume[voxel_show]
                colors = assigncolor(labels)

                point_camera = o3d.geometry.PointCloud()
                point_camera.points = o3d.utility.Vector3dVector(points)
                point_camera.colors = o3d.utility.Vector3dVector(colors)

                # 加入 相机可视化
                seq_data = pose_data
                start_camera = np.array(seq_data['frames'][0]['transform_matrix'])[:3, 3]
                end_camera = np.array(seq_data['frames'][-1]['transform_matrix'])[:3, 3]
                cam = np.stack([start_camera, end_camera], axis=0)

                ## 减掉 volume 的起点
                cam[:, 0] -= X_MIN
                cam[:, 1] -= Y_MIN
                cam[:, 2] -= Z_MIN

                camear_color = np.zeros_like(cam)
                camear_color[:, 0] = 1
                points = o3d.geometry.PointCloud()
                points.points = o3d.utility.Vector3dVector(cam)
                points.colors = o3d.utility.Vector3dVector(camear_color)

                o3d.visualization.draw_geometries([point_camera] + [points])
            scene_id = int(scene.split('_')[-2])
            voxel_dir = os.path.join(scene_dir_path, scene, "voxel")
            save_name = str(scene_id) + "_semantic.npy"
            np.save(os.path.join(voxel_dir, save_name), output_volume[...,0])

        else:
            for voxel in voxels:
                index = voxel.grid_index
                coords = voxel_grid.get_voxel_center_coordinate(index)
                x_index = int((coords[0] - X_MIN) / Voxel_size)
                y_index = int((coords[1] - Y_MIN) / Voxel_size)
                z_index = int((coords[2] - bbx_min[2]) / Voxel_size)
                if (z_index >= output_volume.shape[2]) | (x_index >= output_volume.shape[0]) | (
                        y_index >= output_volume.shape[1]):
                    continue
                output_volume[x_index, y_index, z_index] = voxel.color.astype(np.float32)
            np.save(str(save_dir) + '/' + Data_type + f"/{image_idx_list[train_id]}_volume.npy", output_volume)
