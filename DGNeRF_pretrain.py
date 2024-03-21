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
from tqdm import tqdm


"""
 这个脚本 主要是 EDUS pretrain 的 数据预处理脚本
 jiaxin 得到的点云已经将点云和 Pose 处于同一个坐标系，这里将ply 文件转换成 numpy 文件，
 并进行了可视化的验证 和 Voxelize 的操作
 Voxel_size 设置为 0.2 m
 X = [-12.8,12.8]
 Y = [-9,3.8]
 Z = [-20,31.2]
 
"""

X_MIN,X_MAX = -12.8, 12.8
Y_MIN,Y_MAX = -9, 3.8
Z_MIN,Z_MAX = -20, 31.2
Voxel_size = 0.2

def load_from_json(filename: Path):
    """Load a dictionary from a JSON filename.

    Args:
        filename: The filename to load from.
    """
    assert filename.suffix == ".json"
    with open(filename, encoding="UTF-8") as file:
        return json.load(file)

def verify_p3d(color_volume,pose_dir,voxel_dir):
    seq_data = load_from_json(pose_dir)
    start_camera = np.array(seq_data['frames'][0]['transform_matrix'])[:3, 3]
    mid_camera = np.array(seq_data['frames'][2]['transform_matrix'])[:3, 3]
    end_camera = np.array(seq_data['frames'][-1]['transform_matrix'])[:3, 3]
    cam = np.stack([start_camera, end_camera, mid_camera], axis=0)

    ## 减掉 volume 的起点
    cam[:, 0] -= X_MIN
    cam[:, 1] -= Y_MIN
    cam[:, 2] -= Z_MIN

    camear_color = np.zeros_like(cam)
    camear_color[:, 0] = 1
    Camera_points = o3d.geometry.PointCloud()
    Camera_points.points = o3d.utility.Vector3dVector(cam)
    Camera_points.colors = o3d.utility.Vector3dVector(camear_color)


    voxel_show = color_volume != 0
    occIdx = np.where(voxel_show)
    points = np.concatenate((occIdx[0][:, None] * Voxel_size, \
                             occIdx[1][:, None] * Voxel_size, \
                             occIdx[2][:, None] * Voxel_size), axis=1)

    colors = color_volume[occIdx[0], occIdx[1], occIdx[2]][:, :3]

    point_camera = o3d.geometry.PointCloud()
    point_camera.points = o3d.utility.Vector3dVector(points)
    point_camera.colors = o3d.utility.Vector3dVector(colors)

    o3d.visualization.draw_geometries([point_camera] + [Camera_points])
    # o3d.io.write_point_cloud([pointcloud] + [points],os.path.join(voxel_dir,"re.ply"))

root_dir = os.path.join("E:\80scene","train")
seqs = os.listdir(root_dir)
num_scenes = len(seqs)

voxel_type = "mono_voxel"  ## "voxel"

if __name__== "__main__":
    for i in tqdm(range(25,num_scenes)):
        scene_id = int(seqs[i].split('_')[-2])
        pcd_filename = f"{scene_id}.ply"
        voxel_dir = os.path.join(root_dir,seqs[i],voxel_type)
        pose_dir = Path(os.path.join(root_dir,seqs[i],"transforms.json"))

        ply_data = PlyData.read(os.path.join(voxel_dir,pcd_filename))
        vertex_element = ply_data['vertex'].data

        points_x = vertex_element['x']
        points_y = vertex_element['y']
        points_z = vertex_element['z']
        r, g, b = vertex_element['red'], vertex_element['green'], vertex_element['blue']
        color = np.stack([r, g, b], axis=1) / 255
        position = np.stack([points_x, points_y, points_z], axis=1)

        point_cloud = o3d.geometry.PointCloud()
        point_cloud.points = o3d.utility.Vector3dVector(position)
        point_cloud.colors = o3d.utility.Vector3dVector(color)

        ## filter noisy pcd
        cl, ind = point_cloud.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
        point_cloud = point_cloud.select_by_index(ind)
        # o3d.visualization.draw_geometries([point_cloud])
        # exit()
        voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(point_cloud,voxel_size=Voxel_size)

        """visualize voxel grid"""
        # o3d.visualization.draw_geometries([voxel_grid])

        X_Num = int((X_MAX - X_MIN) / Voxel_size)
        Y_Num = int((Y_MAX - Y_MIN) / Voxel_size)
        Z_Num = int((Z_MAX - Z_MIN) / Voxel_size)
        output_volume = np.zeros([X_Num,Y_Num,Z_Num,3])

        voxels = voxel_grid.get_voxels()

        for voxel in voxels:
            index = voxel.grid_index
            coords = voxel_grid.get_voxel_center_coordinate(index)
            x_index = int( (coords[0] - X_MIN )/ Voxel_size)
            y_index = int((coords[1] - Y_MIN) / Voxel_size)
            z_index = int((coords[2] - Z_MIN) / Voxel_size)
            if (z_index >= output_volume.shape[2]) | (x_index >= output_volume.shape[0]) | (y_index >= output_volume.shape[1]):
                continue
            output_volume[x_index,y_index,z_index] = voxel.color.astype(np.float32)

        "verify the volume"
        if i % 10 == 0:
            verify_p3d(output_volume, pose_dir, voxel_dir)

        # save_name = str(scene_id) + "_volume.npy"
        save_name = str(scene_id) + voxel_type +  "_volume.npy"
        np.save(os.path.join(voxel_dir,save_name), output_volume)

