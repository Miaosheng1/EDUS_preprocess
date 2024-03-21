import os
import open3d as o3d
import numpy as np
import json
from pathlib import Path
import yaml
from tqdm import tqdm
import glob
import matplotlib.pyplot as plt

"""这个Lidar 点是KITTI360 在231 服务器上面 Accumulation 起来的 数据, 包含每条Lidar Ray 的起点和终点
   将这些 Lidar 点转换到相机坐标系之下，保存，并进行可视化验证.
   [:3] 是 Lidar 的起点 xyz
   [3:6] 是Lidar 的终点 xyz

"""

X_MIN, X_MAX = -12.8, 12.8
Y_MIN, Y_MAX = -9, 3.8
Z_MIN, Z_MAX = -20, 31.2

def load_from_json(filename: Path):
    """Load a dictionary from a JSON filename.

    Args:
        filename: The filename to load from.
    """
    assert filename.suffix == ".json"
    with open(filename, encoding="UTF-8") as file:
        return json.load(file)

def get_bbx():
    return np.array([X_MIN, Y_MIN, Z_MIN]), np.array([X_MAX, Y_MAX, Z_MAX])

def crop_pointcloud(bbx_min, bbx_max, points):
    mask = (points[:, 0] > bbx_min[0]) & (points[:, 0] < bbx_max[0]) & \
           (points[:, 1] > bbx_min[1]) & (points[:, 1] < bbx_max[1]) & \
           (points[:, 2] > bbx_min[2]) & (points[:, 2] < bbx_max[2])

    return mask

def transform_lidar(pose,pnt):
    pts_camera = np.hstack((pnt, np.ones((pnt.shape[0], 1))))
    pts_camera = np.dot(pts_camera, np.transpose(pose))  ## 得到camera 系下的点
    return pts_camera[:, :3]

def verify_p3d(lidar_xyz,lidar_loc,pose_dir):
    seq_data = pose_dir
    cameras = []

    # 遍历 seq_data['frames'] 中的每个 frame
    for frame in seq_data['frames']:
        # 从每个 frame 的 'transform_matrix' 中提取 camera 位置
        camera_pos = np.array(frame['transform_matrix'])[:3, 3]
        # 将提取的 camera 位置添加到列表中
        cameras.append(camera_pos)

    # 将所有提取的 camera 位置堆叠成一个 numpy 数组
    cam = np.stack(cameras, axis=0)

    pcd_start = o3d.geometry.PointCloud()
    pcd_start.points = o3d.utility.Vector3dVector(lidar_loc)

    pcd_end = o3d.geometry.PointCloud()
    pcd_end.points = o3d.utility.Vector3dVector(lidar_xyz)

    Camera_points = o3d.geometry.PointCloud()
    Camera_points.points = o3d.utility.Vector3dVector(cam)

    # o3d.visualization.draw_geometries([pcd_start] + [pcd_end] + [Camera_points])
    combine_pnt = np.vstack((lidar_loc, lidar_xyz, cam))
    combined_pcd = o3d.geometry.PointCloud()
    z_coords = np.array(combine_pnt[:, 2])
    z_min, z_max = np.min(z_coords), np.max(z_coords)
    colors = plt.cm.jet((z_coords - z_min) / (z_max - z_min))[:, :3]  # 使用 matplotlib 颜色映射

    combined_pcd.points = o3d.utility.Vector3dVector(combine_pnt)
    combined_pcd.colors = o3d.utility.Vector3dVector(colors)
    return combined_pcd



scene_dir_path = os.path.join("E:\80scene","train")

if __name__ == "__main__":
    scenes_dirs = os.listdir(scene_dir_path)

    for i, scene in enumerate(tqdm(scenes_dirs)):

        if i<36:
            seq = "0000"
            continue;
        else:
            seq = "0002"

        current_seq = scene.split('_')[-2]
        pose_data = load_from_json(Path(os.path.join(scene_dir_path, scene, "transforms.json")))

        # lidar_xyz = np.loadtxt("2013_05_28_drive_0000_sync_000295_000345/lidar_points_all.dat")[:, :3]
        # lidar_loc = np.loadtxt('./2013_05_28_drive_0000_sync_000295_000345/lidar_loc.dat')
        lidar_dir = os.path.join(scene_dir_path, scene, "2013_05_28_drive_{}_sync_0{:05d}_*".format(seq,int(current_seq)))
        lidar_dir = glob.glob(lidar_dir)[0]
        lidar_xyz = np.loadtxt(os.path.join(lidar_dir,"lidar_points_all.dat"))[:, :3]
        lidar_loc = np.loadtxt(os.path.join(lidar_dir,"lidar_loc.dat"))


        w2c = np.array(pose_data['inv_pose']).astype(np.float32)
        lidar_xyz = transform_lidar(pose=w2c,pnt=lidar_xyz)
        lidar_loc = transform_lidar(pose=w2c, pnt=lidar_loc)

        bbx_min, bbx_max = get_bbx()
        mask = crop_pointcloud(bbx_min, bbx_max, lidar_xyz)

        lidar_xyz =lidar_xyz[mask]
        lidar_loc =lidar_loc[mask]

        lidar_ray = np.concatenate([lidar_loc,lidar_xyz],axis=1)
        save_dir = os.path.join(scene_dir_path, scene, str(current_seq +"_lidar.npy"))
        np.save(save_dir,lidar_ray)

        combined_pcd = verify_p3d(lidar_xyz=lidar_xyz, lidar_loc=lidar_loc, pose_dir=pose_data)
        o3d.io.write_point_cloud(os.path.join(scene_dir_path, scene,"combined_point_cloud.ply"), combined_pcd)




