'''
@filename     : dataset_readers.py
@description     :    加载数据集辅助函数
@time     : 2024/04/16/15
@author     : Enyun Xuan
'''

import os
import sys
import numpy as np
from PIL import Image
from typing import NamedTuple
from plyfile import PlyData, PlyElement
from scene.colmap_loader import read_extrinsics_binary, read_intrinsics_binary, qvec2rotmat, read_points3D_binary
from utils.graphics_utils import getWorld2View2, focal2fov, fov2focal, BasicPointCloud



class CameraInfo(NamedTuple):
    """用于存储每个相机和图片匹配之后的参数
    """
    uid: int
    R: np.array
    t: np.array
    FovY: np.array
    FovX: np.array
    image: np.array
    image_path: str
    image_name: str
    width: int
    height: int


class SceneInfo(NamedTuple):
    point_cloud: BasicPointCloud
    train_cameras: list
    test_cameras: list
    nerf_normalization: dict
    ply_path: str


def getNerfppNorm(cam_info):
    """归一化函数，这部分借鉴了NeRF++的思想

    Parameters
    ----------
    cam_info : _type_
        相机的信息
    """
    def get_center_and_diag(cam_centers):
        cam_centers = np.hstack(cam_centers)
        avg_cam_center = np.mean(cam_centers, axis=1, keepdims=True)
        center = avg_cam_center
        dist = np.linalg.norm(cam_centers - center, axis=0, keepdims=True)
        diagonal = np.max(dist)
        return center.flatten(), diagonal

    cam_centers = []

    for cam in cam_info:
        W2C = getWorld2View2(cam.R, cam.t)
        C2W = np.linalg.inv(W2C)
        cam_centers.append(C2W[:3, 3:4])

    center, diagonal = get_center_and_diag(cam_centers)
    radius = diagonal * 1.1

    translate = -center

    return {"translate": translate, "radius": radius}


def store_Ply(path : str, xyzs : tuple, rgbs : tuple):
    """将存储在xyzs和rgbs中的点位置和颜色值存入一个点云文件

    Parameters
    ----------
    path : str
        点云文件路径
    xyzs : tuple
        点坐标
    rgbs : tuple
        颜色
    """
    # 创建法向量
    normals = np.zeros_like(xyzs)
    elements = np.empty(xyzs.shape[0], dtype=[
        ('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
        ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
        ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')
    ])
    attributes = np.concatenate([xyzs, normals, rgbs], axis=1)
    elements[:] = list(map(tuple, attributes))
    vertex_element = PlyElement.describe(elements, "vertex")
    # 存入点云文件
    ply_data = PlyData([vertex_element])
    print("created ply file in {}".format(path))
    ply_data.write(path)


def readColmapCameras(cam_extrinsics : dict, cam_intrinsics : dict, images_folder : str):
    """将Colmap的摄像机内外参数整理在列表中

    Parameters
    ----------
    cam_extrinsics : _type_
        外参数字典
    cam_intrinsics : _type_
        内参数字典
    images_folder : _type_
        图片文件夹路径
    """
    cam_infos = []

    # 遍历摄像机外参数字典
    for idx, key in enumerate(cam_extrinsics):
        # 显示读取摄像机的过程
        sys.stdout.write('\r')
        sys.stdout.write("Reading camera {}/{}".format(idx + 1, len(cam_extrinsics)))
        sys.stdout.flush()

        """
        外参数
        images[image_id] = Image(
                id=image_id, qvec=qvec, tvec=tvec,
                camera_id=camera_id, name=image_name,
                xys=xys, point3D_ids=point3D_ids)
                
                qvec : 旋转四元数
                tvec : 平移向量
        """
        extr = cam_extrinsics[key]
        """
        内参数
        cameras[camera_id] = Camera(id=camera_id,
                                        model=model_name,
                                        width=width,
                                        height=height,
                                        params=np.array(params))
        """
        intr = cam_intrinsics[extr.camera_id]

        # 获得外参数的旋转矩阵和平移向量
        R = np.transpose(qvec2rotmat(extr.qvec))
        t = np.array(extr.tvec)

        if intr.model == "SIMPLE_PINHOLE":
            # 计算出X和Y方向上的视场角
            focal_length_x = intr.params[0]
            FovY = focal2fov(focal_length_x, intr.height)
            FovX = focal2fov(focal_length_x, intr.width)
        elif intr.model == "PINHOLE":
            focal_length_x = intr.params[0]
            focal_length_y = intr.params[1]
            FovX = focal2fov(focal_length_x, intr.height)
            FovY = focal2fov(focal_length_y, intr.width)
        else:
            assert False, "Colmap camera model handles only SIMPLE_PINHOLE and PINHOLE datasets."

        image_path = os.path.join(images_folder, os.path.basename(extr.name))
        image_name = os.path.basename(extr.name).split('.')[0]
        image = Image.open(image_path)

        caminfo = CameraInfo(uid=intr.id, R=R, t=t, FovY=FovY, FovX=FovX, image=image,image_path=image_path, image_name=image_name, width=intr.width, height=intr.height)

        cam_infos.append(caminfo)

    # print(cam_infos)
    print("\nFinish reading cameras.")
    return cam_infos


def readColmapScene(path : str, images : str):
    cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.bin") # 摄像机外参数二进制文件
    cameras_intrinsics_file = os.path.join(path, "sparse/0", "cameras.bin") # 摄像机内参数二进制文件

    cam_extrinsic = read_extrinsics_binary(cameras_extrinsic_file) # 摄像机外参数二进制文件，返回的是一个images字典，key是image的id，value是image的信息
    cam_intrinsic = read_intrinsics_binary(cameras_intrinsics_file) # 摄像机内参数二进制文件，返回的是一个cameras字典，key是camera的id，value是camera的信息

    reading_dir = images

    cam_infos_unsorted = readColmapCameras(cam_extrinsics=cam_extrinsic, cam_intrinsics=cam_intrinsic, images_folder=os.path.join(path, reading_dir))
    # 根据image_name对图片进行排序
    cam_infos = sorted(cam_infos_unsorted.copy(), key=lambda x : x.image_name)

    # 区分训练集和测试集
    # TODO: 评估模式下需要按照一定的比例来划分
    
    train_cam_infos = cam_infos
    test_cam_infos = []

    # 归一化处理
    nerf_normalization = getNerfppNorm(train_cam_infos)

    # TODO: 2024.4.16

    # 从二进制或者文本文件中读取点云（仅在第一次时需要）
    ply_path = os.path.join(path, "sparse/0/points3D.ply")
    if not os.path.exists(ply_path):
        print("Converting the points3D.bin to ply file.")
        bin_path = os.path.join(path, "sparse/0/points3D.bin")
        # 读取点云的坐标和颜色
        xyzs, rgbs, _ = read_points3D_binary(bin_path)
        # xyzs、rgbs的形状是(num_points, 3)
        store_Ply(ply_path, xyzs, rgbs) 

def readBlenderScene():
    # TODO
    pass


# 根据数据集类型选择读取函数
LoadSceneType = {
    "Colmap" : readColmapScene,
    "Blender" : readBlenderScene
}


# test_path = "./debugData/"
# readColmapScene(test_path, "images")
# xyzs, rgbs, _ = read_points3D_binary(test_path)
# store_Ply(test_path, xyzs=xyzs, rgbs=rgbs)
