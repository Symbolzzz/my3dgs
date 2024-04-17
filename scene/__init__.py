'''
@filename     : __init__.py
@description     :    初始化场景
@time     : 2024/04/15/00
@author     : Enyun Xuan
'''

import os
from scene.gaussian_model import GaussianModel
from scene.dataset_readers import LoadSceneType
class Scene:
    gaussians : GaussianModel
    """这是场景类，需要完成的功能是读入摄像机内外参数，并且用稀疏点云初始化高斯球
    """
    def __init__(self, args, gaussians : GaussianModel):
        self.gaussians = gaussians
        
        # 定义训练相机和测试相机
        self.train_cameras = {}
        self.test_cameras = {}

        # 分别处理Colmap数据集和Blender数据集
        if os.path.exists(os.path.join(args.data, "sparse")):
            print("Loading Colmap dataset...")
            scene_info = LoadSceneType["Colmap"](args.data, "images")
            # TODO: 在这里加入评估模式和不同分辨率下的图片集读取，目前默认处理全分辨率下的图片
        elif os.path.exists(os.path.join(args.data, "transforms_train.json")):
            print("Loading Blender dataset...")
            scene_info = LoadSceneType["Blender"]()

    pass