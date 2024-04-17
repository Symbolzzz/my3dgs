"""
@filename     : train.py
@description     :    训练函数，也是程序的入口，读取colmap相机，开始训练
@time     : 2024/04/14/23
@author     : Enyun Xuan
"""

import os
from scene import GaussianModel, Scene
from argparse import ArgumentParser
# 读取colmap相机位姿

# 训练函数
def training(parser : ArgumentParser):
    """训练函数

    Parameters
    ----------
    args : ArgumentParser
        从命令行获取的参数
    """
    args = parser.parse_args()

    first_iteration = 0

    sh_degree = 3 # 球谐函数的阶数默认为3，每个阶有16个基函数

    gaussians = GaussianModel(sh_degree=sh_degree)

    scene = Scene(args=args, gaussians=gaussians)
    
    # 开始训练
    for iteration in range(first_iteration, args.training_iterations):
        # TODO
        pass

        # 选择一个随机的相机
        # TODO

        # 渲染图片
        # TODO

        # 计算loss
        # TODO

        # 反向传播
        # TODO

        # 优化高斯
        # TODO


# 读取命令行参数

if __name__ == "__main__":
    parser = ArgumentParser(description="Training script parameters")

    parser.add_argument("--data", help="imput the path of the colmap, if you don't have one, please colmap first.")

    parser.add_argument("--device", help="the training device.", default="cuda")

    parser.add_argument("--training_iterations", help="the iterations of the training process.", type=int, default=7000)

    parser.add_argument("--output_dir", help="the output path of the results.", default="./output/")

    print(os.path)

    training(parser)