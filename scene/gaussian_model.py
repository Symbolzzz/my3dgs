'''
@filename     : gaussian_model.py
@description     :    高斯类
@time     : 2024/04/15/00
@author     : Enyun Xuan
'''

import torch

class GaussianModel:
    """这个类应该包含高斯球的各个属性，并且定义高斯球的优化函数
    """
    def __init__(self, sh_degree : int):
        self.max_sh_degree = sh_degree
        self.curr_sh_degree = 0 # 初始的球谐函数阶数为0，没1000次迭代上升1阶，直到最大阶数为止

        self._xyz = torch.empty(0)
        self._scales = torch.empty(0)
        self._rotation = torch.empty(0)
        self._opacity = torch.empty(0)
        
    # TODO
    pass