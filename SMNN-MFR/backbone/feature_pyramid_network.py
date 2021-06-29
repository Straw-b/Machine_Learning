from collections import OrderedDict

import torch.nn as nn
import torch
from torch import Tensor
import torch.nn.functional as F

from torch.jit.annotations import Tuple, List, Dict


class FeaturePyramidNetwork(nn.Module):

    def __init__(self, in_channels_list, out_channels, extra_blocks=None):
        super(FeaturePyramidNetwork, self).__init__()
        # 用来调整resnet特征矩阵(layer1,2,3,4)的channel（kernel_size=1）
        self.inner_blocks = nn.ModuleList()
        # 对调整后的特征矩阵使用3x3的卷积核来得到对应的预测特征矩阵
        self.layer_blocks = nn.ModuleList()
        for in_channels in in_channels_list:
            if in_channels == 0:
                continue
            inner_block_module = nn.Conv2d(in_channels, out_channels, 1)
            layer_block_module = nn.Conv2d(out_channels, out_channels, 3, padding=1)
            self.inner_blocks.append(inner_block_module)
            self.layer_blocks.append(layer_block_module)

        # initialize parameters now to avoid modifying the initialization of top_blocks
        for m in self.children():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight, a=1)
                nn.init.constant_(m.bias, 0)

        self.extra_blocks = extra_blocks

    def get_result_from_inner_blocks(self, x, idx):
        # type: (Tensor, int) -> Tensor

        num_blocks = len(self.inner_blocks)
        if idx < 0:
            idx += num_blocks
        i = 0
        out = x
        for module in self.inner_blocks:
            if i == idx:
                out = module(x)
            i += 1
        return out

    def get_result_from_layer_blocks(self, x, idx):
        # type: (Tensor, int) -> Tensor

        num_blocks = len(self.layer_blocks)
        if idx < 0:
            idx += num_blocks
        i = 0
        out = x
        for module in self.layer_blocks:
            if i == idx:
                out = module(x)
            i += 1
        return out

    def forward(self, x):
        # type: (Dict[str, Tensor]) -> Dict[str, Tensor]

        # unpack OrderedDict into two lists for easier handling
        names = list(x.keys())
        x = list(x.values())

        # 将resnet layer4的channel调整到指定的out_channels
        # last_inner = self.inner_blocks[-1](x[-1])
        last_inner = self.get_result_from_inner_blocks(x[-1], -1)
        # result中保存着每个预测特征层
        results = []
        # 将layer4调整channel后的特征矩阵，通过3x3卷积后得到对应的预测特征矩阵
        # results.append(self.layer_blocks[-1](last_inner))
        results.append(self.get_result_from_layer_blocks(last_inner, -1))

        for idx in range(len(x) - 2, -1, -1):
            inner_lateral = self.get_result_from_inner_blocks(x[idx], idx)
            feat_shape = inner_lateral.shape[-2:]
            inner_top_down = F.interpolate(last_inner, size=feat_shape, mode="nearest")
            last_inner = inner_lateral + inner_top_down
            results.insert(0, self.get_result_from_layer_blocks(last_inner, idx))

        # 在layer4对应的预测特征层基础上生成预测特征矩阵5
        if self.extra_blocks is not None:
            results, names = self.extra_blocks(results, x, names)

        # make it back an OrderedDict
        out = OrderedDict([(k, v) for k, v in zip(names, results)])

        return out


class LastLevelMaxPool(torch.nn.Module):

    def forward(self, x, y, names):
        # type: (List[Tensor], List[Tensor], List[str]) -> Tuple[List[Tensor], List[str]]
        names.append("pool")
        x.append(F.max_pool2d(x[-1], 1, 2, 0))  # input, kernel_size, stride, padding
        return x, names
