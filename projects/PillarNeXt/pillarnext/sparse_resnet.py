import torch
from torch import nn
import spconv
import spconv.pytorch
from spconv.pytorch import SparseSequential, SparseConv2d

from projects.PillarNeXt.pillarnext.utils import SparseConvBlock, SparseBasicBlock

from mmdet3d.registry import MODELS


@MODELS.register_module()
class SparseResNet(spconv.pytorch.SparseModule):
    def __init__(
            self,
            layer_nums,          # [2, 2, 2, 2]
            ds_layer_strides,    # [1, 2, 2, 2]
            ds_num_filters,      # [64, 128, 256, 256]
            num_input_features,  # 64
            kernel_size=[3, 3, 3, 3],
            out_channels=256):

        super(SparseResNet, self).__init__()
        self._layer_strides = ds_layer_strides
        self._num_filters = ds_num_filters
        self._layer_nums = layer_nums
        self._num_input_features = num_input_features

        assert len(self._layer_strides) == len(self._layer_nums)
        assert len(self._num_filters) == len(self._layer_nums)

        in_filters = [self._num_input_features, *self._num_filters[:-1]]
        blocks = []

        for i, layer_num in enumerate(self._layer_nums):
            block = self._make_layer(
                in_filters[i],          # [64, 64, 128, 256]
                self._num_filters[i],   # [64, 128, 256, 256]
                kernel_size[i],         # [3, 3, 3, 3]
                self._layer_strides[i], # [1, 2, 2, 2]
                layer_num)              # [2, 2, 2, 2]
            blocks.append(block)

        self.blocks = nn.ModuleList(blocks)

        self.mapping = SparseSequential(
            SparseConv2d(self._num_filters[-1],            # 256
                         out_channels, 1, 1, bias=False),  # 256
            nn.BatchNorm1d(out_channels, eps=1e-3, momentum=0.01),
            nn.ReLU(),
        )

    def _make_layer(self, inplanes, planes, kernel_size, stride, num_blocks):

        layers = []
        layers.append(SparseConvBlock(inplanes, planes,
                      kernel_size=kernel_size, stride=stride, use_subm=False))

        for j in range(num_blocks):
            layers.append(SparseBasicBlock(planes, kernel_size=kernel_size))

        return spconv.pytorch.SparseSequential(*layers)

    def forward(self, pillar_features, coors, input_shape):
        batch_size = len(torch.unique(coors[:, 0]))
        x = spconv.pytorch.SparseConvTensor(
            pillar_features, coors, input_shape, batch_size)
        for i in range(len(self.blocks)):
            x = self.blocks[i](x)
        x = self.mapping(x)
        return x.dense()