import warnings
from typing import Optional, Sequence, Tuple

from mmengine.model import BaseModule
from torch import Tensor
from torch import nn as nn

from mmdet3d.registry import MODELS
from mmdet3d.utils import ConfigType, OptMultiConfig

from .common import RepBlock

@MODELS.register_module()
class RepDWC(BaseModule):
    """Re-parameterizable backbone with MobileOne's Architecture

    Args:
        in_channels (int): Input channels.
        out_channels (list[int]): Output channels for multi-scale feature maps.
        layer_nums (list[int]): Number of layers in each stage.
        layer_strides (list[int]): Strides of each stage.
        inference_mode (bool): Whether to define the re-parameterized model for inference
        use_se (bool): Whether to use the SE-ReLU with more parameters
        num_conv_branches (int): The number of convolutional branches of the model during training
        num_outputs (int): The total number of outputs  
    """

    def __init__(self,
                 in_channels: int = 128,                        # 64
                 out_channels: Sequence[int] = [128, 128, 256], # [64, 128, 256, 256/512]
                 layer_nums: Sequence[int] = [3, 5, 5],         # [2, 8, 10, 1]
                 layer_strides: Sequence[int] = [2, 2, 2],      # [2, 2, 2, 2] -> block1: H/2, W/2; block2: H/4, W/4; 
                 inference_mode: bool = False,                  # block3: H/8, W/8; Block4: H/16, W/16 or H/8, W/8 for stride=1;
                 use_se: bool = False,
                 num_conv_branches: int = 1,
                 num_outputs: int = 3,
                 use_normconv: bool = False,
                 use_dwconv: bool = True,
                 init_cfg: OptMultiConfig = None,
                 pretrained: Optional[str] = None) -> None:
        super(RepDWC, self).__init__(init_cfg=init_cfg)
        assert len(layer_strides) == len(layer_nums)
        assert len(out_channels) == len(layer_nums)
        assert not use_normconv and use_dwconv, 'only one type of convolutional layers can be built'
        assert use_normconv or use_dwconv, 'must choose one type for convolutional layers'
        self.num_outputs = num_outputs

        in_filters = [in_channels, *out_channels[:-1]]
        # note that when stride > 1, conv2d with same padding isn't
        # equal to pad-conv2d. we should use pad-conv2d.
        blocks = []
        for i, layer_num in enumerate(layer_nums):
            blocks.append(
                RepBlock(
                    in_channels=in_filters[i],
                    out_channels=out_channels[i],
                    kernel_size=3,
                    stride=layer_strides[i],
                    n=layer_num,
                    inference_mode=inference_mode,
                    use_se=use_se,
                    num_conv_branches=num_conv_branches,
                    use_dwconv=use_dwconv,
                    use_normconv=use_normconv
                )
            )

        self.blocks = nn.ModuleList(blocks)

        assert not (init_cfg and pretrained), \
            'init_cfg and pretrained cannot be setting at the same time'
        if isinstance(pretrained, str):
            warnings.warn('DeprecationWarning: pretrained is a deprecated, '
                          'please use "init_cfg" instead')
            self.init_cfg = dict(type='Pretrained', checkpoint=pretrained)
        else:
            self.init_cfg = dict(type='Kaiming', layer='Conv2d')

    def forward(self, x: Tensor) -> Tuple[Tensor, ...]:
        """Forward function.

        Args:
            x (torch.Tensor): Input with shape (N, C, H, W).

        Returns:
            tuple[torch.Tensor]: Multi-scale features.
        """
        outs = []
        for i in range(len(self.blocks)):
            x = self.blocks[i](x)
            if i >= (len(self.blocks) - self.num_outputs): # only return the outputs of the last three blocks
                outs.append(x)
        return tuple(outs)