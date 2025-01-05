import torch
from torch import nn

from .mobileone_blocks import MobileOneBlock
from .DeformFFN import build_norm_layer, build_act_layer
from .DeformFFN import DCNv3, DCNv3_pytorch, DeformFFN

from mmdet3d.registry import MODELS

class ConvBNReLU(nn.Module):
    '''Conv and BN with ReLU activation'''
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=None, groups=1, bias=False):
        super().__init__()
        if padding is None:
            padding = kernel_size // 2
        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                groups=groups,
                bias=bias,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

    def forward(self, x):
        return self.block(x)


class Transpose(nn.Module):
    '''Normal Transpose, default for upsampling'''
    def __init__(self, in_channels, out_channels, kernel_size=2, stride=2):
        super().__init__()
        self.upsample_transpose = nn.ConvTranspose2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            bias=True
        )

    def forward(self, x):
        return self.upsample_transpose(x)


class BiFusion(nn.Module):
    '''BiFusion Block in PAN'''
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.cv1 = ConvBNReLU(in_channels[0], out_channels, 1, 1)
        self.cv2 = ConvBNReLU(in_channels[1], out_channels, 1, 1)
        self.cv3 = ConvBNReLU(out_channels * 3, out_channels, 1, 1)

        self.upsample = Transpose(
            in_channels=out_channels,
            out_channels=out_channels,
        )
        self.downsample = ConvBNReLU(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=2
        )

    def forward(self, x):
        x0 = self.upsample(x[0])
        x1 = self.cv1(x[1])
        x2 = self.downsample(self.cv2(x[2]))
        return self.cv3(torch.cat((x0, x1, x2), dim=1))
    
@MODELS.register_module()
class DeformLayer(nn.Module):
    '''
        The Deformable Convolution Layers to process feature maps
        Inputs:
            Multi-scale feature maps from the backbone or concatenated features or fused features from RepBlocks
        Output:
            Multi-scale feature maps modeling by adaptive receptive field of DCN
        args:
            channels (List or int): the channels of inputs (multi-scale feature maps) or the input (single feature map)
            group (int): the total number of groups
            offset_scale (float): the parameter of DCNv3
            use_ffn (bool): Whether to use the DeformFFN block instead of a single DCNv3
            use_norm (bool): Whether to use the LayerNorm while setting a single DCNv3
    '''
    def __init__(
        self,
        channels,
        group,
        offset_scale = 2.0,
        use_ffn = False,
        use_norm = False
    ):
        super().__init__()
        
        self.channels_list = isinstance(channels, list)

        if not self.channels_list:  # if the input is a single feature map
            assert isinstance(channels, int), 'channels has to be a list of ints or an int'
            if not use_ffn:
                block = [
                    DCNv3(channels=channels, group=group, offset_scale=offset_scale)
                ]
                if use_norm:
                    block.append(build_norm_layer(channels, 'BN'))
                    block.append(build_act_layer('ReLU'))
            else:
                block = [
                    DeformFFN(core_op='DCNv3', channels=channels, groups=group, offset_scale=offset_scale)
                ]
            self.blocks = nn.Sequential(*block)
        else:                       # if the inputs is a list or a tuple of multi-scale feature maps
            blocks = []
            for i, channel in enumerate(channels):
                if not use_ffn:
                    block = [
                        DCNv3(channels=channel, group=group, offset_scale=offset_scale)
                    ]
                    if use_norm:
                        block.append(build_norm_layer(channels, 'BN'))
                        block.append(build_act_layer('ReLU'))
                else:
                    block = [
                        DeformFFN(core_op='DCNv3', channels=channel, groups=group, offset_scale=offset_scale)
                    ]
                block = nn.Sequential(*block)
                blocks.append(block)
        
            self.blocks = nn.ModuleList(blocks)  # to process the multi-scale feature maps by each block

    def forward(self, inputs):
        if self.channels_list:
            outs = []
            for i, x in enumerate(inputs):
                x = self.blocks[i](x.permute((0, 2, 3, 1))).permute((0, 3, 1, 2))
                outs.append(x)
            return tuple(outs)
        else:
            assert isinstance(inputs, torch.Tensor), 'the input has to be a single torch.Tensor for this single DCN block'
            out = self.blocks(inputs.permute((0, 2, 3, 1))).permute((0, 3, 1, 2))
            return out
        
@MODELS.register_module()
class FastDeformLayer(nn.Module):
    '''
        Fast Deformable Convolution Layers to process feature maps before PANNeck
        Inputs:
            Multi-scale feature maps from the backbone
        Output:
            Multi-scale feature maps modeling by adaptive receptive field of DCN
        args:
            channels (List or int): the channels of inputs (multi-scale feature maps) or the input (single feature map)
            group (int): the total number of groups
            dcn_index (List, int): the position of DCN block in this layer
            offset_scale (float): the parameter of DCNv3
            use_ffn (bool): Whether to use the DeformFFN block instead of a single DCNv3
            layer_norm (bool): Whether to use the LayerNorm while setting a single DCNv3
    '''
    def __init__(
        self,
        channels,
        group,
        dcn_index,     # dcn_index: =0, DCN processes first feature map; =1, DCN processes second feature map; =2, DCN processes third feature map; =3, DCN processes fourth feature map
        offset_scale = 2.0,
        use_ffn = False,
        use_norm = False
    ):
        super().__init__()
        
        assert isinstance(dcn_index, list), 'dcn_index has to be a list'
        self.dcn_index = dcn_index

        blocks = []
        for i, channel in enumerate(channels):
            if i in dcn_index:
                if not use_ffn:
                    block = [
                        DCNv3(channels=channel, group=group, offset_scale=offset_scale)
                    ]
                    if use_norm:
                        block.append(build_norm_layer(channels, 'BN'))
                        block.append(build_act_layer('ReLU'))
                else:
                    block = [
                        DeformFFN(core_op='DCNv3', channels=channel, groups=group, offset_scale=offset_scale)
                    ]
            else:
                block = [
                    nn.Conv2d(channel, channel, kernel_size=1, stride=1, padding=0)
                ]
                if use_norm:
                    block.append(build_norm_layer(channels, 'BN'))
                    block.append(build_act_layer('ReLU'))

            block = nn.Sequential(*block)
            blocks.append(block)
        
        self.blocks = nn.ModuleList(blocks)  # to process the multi-scale feature maps by each block

    def forward(self, inputs):
        outs = []
        for i, x in enumerate(inputs):
            if i in self.dcn_index:
                x = self.blocks[i](x.permute((0, 2, 3, 1))).permute((0, 3, 1, 2))
            else:
                x = self.blocks[i](x)
            outs.append(x)
        return tuple(outs)

class DeformBiFusion(nn.Module):
    '''BiFusion Block in PAN'''
    def __init__(self, in_channels, out_channels, group, use_ffn, layer_norm):
        super().__init__()
        self.cv1 = ConvBNReLU(in_channels[0], out_channels, 1, 1)
        self.cv2 = ConvBNReLU(in_channels[1], out_channels, 1, 1)
        self.cv3 = ConvBNReLU(out_channels * 3, out_channels, 1, 1)

        self.deform_layer = DeformLayer(channels=out_channels*3, group=group, 
                                        use_ffn=use_ffn, layer_norm=layer_norm)

        self.upsample = Transpose(
            in_channels=out_channels,
            out_channels=out_channels,
        )
        self.downsample = ConvBNReLU(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=2
        )

    def forward(self, x):
        x0 = self.upsample(x[0])
        x1 = self.cv1(x[1])
        x2 = self.downsample(self.cv2(x[2]))
        x_fuse = self.deform_layer(torch.cat((x0, x1, x2), dim=1))
        return self.cv3(x_fuse)


class RepBlock(nn.Module):
    '''
        RepBlock fuses the concatenated feature maps with the MobileOne Block (dwconv + overparameterized training)
        Fusing the multi-scale feature maps along with the channel dimension
        args:
            in_channels (int): the channels of the concatenated feature map
            out_channels (int): the channels of the outputs
            kernel_size (int): the kernel size of convolution layers in each RepBlock
            stride (int): the stride of fusion layer in this RepBlock
            n (int): the total number of stacked MobileOne Blocks (including the fusion block)
            inference_mode (bool): Whether to define a single-path model for inference
            use_se (bool): Whether to use SE-ReLU as the activation function
            num_conv_branches (int): the number of convolutional layers stacked on the rbr_conv branch
            use_dwconv (bool): Whether to use Depthwise Separate Convolution
            use_normconv (bool): Whether to use normal Convolution
    '''
    def __init__(self, 
                 in_channels, 
                 out_channels,
                 kernel_size=3,
                 stride=1,
                 n=1,           # the depth of RepBlock
                 inference_mode = False,
                 use_se = False,
                 num_conv_branches = 1,
                 use_dwconv = True,
                 use_normconv = False):
        super().__init__()

        self.inference_mode = inference_mode
        self.use_se = use_se
        self.num_conv_branches = num_conv_branches

        assert not use_normconv and use_dwconv, 'only one type of convolutional layers can be built'
        assert use_normconv or use_dwconv, 'must choose one type for convolutional layers'
        self.use_dwconv = use_dwconv
        self.use_normconv = use_normconv

        self.fuse = self._make_stage(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=1)
        self.block = nn.Sequential(*(self._make_stage(out_channels, out_channels, kernel_size=kernel_size, stride=1, padding=1) for _ in range(n - 1))) if n > 1 else None

    def _make_stage(self, 
                    in_planes, 
                    planes,
                    kernel_size=3, 
                    stride=1,
                    padding=None):
        if padding is None:
            padding = kernel_size // 2
        
        if self.use_dwconv:
            # Depthwise conv
            block = [MobileOneBlock(in_channels=in_planes,
                                    out_channels=in_planes,
                                    kernel_size=kernel_size,
                                    stride=stride,
                                    padding=padding,
                                    groups=in_planes,
                                    inference_mode=self.inference_mode,
                                    use_se=self.use_se,
                                    num_conv_branches=self.num_conv_branches)]
        
            # Pointwise conv
            block.append(MobileOneBlock(in_channels=in_planes,
                                        out_channels=planes,
                                        kernel_size=1,
                                        stride=1,
                                        padding=0,
                                        groups=1,
                                        inference_mode=self.inference_mode,
                                        use_se=self.use_se,
                                        num_conv_branches=self.num_conv_branches))
            return nn.Sequential(*block)
        elif self.use_normconv:
            block = MobileOneBlock(in_channels=in_planes,
                                   out_channels=planes,
                                   kernel_size=kernel_size,
                                   stride=stride,
                                   padding=padding,
                                   groups=1,
                                   inference_mode=self.inference_mode,
                                   use_se=self.use_se,
                                   num_conv_branches=self.num_conv_branches)
            return block
        else:
            raise ValueError(f'one of convolution types should be chosen, but use_normconv: {self.use_normconv}, and use_dwconv: {self.use_dwconv}.')
        
    def forward(self, x):
        x = self.fuse(x)
        if self.block is not None:
            x = self.block(x)
        return x