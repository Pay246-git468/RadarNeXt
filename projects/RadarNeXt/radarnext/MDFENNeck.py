import torch
from torch import nn

from .DeformFFN import build_norm_layer, build_act_layer
from .common import ConvBNReLU, Transpose, DeformLayer, FastDeformLayer, BiFusion, DeformBiFusion, RepBlock

from mmdet3d.registry import MODELS


class MultiMAPFusion(nn.Module):
    '''
        To fuse the multi-scale feature maps derived from PANNeck
        Inputs:
            Multi-scale feature maps
        Output:
            a fused feature map
        Args:
            in_channels (List, int): the channels of multi-scale feature maps from PANNeck ([64, 128, 256] or [64, 128, 256, 512])
            out_channels (List, int): the channels of the outputs of upsampling layers ([128, 128, 128] or [128, 128, 128, 128])
            strides (List, int): the strides and kernel sizes of upsampling layers ([1, 2] or [1, 2, 4])    
    '''
    def __init__(self,
                 in_channels,
                 out_channels,
                 strides):
        super(MultiMAPFusion, self).__init__()
        blocks = []
        if len(strides) == len(in_channels):
            assert len(strides) == len(out_channels), 'in_channels, out_channels, and strides should be in the same length for upsampling to the largest scale.'
            for i in range(len(in_channels)):
                blocks.append(
                    Transpose(
                    in_channels=in_channels[i],
                    out_channels=out_channels[i],
                    kernel_size=strides[i],
                    stride=strides[i])
                )
        else:
            blocks.append(
                ConvBNReLU(
                    in_channels=in_channels[0],
                    out_channels=out_channels[0],
                    kernel_size=3,
                    stride=2
                )
            )

            for i in range(len(in_channels)-1):
                blocks.append(
                    Transpose(
                        in_channels=in_channels[i+1],
                        out_channels=out_channels[i+1],
                        kernel_size=strides[i],
                        stride=strides[i])
                )
        
        self.blocks = nn.ModuleList(blocks)

        
    
    def forward(self, inputs):
        outs = []

        for i, x in enumerate(inputs):
            outs.append(self.blocks[i](x))
        
        return torch.cat(outs, dim=1)
 
@MODELS.register_module()
class MDFENNeck(nn.Module):       # for DeformPAN (former), DCNs are between concat and RepBlock; for DeformPAN (latter), DCNs are at the end of each branch
    """
    DeformPAN Neck:
        using three2five more DCNs (DeformLayer) to process three multi-scale feature maps before each fusion layers.
        if specify the former_half = True and the latter_half = True, DCNs are before the fusion layers of every branches
        if specify the former_half = True and the latter_half = False, DCNs are before the fusion layers of the first half branches
        if specify the former_half = False and the latter_half = True, DCNs are before the fusion layers of the second half branches
    Inputs:
        three multi-scale feature maps
    Outputs
        three aggregated multi-scale feature maps
    args:
        channels_list (List, int): the channels of four inputs and corresponding outputs
        num_repeats (List, int): the depth of re-parameterizable aggregation stages
        dcn_layer (bool): Whether to use an independent deformable convolution layer
        dcn_index (int): The index of activated deformable convolution for the feature map in a specific scale
        dcn_ids (List, int): The position of deformable convolutions inside of PAN
        former (bool): Whether to activate the DCNs before the fusion layers
        latter (bool): Whether to activate the DCNs after the fusion layers
        group (int): the total number of groups
        use_ffn (bool): Whether to use the DeformFFN block instead of a single DCNv3
        use_norm (bool): Whether to use the LayerNorm while setting a single DCNv3
        inference_mode (bool): Whether to define a single-path model for inference
        use_se (bool): Whether to use SE-ReLU as the activation function
        num_conv_branches (int): the number of convolutional layers stacked on the rbr_conv branch
        use_dwconv (bool): Whether to use Depthwise Separate Convolution
        use_normconv (bool): Whether to use normal Convolution
    """

    def __init__(
        self,
        channels_list=None,  # [64(0), 128(1), 256(2), 128(3), 64(4), 128(5), 256(6)] = inputs + fpn_outs + pan_outs
        num_repeats=None,    # [3, 3, 3, 3]
        dcn_layer = True,    # Default: using independent FastDeformLayer
        dcn_index = [0],       # Default: activating the first layer of FastDeformLayer to process the first feature map
        dcn_ids = [2],
        former = True,
        latter = False,
        group=4,
        use_ffn = False,
        use_norm = False,
        inference_mode=False,
        use_se=False,
        num_conv_branches=1,
        use_dwconv=True,
        use_normconv=False,
        multi_fusion=True,
        fused_channels=[128, 128, 128],
        fusion_strides=[1, 2],
    ):
        super().__init__()

        assert channels_list is not None
        assert num_repeats is not None

        assert not (former and latter), 'former and latter cant be True simultaneously.'
        self.former = former
        self.latter = latter

        # Define the positions of deformable convolutions
        self.dcn_layer = dcn_layer  # True -> deformable convolutions are before the PAN
                                    # False -> deformable convolutions are inside of the PAN
        self.dcn_index = dcn_index  # The position of deformable convolutions when dcn_layer == True
        self.dcn_ids = dcn_ids      # The positions of deformable convolutions when dcn_layer == False

        if dcn_layer:
            self.deform_layer = FastDeformLayer(
                channels=channels_list[0:3],  # this four values are the number of channels of inputs
                dcn_index=dcn_index,
                group=group,
                use_ffn=use_ffn,
                use_norm=use_norm
            )  # To process multi-scale feature maps by DCNs before the aggregation by PAN
        
        
        if not dcn_layer and former and 0 in dcn_ids:
            self.former_deform0 = DeformLayer(
                channels=channels_list[2],
                group=group,
                use_ffn=use_ffn,
                use_norm=use_norm
            )

        self.reduce_layer0 = ConvBNReLU(
            in_channels=channels_list[2],  # 384
            out_channels=channels_list[3], # 256 (fpn_out0)
            kernel_size=1,
            stride=1
        )

        if not dcn_layer and latter and 0 in dcn_ids:
            self.latter_deform0 = DeformLayer(
                channels=channels_list[3],
                group=group,
                use_ffn=use_ffn,
                use_norm=use_norm
            )

        self.upsample0 = Transpose(
            in_channels=channels_list[3],  # 256
            out_channels=channels_list[3], # 256
        )

        if not dcn_layer and former and 1 in dcn_ids:
            self.former_deform1 = DeformLayer(
                channels=channels_list[1] + channels_list[3],
                group=group,
                use_ffn=use_ffn,
                use_norm=use_norm
            )
        
        self.Rep_p4 = RepBlock(
            in_channels=channels_list[1] + channels_list[3],  # 256+256
            out_channels=channels_list[3],                    # 256 (f_out0)
            n=num_repeats[0],
            inference_mode=inference_mode,
            use_se=use_se,
            num_conv_branches=num_conv_branches,
            use_dwconv=use_dwconv,
            use_normconv=use_normconv
        )

        self.reduce_layer1 = ConvBNReLU(
            in_channels=channels_list[3],   # 256
            out_channels=channels_list[4],  # 128 (fpn_out1)
            kernel_size=1,
            stride=1
        )

        if not dcn_layer and latter and 1 in dcn_ids:
            self.latter_deform1 = DeformLayer(
                channels=channels_list[4],
                group=group,
                use_ffn=use_ffn,
                use_norm=use_norm
            )

        self.upsample1 = Transpose(
            in_channels=channels_list[4],  # 128
            out_channels=channels_list[4]  # 128
        )

        if not dcn_layer and former and 2 in dcn_ids:
            self.former_deform2 = DeformLayer(
                channels=channels_list[0] + channels_list[4],
                group=group,
                use_ffn=use_ffn,
                use_norm=use_norm
            )

        self.Rep_p3 = RepBlock(
            in_channels=channels_list[0] + channels_list[4], # 128+128
            out_channels=channels_list[4],                   # 128  (pan_out2)
            n=num_repeats[1],
            inference_mode=inference_mode,
            use_se=use_se,
            num_conv_branches=num_conv_branches,
            use_dwconv=use_dwconv,
            use_normconv=use_normconv
        )
        
        if not dcn_layer and latter and 2 in dcn_ids:
            self.latter_deform2 = DeformLayer(
                channels=channels_list[4],
                group=group,
                use_ffn=use_ffn,
                use_norm=use_norm
            )

        self.downsample2 = ConvBNReLU(
            in_channels=channels_list[4],  # 128
            out_channels=channels_list[4], # 128
            kernel_size=3,
            stride=2
        )

        if not dcn_layer and former and 3 in dcn_ids:
            self.former_deform3 = DeformLayer(
                channels=channels_list[4] + channels_list[4],
                group=group,
                use_ffn=use_ffn,
                use_norm=use_norm
            )

        self.Rep_n3 = RepBlock(
            in_channels=channels_list[4] + channels_list[4],  # 128+128
            out_channels=channels_list[5],                    # 256 (pan_out1)
            n=num_repeats[2],
            inference_mode=inference_mode,
            use_se=use_se,
            num_conv_branches=num_conv_branches,
            use_dwconv=use_dwconv,
            use_normconv=use_normconv
        )

        if not dcn_layer and latter and 3 in dcn_ids:
            self.latter_deform3 = DeformLayer(
                channels=channels_list[5],
                group=group,
                use_ffn=use_ffn,
                use_norm=use_norm
            )

        self.downsample1 = ConvBNReLU(
            in_channels=channels_list[5],  # 256
            out_channels=channels_list[5], # 256
            kernel_size=3,
            stride=2
        )

        if not dcn_layer and former and 4 in dcn_ids:
            self.former_deform4 = DeformLayer(
                channels=channels_list[3] + channels_list[5],
                group=group,
                use_ffn=use_ffn,
                use_norm=use_norm
            )

        self.Rep_n4 = RepBlock(
            in_channels=channels_list[3] + channels_list[5], # 256+256
            out_channels=channels_list[6],                   # 512  (pan_out0)
            n=num_repeats[3],
            inference_mode=inference_mode,
            use_se=use_se,
            num_conv_branches=num_conv_branches,
            use_dwconv=use_dwconv,
            use_normconv=use_normconv
        )

        if not dcn_layer and latter and 4 in dcn_ids:
            self.latter_deform4 = DeformLayer(
                channels=channels_list[6],
                group=group,
                use_ffn=use_ffn,
                use_norm=use_norm
            )
        
        self.multi_fusion = multi_fusion
        if self.multi_fusion:
            self.fusion = MultiMAPFusion(in_channels=channels_list[4:],
                                         out_channels=fused_channels,
                                         strides=fusion_strides)

        

    def forward(self, input):
        # using deformable convolution to process multi-scale feature maps before feeding them into PAN
        if self.dcn_layer:
            (x2, x1, x0) = self.deform_layer(input)
        else:
            # feed the feature maps from backbone to PAN directly
            (x2, x1, x0) = input
        
        # using deformable convolution to process the input of first branch of PAN
        if not self.dcn_layer and self.former and 0 in self.dcn_ids:
            dcn_out0 = self.former_deform0(x0)
        else:
            # the first branch receives a normal input
            dcn_out0 = x0
        
        # using deformable convolution to process the output of first branch of PAN
        if not self.dcn_layer and self.latter and 0 in self.dcn_ids:
            fpn_out0 = self.latter_deform0(self.reduce_layer0(dcn_out0))
        else:
            # the first branch provides a normal output
            fpn_out0 = self.reduce_layer0(dcn_out0)
        
        upsample_feat0 = self.upsample0(fpn_out0)
        f_concat_layer0 = torch.cat([upsample_feat0, x1], 1)
        
        # using deformable convolution to process the input of second branch of PAN
        if not self.dcn_layer and self.former and 1 in self.dcn_ids:
            dcn_out1 = self.former_deform1(f_concat_layer0)
        else:
            # the second branch receives a normal input
            dcn_out1 = f_concat_layer0
        
        f_out0 = self.Rep_p4(dcn_out1)
        
        # using deformable convolution to process the output of second branch of PAN
        if not self.dcn_layer and self.latter and 1 in self.dcn_ids:
            fpn_out1 = self.latter_deform1(self.reduce_layer1(f_out0))
        else:
            # the second branch provides a normal output
            fpn_out1 = self.reduce_layer1(f_out0)
        
        upsample_feat1 = self.upsample1(fpn_out1)
        f_concat_layer1 = torch.cat([upsample_feat1, x2], 1)
        
        # using deformable convolution to process the input of third branch of PAN
        if not self.dcn_layer and self.former and 2 in self.dcn_ids:
            dcn_out2 = self.former_deform2(f_concat_layer1)
        else:
            # the third branch receives a normal input
            dcn_out2 = f_concat_layer1
        
        # using deformable convolution to process the output of third branch of PAN (the first output)
        if not self.dcn_layer and self.latter and 2 in self.dcn_ids:
            pan_out2 = self.latter_deform2(self.Rep_p3(dcn_out2))
        else:
            # the third branch provides a normal output
            pan_out2 = self.Rep_p3(dcn_out2)

        down_feat1 = self.downsample2(pan_out2)
        p_concat_layer1 = torch.cat([down_feat1, fpn_out1], 1)
        
        # using deformable convolution to process the input of fourth branch of PAN
        if not self.dcn_layer and self.former and 3 in self.dcn_ids:
            dcn_out3 = self.former_deform3(p_concat_layer1)
        else:
            # the fourth branch receives a normal input
            dcn_out3 = p_concat_layer1
        
        # using deformable convolution to process the output of fourth branch of PAN (the second output)
        if not self.dcn_layer and self.latter and 3 in self.dcn_ids:
            pan_out1 = self.latter_deform3(self.Rep_n3(dcn_out3))
        else:
            # the fourth branch provides a normal output
            pan_out1 = self.Rep_n3(dcn_out3)

        down_feat0 = self.downsample1(pan_out1)
        p_concat_layer2 = torch.cat([down_feat0, fpn_out0], 1)
        
        # using deformable convolution to process the input of fifth branch of PAN
        if not self.dcn_layer and self.former and 4 in self.dcn_ids:
            dcn_out4 = self.former_deform4(p_concat_layer2)
        else:
            # the fifth branch receives a normal input
            dcn_out4 = p_concat_layer2
        
        # using deformable convolution to process the output of fifth branch of PAN (the third output)
        if not self.dcn_layer and self.latter and 4 in self.dcn_ids:
            pan_out0 = self.latter_deform4(self.Rep_n4(dcn_out4))
        else:
            # the fifth branch provides a normal output
            pan_out0 = self.Rep_n4(dcn_out4)

        outputs = [pan_out2, pan_out1, pan_out0]

        if self.multi_fusion:
            return [self.fusion(outputs)]
        else:
            return outputs