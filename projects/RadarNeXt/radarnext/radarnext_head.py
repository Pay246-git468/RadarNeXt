import logging

import torch
from mmengine.logging import print_log
from mmengine.structures import InstanceData
from projects.PillarNeXt.pillarnext.utils.box_torch_ops import rotate_nms_pcdet

import copy
from torch import nn
from mmdet.models.utils import multi_apply
from mmdet3d.registry import MODELS
from mmdet3d.structures import center_to_corner_box2d
from mmdet3d.models.utils import draw_heatmap_gaussian, gaussian_radius
from projects.PillarNeXt.pillarnext.loss import FastFocalLoss, RegLoss, IouLoss, IouRegLoss
from projects.PillarNeXt.pillarnext.utils.conv import ConvBlock

from .common import ConvBNReLU, Transpose


class SepHead(nn.Module):
    def __init__(
        self,
        in_channels,
        heads,
        stride=1,
        head_conv=64,
        final_kernel=1,
        bn=True,
        init_bias=-2.19,
        **kwargs,
    ):
        super(SepHead, self).__init__(**kwargs)
        if stride > 1:
            self.deblock = ConvBlock(in_channels, head_conv, kernel_size=int(stride), 
                            stride=int(stride), padding=0, conv_layer=nn.ConvTranspose2d)
            in_channels = head_conv
        else:
            self.deblock = nn.Identity()
        self.heads = heads
        for head in self.heads:
            classes, num_conv = self.heads[head]

            fc = nn.Sequential()
            for i in range(num_conv-1):
                fc.append(nn.Conv2d(in_channels, head_conv,
                                    kernel_size=final_kernel, stride=1,
                                    padding=final_kernel // 2, bias=True))
                if bn:
                    fc.append(nn.BatchNorm2d(head_conv))
                fc.append(nn.ReLU())

            fc.append(nn.Conv2d(head_conv, classes,
                                kernel_size=final_kernel,  stride=1,
                                padding=final_kernel // 2, bias=True))

            if 'hm' in head:
                fc[-1].bias.data.fill_(init_bias)

            self.__setattr__(head, fc)

    def forward(self, x):
        x = self.deblock(x)

        ret_dict = dict()
        for head in self.heads:
            if not self.training and head == 'corner_hm':  # When inference, deactivate the auxiliary classification head on corner points
                continue    
            ret_dict[head] = self.__getattr__(head)(x)

        return ret_dict
    
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
class RadarNeXt_Head(nn.Module):
    def __init__(
            self,
            in_channels,
            multi_fusion,
            fusion_channels,
            fusion_strides,
            tasks,
            weight,
            corner_weight,
            iou_weight,
            iou_reg_weight,
            code_weights,
            common_heads,
            strides,
            logger=None,
            init_bias=-2.19,
            share_conv_channel=64,
            num_hm_conv=2,
            num_corner_hm_conv=2,
            with_corner=False,
            with_reg_iou=False,
            voxel_size=None,
            pc_range=None,
            out_size_factor=None,
            rectifier=[[0.], [0.], [0.]],
            bbox_code_size=7,
            train_cfg=None,
            test_cfg=None):

        super(RadarNeXt_Head, self).__init__()

        # num_classes = [len(t) for t in tasks]
        # self.class_names = tasks

        # by jly
        num_classes = [len(t['class_names']) for t in tasks]
        self.class_names = [t['class_names'] for t in tasks]

        self.code_weights = code_weights
        self.weight = weight  # weight between hm loss and loc loss
        self.corner_weight = corner_weight
        self.iou_weight = iou_weight
        self.iou_reg_weight = iou_reg_weight

        self.in_channels = in_channels
        self.num_classes = num_classes
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

        self.crit = FastFocalLoss()
        self.crit_reg = RegLoss()

        self.with_corner = with_corner
        self.with_reg_iou = with_reg_iou
        if self.with_corner and self.training:
            self.corner_crit = torch.nn.MSELoss(reduction='none')
        if self.with_reg_iou:
            self.crit_iou_reg = IouRegLoss()

        self.with_iou = 'iou' in common_heads
        if self.with_iou:
            self.crit_iou = IouLoss()

        if self.with_iou or with_reg_iou:
            self.voxel_size = voxel_size
            self.pc_range = pc_range
            self.out_size_factor = out_size_factor
        
        if not logger:
            logger = logging.getLogger('CenterFormerBboxHead')
        self.logger = logger

        logger.info(f'num_classes: {num_classes}')

        self.strides = strides

        self.rectifier = rectifier

        self.bbox_code_size = bbox_code_size  # by jly

        self.multi_fusion = multi_fusion
        if self.multi_fusion:
            self.fuse = MultiMAPFusion(
                in_channels=in_channels,
                out_channels=fusion_channels,
                strides=fusion_strides,
            )
            channels = sum(fusion_channels)
        else:
            channels = in_channels

        # a shared convolution
        self.shared_conv = nn.Sequential(
            nn.Conv2d(channels, share_conv_channel,
                      kernel_size=3, padding=1, bias=True),
            nn.BatchNorm2d(share_conv_channel),
            nn.ReLU(inplace=True)
        )

        self.tasks = nn.ModuleList()
        print_log(f'Use HM Bias: {init_bias}', 'current')

        for (num_cls, stride) in zip(num_classes, strides):
            heads = copy.deepcopy(common_heads)
            if with_corner:
                heads.update(dict(hm=(num_cls, num_hm_conv), corner_hm=(1, num_corner_hm_conv)))
            else:
                heads.update(dict(hm=(num_cls, num_hm_conv)))
            self.tasks.append(
                SepHead(share_conv_channel, heads, stride=stride,
                        bn=True, init_bias=init_bias, final_kernel=3)
            )
        
        logger.info('Finish RepPillarHead Initialization')

    def forward(self, x, *kwargs):
        ret_dicts = []
        if self.multi_fusion:
            x = self.shared_conv(self.fuse(x))
        else:
            x = self.shared_conv(x[0])

        for task in self.tasks:
            ret_dicts.append(task(x))

        return ret_dicts

    def _sigmoid(self, x):
        y = torch.clamp(x.sigmoid_(), min=1e-4, max=1-1e-4)
        return y

    def loss(self, feats, batch_data_samples, *args, **kwargs):
        """Forward function for point cloud branch.

        Args:
            pts_feats (list[torch.Tensor]): Features of point cloud branch
            batch_data_samples (List[:obj:`Det3DDataSample`]): The Data
                Samples. It usually includes information such as
                `gt_instance_3d`, .

        Returns:
            dict: Losses of each branch.
        """
        preds_dicts = self(feats)

        batch_gt_instance_3d = []
        for data_sample in batch_data_samples:
            batch_gt_instance_3d.append(data_sample.gt_instances_3d)
        losses = self.loss_by_feat(preds_dicts, batch_gt_instance_3d)
        return losses
    
    def loss_by_feat(self, preds_dicts, batch_gt_instances_3d, **kwargs):
        heatmaps, anno_boxes, gt_inds, gt_masks, corner_heatmaps, cat_labels, gt_boxes = self.get_targets(batch_gt_instances_3d)
        
        losses = {}
        for task_id, preds_dict in enumerate(preds_dicts):
            # heatmap focal loss
            preds_dict['hm'] = self._sigmoid(preds_dict['hm'])

            hm_loss = self.crit(preds_dict['hm'], heatmaps[task_id], gt_inds[task_id], 
                                gt_masks[task_id], cat_labels[task_id])
            
            if self.with_corner:
                preds_dict['corner_hm'] = self._sigmoid(preds_dict['corner_hm'])
                corner_loss = self.corner_crit(preds_dict['corner_hm'],
                                               corner_heatmaps[task_id])
                corner_mask = (corner_heatmaps[task_id] > 0).to(corner_loss)
                corner_loss = (corner_loss * corner_mask).sum() / (
                    corner_mask.sum() + 1e-4)
                losses.update({
                    f'{task_id}_corner_loss':
                    corner_loss * self.corner_weight
                })
            
            target_box = anno_boxes[task_id]
            # reconstruct the anno_box from multiple reg heads
            if 'vel' in preds_dict:
                preds_dict['anno_box'] = torch.cat((preds_dict['reg'], preds_dict['height'], preds_dict['dim'],
                                                    preds_dict['vel'], preds_dict['rot']), dim=1)
            else:
                preds_dict['anno_box'] = torch.cat((preds_dict['reg'], preds_dict['height'], preds_dict['dim'],
                                                    preds_dict['rot']), dim=1)

            # Regression loss for dimension, offset, height, rotation
            box_loss = self.crit_reg(
                preds_dict['anno_box'], gt_masks[task_id], gt_inds[task_id], target_box)

            loc_loss = (box_loss*box_loss.new_tensor(self.code_weights)).sum()

            losses.update({
                f'{task_id}_hm_loss': hm_loss,
                f'{task_id}_loc_loss': loc_loss * self.weight
            })

            if self.with_iou or self.with_reg_iou:
                batch_dim = torch.exp(torch.clamp(
                    preds_dict['dim'], min=-5, max=5))
                batch_dim = batch_dim.permute(0, 2, 3, 1).contiguous()
                batch_rot = preds_dict['rot'].clone()
                batch_rot = batch_rot.permute(0, 2, 3, 1).contiguous()
                batch_rots = batch_rot[..., 0:1]
                batch_rotc = batch_rot[..., 1:2]
                batch_rot = torch.atan2(batch_rots, batch_rotc)
                batch_reg = preds_dict['reg'].clone().permute(
                    0, 2, 3, 1).contiguous()
                batch_hei = preds_dict['height'].clone().permute(
                    0, 2, 3, 1).contiguous()

                batch, H, W, _ = batch_dim.size()

                batch_reg = batch_reg.reshape(batch, H * W, 2)
                batch_hei = batch_hei.reshape(batch, H * W, 1)

                batch_rot = batch_rot.reshape(batch, H * W, 1)
                batch_dim = batch_dim.reshape(batch, H * W, 3)

                ys, xs = torch.meshgrid(
                    [torch.arange(0, H), torch.arange(0, W)])
                ys = ys.view(1, H, W).repeat(batch, 1, 1).to(batch_dim)
                xs = xs.view(1, H, W).repeat(batch, 1, 1).to(batch_dim)

                xs = xs.view(batch, -1, 1) + batch_reg[:, :, 0:1]
                ys = ys.view(batch, -1, 1) + batch_reg[:, :, 1:2]

                xs = xs * self.out_size_factor * \
                    self.voxel_size[0] + self.pc_range[0]
                ys = ys * self.out_size_factor * \
                    self.voxel_size[1] + self.pc_range[1]

                batch_box_preds = torch.cat(
                    [xs, ys, batch_hei, batch_dim, batch_rot], dim=2)
                batch_box_preds = batch_box_preds.permute(
                    0, 2, 1).contiguous().reshape(batch, -1, H, W)

                if self.with_iou:
                    pred_boxes_for_iou = batch_box_preds.detach()
                    iou_loss = self.crit_iou(preds_dict['iou'], gt_masks[task_id], gt_inds[task_id],
                                             pred_boxes_for_iou, gt_boxes[task_id])
                    losses.update({
                    f'{task_id}_iou_loss':
                    iou_loss * self.iou_weight
                    })

                if self.with_reg_iou:
                    iou_reg_loss = self.crit_iou_reg(batch_box_preds, gt_masks[task_id], gt_inds[task_id],
                                                     gt_boxes[task_id])
                    losses.update({
                    f'{task_id}_iou_reg_loss':
                    iou_reg_loss * self.iou_reg_weight
                    })

        return losses

    @torch.no_grad()
    def predict(self, feats, batch_input_metas):
        """decode, nms, then return the detection result. Additionaly support double flip testing
        """
        preds_dicts = self(feats)
        
        # get loss info
        rets = []

        post_center_range = self.test_cfg.post_center_limit_range
        if len(post_center_range) > 0:
            post_center_range = torch.tensor(
                post_center_range,
                dtype=preds_dicts[0]['hm'].dtype,
                device=preds_dicts[0]['hm'].device,
            )

        for task_id, preds_dict in enumerate(preds_dicts):
            # convert N C H W to N H W C
            for key, val in preds_dict.items():
                preds_dict[key] = val.permute(0, 2, 3, 1).contiguous()

            batch_size = preds_dict['hm'].shape[0]

            batch_hm = torch.sigmoid(preds_dict['hm'])

            batch_dim = torch.exp(preds_dict['dim'])

            batch_rots = preds_dict['rot'][..., 0:1]
            batch_rotc = preds_dict['rot'][..., 1:2]
            batch_reg = preds_dict['reg']
            batch_hei = preds_dict['height']
            if 'iou' in preds_dict.keys():
                batch_iou = (preds_dict['iou'].squeeze(dim=-1) + 1) * 0.5
                batch_iou = batch_iou.type_as(batch_dim)
            else:
                batch_iou = torch.ones((batch_hm.shape[0], batch_hm.shape[1], batch_hm.shape[2]),
                                       dtype=batch_dim.dtype).to(batch_hm.device)

            batch_rot = torch.atan2(batch_rots, batch_rotc)

            batch, H, W, num_cls = batch_hm.size()

            batch_reg = batch_reg.reshape(batch, H*W, 2)
            batch_hei = batch_hei.reshape(batch, H*W, 1)

            batch_rot = batch_rot.reshape(batch, H*W, 1)
            batch_dim = batch_dim.reshape(batch, H*W, 3)
            batch_hm = batch_hm.reshape(batch, H*W, num_cls)

            ys, xs = torch.meshgrid([torch.arange(0, H), torch.arange(0, W)])
            ys = ys.view(1, H, W).repeat(
                batch, 1, 1).to(batch_hm.device).float()
            xs = xs.view(1, H, W).repeat(
                batch, 1, 1).to(batch_hm.device).float()

            xs = xs.view(batch, -1, 1) + batch_reg[:, :, 0:1]
            ys = ys.view(batch, -1, 1) + batch_reg[:, :, 1:2]

            xs = xs * self.test_cfg.out_size_factor * \
                self.test_cfg.voxel_size[0] + self.test_cfg.pc_range[0]
            ys = ys * self.test_cfg.out_size_factor * \
                self.test_cfg.voxel_size[1] + self.test_cfg.pc_range[1]
            
            if 'vel' in preds_dict:
                batch_vel = preds_dict['vel']
                batch_vel = batch_vel.reshape(batch, H*W, 2)
                batch_box_preds = torch.cat(
                    [xs, ys, batch_hei, batch_dim, batch_vel, batch_rot], dim=2)
            else:
                batch_box_preds = torch.cat(
                    [xs, ys, batch_hei, batch_dim, batch_rot], dim=2)
            rets.append(self.post_processing(task_id, batch_box_preds,
                        batch_hm, self.test_cfg, post_center_range, batch_iou))

        # Merge branches results
        ret_list = []
        num_samples = len(rets[0])

        ret_list = []
        for i in range(num_samples):
            temp_instances = InstanceData()
            for k in rets[0][i].keys():
                if k == 'bboxes':
                    bboxes = torch.cat([ret[i][k] for ret in rets])
                    bboxes[:, 2] = bboxes[:, 2] - bboxes[:, 5] * 0.5
                    bboxes = batch_input_metas[i]['box_type_3d'](
                        bboxes, self.bbox_code_size)
                elif k == 'labels':
                    flag = 0
                    for j, num_class in enumerate(self.num_classes):
                        rets[j][i][k] += flag
                        flag += num_class
                    labels = torch.cat([ret[i][k] for ret in rets])
                elif k == 'scores':
                    scores = torch.cat([ret[i][k] for ret in rets])

            temp_instances.bboxes_3d = bboxes
            temp_instances.scores_3d = scores
            temp_instances.labels_3d = labels
            ret_list.append(temp_instances)

        return ret_list

    @torch.no_grad()
    def post_processing(self, task_id, batch_box_preds, batch_hm, test_cfg, post_center_range, batch_iou):
        batch_size = len(batch_hm)

        prediction_dicts = []
        for i in range(batch_size):
            box_preds = batch_box_preds[i]
            hm_preds = batch_hm[i]
            iou_preds = batch_iou[i].view(-1)
            scores, labels = torch.max(hm_preds, dim=-1)
            score_mask = scores > test_cfg.score_threshold
            distance_mask = (box_preds[..., :3] >= post_center_range[:3]).all(1) \
                & (box_preds[..., :3] <= post_center_range[3:]).all(1)

            mask = distance_mask & score_mask

            box_preds = box_preds[mask]
            scores = scores[mask]
            labels = labels[mask]
            iou_preds = torch.clamp(iou_preds[mask], min=0., max=1.)
            rectifier = torch.tensor(self.rectifier[task_id]).to(hm_preds)
            scores = torch.pow(
                scores, 1-rectifier[labels]) * torch.pow(iou_preds, rectifier[labels])
            # selected_boxes = torch.zeros((0, 9)).to(box_preds)
            selected_boxes = torch.zeros((0, 7)).to(box_preds)  # by jly
            selected_labels = torch.zeros((0,), dtype=torch.int64).to(labels)
            selected_scores = torch.zeros((0,)).to(scores)
            for class_id in range(hm_preds.shape[-1]):
                scores_class = scores[labels == class_id]
                labels_class = labels[labels == class_id]
                box_preds_class = box_preds[labels == class_id]
                boxes_for_nms_class = box_preds_class[:, [
                    0, 1, 2, 3, 4, 5, -1]]
                selected = rotate_nms_pcdet(boxes_for_nms_class, scores_class,
                                            thresh=test_cfg.nms.nms_iou_threshold,
                                            pre_maxsize=test_cfg.nms.nms_pre_max_size,
                                            post_max_size=test_cfg.nms.nms_post_max_size)

                selected_boxes = torch.cat(
                    (selected_boxes, box_preds_class[selected]), dim=0)
                selected_scores = torch.cat(
                    (selected_scores, scores_class[selected]), dim=0)
                selected_labels = torch.cat(
                    (selected_labels, labels_class[selected]), dim=0)

            prediction_dict = {
                'bboxes': selected_boxes,
                'scores': selected_scores,
                'labels': selected_labels
            }

            prediction_dicts.append(prediction_dict)

        return prediction_dicts
    
    # generate the gts to supervise the network
    def get_targets(self, batch_gt_instances_3d):
        """Generate targets. How each output is transformed: Each nested list
        is transposed so that all same-index elements in each sub-list (1, ...,
        N) become the new sub-lists.

                [ [a0, a1, a2, ... ], [b0, b1, b2, ... ], ... ]
                ==> [ [a0, b0, ... ], [a1, b1, ... ], [a2, b2, ... ] ]
            The new transposed nested list is converted into a list of N
            tensors generated by concatenating tensors in the new sub-lists.
                [ tensor0, tensor1, tensor2, ... ]
        Args:
            batch_gt_instances_3d (list[:obj:`InstanceData`]): Batch of
                gt_instances. It usually includes ``bboxes_3d`` and
                ``labels_3d`` attributes.
        Returns:
            Returns:
                tuple[list[torch.Tensor]]: Tuple of target including
                    the following results in order.
                - list[torch.Tensor]: Heatmap scores.
                - list[torch.Tensor]: Ground truth boxes.
                - list[torch.Tensor]: Indexes indicating the
                    position of the valid boxes.
                - list[torch.Tensor]: Masks indicating which
                    boxes are valid.
                - list[torch.Tensor]: catagrate labels.
        """
        heatmaps, anno_boxes, inds, masks, corner_heatmaps, cat_labels, gt_boxes = multi_apply(  # noqa: E501
            self.get_targets_single, batch_gt_instances_3d)  # derive the supervision signals for data in this batch one by one
        # Transpose heatmaps
        heatmaps = list(map(list, zip(*heatmaps)))
        heatmaps = [torch.stack(hms_) for hms_ in heatmaps]
        # Transpose heatmaps
        corner_heatmaps = list(map(list, zip(*corner_heatmaps)))
        corner_heatmaps = [torch.stack(hms_) for hms_ in corner_heatmaps]
        # Transpose anno_boxes
        anno_boxes = list(map(list, zip(*anno_boxes)))
        anno_boxes = [torch.stack(anno_boxes_) for anno_boxes_ in anno_boxes]
        # Transpose gt_boxes
        gt_boxes = list(map(list, zip(*gt_boxes)))
        gt_boxes = [torch.stack(gt_boxes_) for gt_boxes_ in gt_boxes]
        # Transpose inds
        inds = list(map(list, zip(*inds)))
        inds = [torch.stack(inds_) for inds_ in inds]
        # Transpose inds
        masks = list(map(list, zip(*masks)))
        masks = [torch.stack(masks_) for masks_ in masks]
        # Transpose cat_labels
        cat_labels = list(map(list, zip(*cat_labels)))
        cat_labels = [torch.stack(labels_) for labels_ in cat_labels]
        return heatmaps, anno_boxes, inds, masks, corner_heatmaps, cat_labels, gt_boxes
    
    # called by self.get_targets to generate supervision signals for one data sample
    def get_targets_single(self, gt_instances_3d):
        """Generate training targets for a single sample.
        Args:
            gt_instances_3d (:obj:`InstanceData`): Gt_instances of
                single data sample. It usually includes
                ``bboxes_3d`` and ``labels_3d`` attributes.
        Returns:
            tuple[list[torch.Tensor]]: Tuple of target including
                the following results in order.
                - list[torch.Tensor]: Heatmap scores.
                - list[torch.Tensor]: Ground truth boxes.
                - list[torch.Tensor]: Indexes indicating the position
                    of the valid boxes.
                - list[torch.Tensor]: Masks indicating which boxes
                    are valid.
                - list[torch.Tensor]: catagrate labels.
        """
        gt_labels_3d = gt_instances_3d.labels_3d
        gt_bboxes_3d = gt_instances_3d.bboxes_3d
        device = gt_labels_3d.device
        gt_bboxes_3d = torch.cat(
            (gt_bboxes_3d.gravity_center, gt_bboxes_3d.tensor[:, 3:]),
            dim=1).to(device)
        max_objs = self.train_cfg['max_objs'] * self.train_cfg['dense_reg']
        grid_size = torch.tensor(self.train_cfg['grid_size'])
        pc_range = torch.tensor(self.train_cfg['point_cloud_range'])
        voxel_size = torch.tensor(self.train_cfg['voxel_size'])
        gt_annotation_num = len(self.code_weights)

        feature_map_size = (grid_size[:2] // self.train_cfg['out_size_factor']).int()

        # reorganize the gt_dict by tasks
        task_masks = []
        flag = 0
        for class_name in self.class_names:
            task_masks.append([
                torch.where(gt_labels_3d == class_name.index(i) + flag)
                for i in class_name
            ])
            flag += len(class_name)

        task_boxes = []
        task_classes = []
        flag2 = 0
        for idx, mask in enumerate(task_masks):
            task_box = []
            task_class = []
            for m in mask:
                task_box.append(gt_bboxes_3d[m])
                # 0 is background for each task, so we need to add 1 here.
                task_class.append(gt_labels_3d[m] + 1 - flag2)
            task_boxes.append(torch.cat(task_box, axis=0).to(device))
            task_classes.append(torch.cat(task_class).long().to(device))
            flag2 += len(mask)
        draw_gaussian = draw_heatmap_gaussian  # method to generate the heatmaps
        heatmaps, anno_boxes, inds, masks, corner_heatmaps, cat_labels, gt_boxes = [], [], [], [], [], [], []  # noqa: E501

        for idx in range(len(self.tasks)):  # for one class, generate one center heatmap, five corner heatmaps(four corners + one center in the BEV, idea from AFDetV2)
            heatmap = gt_bboxes_3d.new_zeros(
                (len(self.class_names[idx]), feature_map_size[1],
                 feature_map_size[0]))
            corner_heatmap = torch.zeros(
                (1, feature_map_size[1], feature_map_size[0]),
                dtype=torch.float32,
                device=device)

            anno_box = gt_bboxes_3d.new_zeros((max_objs, gt_annotation_num),
                                              dtype=torch.float32)
            
            gt_box = gt_bboxes_3d.new_zeros((max_objs, 7),
                                              dtype=torch.float32)

            ind = gt_labels_3d.new_zeros((max_objs), dtype=torch.int64)
            mask = gt_bboxes_3d.new_zeros((max_objs), dtype=torch.uint8)
            cat_label = gt_bboxes_3d.new_zeros((max_objs), dtype=torch.int64)

            num_objs = min(task_boxes[idx].shape[0], max_objs)

            for k in range(num_objs):
                cls_id = task_classes[idx][k] - 1

                # gt boxes [xyzlwhr]
                length = task_boxes[idx][k][3]
                width = task_boxes[idx][k][4]
                length = length / voxel_size[0] / self.train_cfg[
                    'out_size_factor']
                width = width / voxel_size[1] / self.train_cfg[
                    'out_size_factor']

                if width > 0 and length > 0:
                    radius = gaussian_radius(
                        (width, length),
                        min_overlap=self.train_cfg['gaussian_overlap'])  # gaussian radius for heatmap generation
                    radius = max(self.train_cfg['min_radius'], int(radius))

                    # be really careful for the coordinate system of
                    # your box annotation.
                    x, y, z = task_boxes[idx][k][0], task_boxes[idx][k][
                        1], task_boxes[idx][k][2]

                    coor_x = (
                        x - pc_range[0]
                    ) / voxel_size[0] / self.train_cfg['out_size_factor']
                    coor_y = (
                        y - pc_range[1]
                    ) / voxel_size[1] / self.train_cfg['out_size_factor']

                    center = torch.tensor([coor_x, coor_y],
                                          dtype=torch.float32,
                                          device=device)
                    center_int = center.to(torch.int32)

                    # throw out not in range objects to avoid out of array
                    # area when creating the heatmap
                    if not (0 <= center_int[0] < feature_map_size[0]
                            and 0 <= center_int[1] < feature_map_size[1]):
                        continue

                    draw_gaussian(heatmap[cls_id], center_int, radius)  # center heatmap generation

                    radius = radius // 2
                    # # draw four corner and center TODO: use torch
                    rot = task_boxes[idx][k][6]
                    corner_keypoints = center_to_corner_box2d(
                        center.unsqueeze(0).cpu().numpy(),
                        torch.tensor([[length, width]],
                                     dtype=torch.float32).numpy(),
                        angles=rot,
                        origin=0.5)
                    corner_keypoints = torch.from_numpy(corner_keypoints).to(
                        center)

                    draw_gaussian(corner_heatmap[0], center_int, radius)  # corner heatmap 1: BEV center heatmap
                    draw_gaussian(                                        # corner heatmap 2: BEV corner heatmap
                        corner_heatmap[0],
                        (corner_keypoints[0, 0] + corner_keypoints[0, 1]) / 2,
                        radius)
                    draw_gaussian(                                        # corner heatmap 3: BEV corner heatmap
                        corner_heatmap[0],
                        (corner_keypoints[0, 2] + corner_keypoints[0, 3]) / 2,
                        radius)
                    draw_gaussian(                                        # corner heatmap 4: BEV corner heatmap
                        corner_heatmap[0],
                        (corner_keypoints[0, 0] + corner_keypoints[0, 3]) / 2,
                        radius)
                    draw_gaussian(                                        # corner heatmap 5: BEV corner heatmap
                        corner_heatmap[0],
                        (corner_keypoints[0, 1] + corner_keypoints[0, 2]) / 2,
                        radius)

                    new_idx = k
                    x, y = center_int[0], center_int[1]          # the 2D index indicating where the object center on the current feature map

                    assert (y * feature_map_size[0] + x <
                            feature_map_size[0] * feature_map_size[1])

                    ind[new_idx] = y * feature_map_size[0] + x   # the 1D index indicating where the object center on the current feature map
                    mask[new_idx] = 1
                    cat_label[new_idx] = cls_id
                    # TODO: support other outdoor dataset
                    # vx, vy = task_boxes[idx][k][7:]
                    rot = task_boxes[idx][k][6]
                    box_dim = task_boxes[idx][k][3:6]
                    box_dim = box_dim.log()
                    anno_box[new_idx] = torch.cat([
                        center - torch.tensor([x, y], device=device),
                        z.unsqueeze(0), box_dim,
                        torch.sin(rot).unsqueeze(0),
                        torch.cos(rot).unsqueeze(0)
                    ])
                    gt_box[new_idx] = task_boxes[idx][k][0:7]

            heatmaps.append(heatmap)
            corner_heatmaps.append(corner_heatmap)
            anno_boxes.append(anno_box)
            gt_boxes.append(gt_box)
            masks.append(mask)
            inds.append(ind)
            cat_labels.append(cat_label)
        return heatmaps, anno_boxes, inds, masks, corner_heatmaps, cat_labels, gt_boxes