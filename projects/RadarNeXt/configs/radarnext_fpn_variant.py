custom_imports = dict(
    imports=['projects.RadarNeXt.radarnext'], allow_failed_imports=False)

num_epoch = 80
train_cfg = dict(by_epoch=True, max_epochs=num_epoch, val_interval=1, val_begin=1) # params: val_begin (int) -> The epoch that starts activating the validation
val_cfg = dict()
test_cfg = dict()

auto_scale_lr = dict(enable=False, base_batch_size=16)

default_scope = 'mmdet3d'

default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=50),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(type='CheckpointHook', interval=5),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='Det3DVisualizationHook'))

env_cfg = dict(
    cudnn_benchmark=False,
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    dist_cfg=dict(backend='nccl'),
)

voxel_size = [0.16, 0.16, 5]
point_cloud_range = [0, -25.6, -3, 51.2, 25.6, 2]
class_names = ['Pedestrian', 'Cyclist', 'Car']
tasks = [dict(num_class=3, class_names=['Pedestrian', 'Cyclist', 'Car'])]
metainfo = dict(classes=class_names)
input_modality = dict(use_lidar=True, use_camera=False)
backend_args = None

inference_mode = True
custom_hooks = [
    dict(type='Rep_Checkpoint_Hook', reparams=True, filename_tmpl=None)
]

model = dict(
    type='RadarNeXt',
    with_auxiliary=True,
    inference_mode=inference_mode,
    data_preprocessor=dict(
        type='Det3DDataPreprocessor',
        voxel=True,
        voxel_layer=dict(
            max_num_points=10,
            max_voxels=(16000, 40000),
            point_cloud_range=point_cloud_range,
            voxel_size=voxel_size)),
    voxel_encoder=dict(
        feat_channels=[64],
        in_channels=0,
        point_cloud_range=point_cloud_range,
        type='Radar7PillarFeatureNet',
        voxel_size=voxel_size,
        with_distance=False,
        use_xyz=True,
        use_rcs=True,
        use_vr=True,
        use_vr_comp=True,
        use_time=True,
        use_elevation=True,
        ),
    middle_encoder=dict(
        in_channels=64, output_shape=[320,320], type='PointPillarsScatter'),
    backbone=dict(
        type='RepDWC',
        in_channels=64,
        layer_nums=[3, 5, 5],
        layer_strides=[2, 2, 2],
        out_channels=[64, 128, 256], # 1/2 -> 1/4 -> 1/8
        num_outputs=3,
        inference_mode=False,
        use_se=False,
        num_conv_branches=1,
        use_normconv=False,
        use_dwconv=True,
    ),
    neck=dict(
        type='SECONDFPN',  # normal FPNNeck
        in_channels=[64, 128, 256],
        norm_cfg=dict(eps=0.001, momentum=0.01, type='BN'),
        out_channels=[128, 128, 128],
        upsample_cfg=dict(bias=False, type='deconv'),
        upsample_strides=[0.5, 1, 2],
        use_conv_for_no_stride=True),
    bbox_head=dict(
        type='RadarNeXt_Head',
        multi_fusion=False,
        fusion_channels=[128, 128, 128],
        fusion_strides=[1, 2],
        in_channels=sum([128, 128, 128]),          
        tasks=tasks,
        strides=[2, 2, 2],
        weight=1,
        corner_weight=1,
        iou_weight=1,
        iou_reg_weight=0.5,
        rectifier=[[0.5, 0.5, 0.5]],
        with_corner=True,
        with_reg_iou=True,
        voxel_size=voxel_size,
        pc_range=point_cloud_range,
        out_size_factor=2,
        code_weights=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        common_heads={
            'reg': (2, 2),
            'height': (1, 2),
            'dim': (3, 2),
            'rot': (2, 2),
            'iou': (1, 2)
        },
    ),
    train_cfg=dict(
        grid_size=[320, 320],
        voxel_size=voxel_size,
        out_size_factor=2,    
        dense_reg=1,
        gaussian_overlap=0.1,
        point_cloud_range=point_cloud_range,
        max_objs=500,
        min_radius=2,
        code_weights=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]),
    test_cfg=dict(
        post_center_limit_range=point_cloud_range,
        nms=dict(
            nms_pre_max_size=1000,
            nms_post_max_size=83,
            nms_iou_threshold=0.2,
        ),
        score_threshold=0.1,
        pc_range=[0, -25.6], 
        out_size_factor=2,
        voxel_size=[0.16, 0.16]
    ))

data_root = 'data/vod/radar_5frames/'

train_pipeline = [
    dict(
        backend_args=None,
        coord_type='LIDAR',
        load_dim=7,
        type='LoadPointsFromFile',
        use_dim=7),
    dict(type='LoadAnnotations3D', with_bbox_3d=True, with_label_3d=True),
    dict(type='GlobalScale', scale_ratio_range=[0.95, 1.05]),
    dict(flip_ratio_bev_horizontal=0.5, type='RandomFlip3D'),
    dict(
        point_cloud_range=point_cloud_range,
        type='PointsRangeFilter'),
    dict(
        point_cloud_range=point_cloud_range,
        type='ObjectRangeFilter'),
    dict(type='PointShuffle'),
    dict(
        keys=['points', 'gt_labels_3d','gt_bboxes_3d'],
        type='Pack3DDetInputs'),
]

test_pipeline = [
    dict(
        backend_args=None,
        coord_type='LIDAR',
        load_dim=7,
        type='LoadPointsFromFile',
        use_dim=7),
    dict(type='RandomFlip3D'),
    dict(
        type='PointsRangeFilter', point_cloud_range=point_cloud_range),
    dict(keys=['points'], type='Pack3DDetInputs'),
]

dataset_type = 'KittiDataset'

train_dataloader = dict(
    batch_size=8,
    dataset=dict(
        dataset=dict(
            ann_file='kitti_infos_train.pkl',
            backend_args=None,
            box_type_3d='LiDAR',
            data_prefix=dict(pts='training/velodyne'),
            data_root='data/vod/radar_5frames/',
            metainfo=metainfo,
            modality=input_modality,
            pipeline=train_pipeline,
            test_mode=False,
            type=dataset_type),
        times=1,
        type='RepeatDataset'),
    num_workers=4,
    persistent_workers=True,
    sampler=dict(shuffle=True, type='DefaultSampler'))

test_dataloader = dict(
    batch_size=1,
    dataset=dict(
        ann_file='kitti_infos_val.pkl',
        backend_args=None,
        box_type_3d='LiDAR',
        data_prefix=dict(pts='training/velodyne'),
        data_root=data_root,
        metainfo=metainfo,
        modality=input_modality,
        pipeline=test_pipeline,
        test_mode=True,
        type=dataset_type),
    drop_last=False,
    num_workers=1,
    persistent_workers=True,
    sampler=dict(shuffle=False, type='DefaultSampler'))
val_dataloader = dict(
    batch_size=1,
    num_workers=1,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(pts='training/velodyne'),
        ann_file='kitti_infos_val.pkl',
        pipeline=test_pipeline,
        modality=input_modality,
        test_mode=True,
        metainfo=metainfo,
        box_type_3d='LiDAR'))

test_evaluator = dict(
    type='KittiMetricv2',
    dataset_name='VoD',
    ann_file='data/vod/radar_5frames/kitti_infos_val.pkl',
    metric='bbox',
    format_only=True,
    submission_prefix='results/kitti-3class/kitti_results') # for submission
val_evaluator = dict(
    ann_file='data/vod/radar_5frames/kitti_infos_val.pkl',
    backend_args=None,
    metric='bbox',
    dataset_name='VoD',
    type='KittiMetricv2')

vis_backends = [dict(type='LocalVisBackend')]
visualizer = dict(
    type='Det3DLocalVisualizer', vis_backends=vis_backends, name='visualizer')

lr = 0.003
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=lr, weight_decay=0.01),
    clip_grad=dict(max_norm=10, norm_type=2))
step_ratio_up = 0.4
param_scheduler = [
    dict(
        type='MultiStepLR',  
        milestones=[35, 45],  
        gamma=0.1,         
        by_epoch=True, 
    ),
    dict(
        eta_min_ratio=0.85 / 0.95,       
        begin=0,
        end=num_epoch * step_ratio_up,
        T_max=num_epoch * step_ratio_up,
        by_epoch=True,
        convert_to_iter_based=True,
        type='CosineAnnealingMomentum'),
    dict(
        eta_min_ratio=1,
        begin=num_epoch * step_ratio_up,
        end=num_epoch,
        T_max=num_epoch - num_epoch * step_ratio_up, 
        by_epoch=True,
        convert_to_iter_based=True,
        type='CosineAnnealingMomentum'),
]

log_processor = dict(type='LogProcessor', window_size=50, by_epoch=True)

log_level = 'INFO'
load_from = None
resume = False

visualizer = dict(
    name='visualizer',
    type='Det3DLocalVisualizer',
    vis_backends=[
        dict(type='LocalVisBackend'),
        dict(type='TensorboardVisBackend'),
    ])