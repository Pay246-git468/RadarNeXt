# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os
import os.path as osp
import time

from mmengine.config import Config, ConfigDict, DictAction
from mmengine.registry import RUNNERS
from mmengine.runner import Runner

import torch
from mmengine import DefaultScope

from mmdet3d.registry import MODELS
import mmdet3d.models

from mmdet3d.utils import replace_ceph_backend

from base.evaluation import cal_flops, process_data, to_device

from mmdet3d.testing import create_detector_inputs, get_detector_cfg, setup_seed


def get_example_data(model, num_gt_instance=2, points_feat_dim=7, img_size=(1936, 1216)):
    packed_inputs = create_detector_inputs(
        num_gt_instance=num_gt_instance,
        points_feat_dim=points_feat_dim,
        with_img=True,
        img_size=tuple(img_size),
    )
    data = model.data_preprocessor(packed_inputs, True)
    return data


# TODO: support fuse_conv_bn and format_only
def parse_args():
    parser = argparse.ArgumentParser(description="MMDet3D test (and eval) a model")
    parser.add_argument("config", help="test config file path")
    return parser.parse_args()


def main():
    args = parse_args()
    # load config
    # cfg = get_detector_cfg(args.config)
    cfg = Config.fromfile(args.config)

    DefaultScope.get_instance("prune", scope_name="mmdet3d")
    model = MODELS.build(cfg.model)
    data = get_example_data(
        model,
        num_gt_instance=3,
        points_feat_dim=7,
        img_size=cfg.image_size if hasattr(cfg, "image_size") else (256, 256),
    )

    flops, params, clever_print = cal_flops(model, data)
    # print(model)
    print(f"Original: {clever_print}")
    model = to_device(model)
    data = process_data(data)
    for _ in range(100):  # warmup
        model._forward(*data)
    start_time = time.time()
    for _ in range(1296):
        model._forward(*data)
    end_time = time.time()
    print(f'fps: {1/((end_time-start_time)/1296)}')


if __name__ == "__main__":
    main()
