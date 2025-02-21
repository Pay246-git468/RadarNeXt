import torch
# After substitution, this import below should be replaced by:
# from mmcv.utils import ext_loader
# ext_module = ext_loader.load_ext('_ext', ['iou3d_nms3d_forward'])
# from projects.PillarNeXt.pillarnext.utils.iou3d_nms import iou3d_nms_cuda

# by jly
from mmcv.utils import ext_loader
ext_module = ext_loader.load_ext('_ext', ['iou3d_nms3d_forward'])


def rotate_nms_pcdet(boxes, scores, thresh, pre_maxsize=None, post_max_size=None):
    """
    :param boxes: (N, 7) [x, y, z, size_x, size_y, size_z, theta]
    :param scores: (N)
    :param thresh:
    :return:
    """
    # transform back to pcdet's coordinate
    order = scores.sort(0, descending=True)[1]
    if pre_maxsize is not None:
        order = order[:pre_maxsize]

    boxes = boxes[order].contiguous()

    # keep = torch.LongTensor(boxes.size(0))

    # by jly
    keep = boxes.new_zeros(boxes.size(0), dtype=torch.long)

    if len(boxes) == 0:
        num_out = 0
    else:
        # This function can be replaced by three rows of codes below:
        # keep = boxes.new_zeros(boxes.size(0), dtype=torch.long)
        # num_out = boxes.new_zeros(size=(), dtype=torch.long)
        # ext_module.iou3d_nms3d_forward(
        #       boxes, keep, num_out, nms_overlap_thresh=iou_threshold)
        # num_out = iou3d_nms_cuda.nms_gpu(boxes, keep, thresh)

        # by jly
        num_out = boxes.new_zeros(size=(), dtype=torch.long)
        ext_module.iou3d_nms3d_forward(
            boxes, keep, num_out, nms_overlap_thresh=thresh)

    selected = order[keep[:num_out].cuda()].contiguous()

    if post_max_size is not None:
        selected = selected[:post_max_size]

    return selected