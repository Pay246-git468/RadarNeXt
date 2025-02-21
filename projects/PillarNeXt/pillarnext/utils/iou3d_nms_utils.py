import torch

# by jly
from mmcv.utils import ext_loader
from mmcv.ops.iou3d import boxes_overlap_bev
ext_module = ext_loader.load_ext('_ext', ['iou3d_nms3d_forward'])
# from projects.PillarNeXt.pillarnext.utils.iou3d_nms import iou3d_nms_cuda

# This function can be completely substituted by mmcv.ops.iou3d.boxes_iou3d()
def boxes_iou3d_gpu(boxes_a, boxes_b):
    """
    Args:
        boxes_a: (N, 7) [x, y, z, dx, dy, dz, heading]
        boxes_b: (N, 7) [x, y, z, dx, dy, dz, heading]

    Returns:
        ans_iou: (N, M)
    """
    assert boxes_a.shape[1] == boxes_b.shape[1] == 7

    # height overlap
    boxes_a_height_max = (boxes_a[:, 2] + boxes_a[:, 5] / 2).view(-1, 1)
    boxes_a_height_min = (boxes_a[:, 2] - boxes_a[:, 5] / 2).view(-1, 1)
    boxes_b_height_max = (boxes_b[:, 2] + boxes_b[:, 5] / 2).view(1, -1)
    boxes_b_height_min = (boxes_b[:, 2] - boxes_b[:, 5] / 2).view(1, -1)

    # bev overlap
    # Two rows of codes below can be replaced by mmcv.ops.iou3d.boxes_overlap_bev()
    # overlaps_bev = torch.cuda.FloatTensor(torch.Size(
    #     (boxes_a.shape[0], boxes_b.shape[0]))).zero_()  # (N, M)
    # iou3d_nms_cuda.boxes_overlap_bev_gpu(
    #     boxes_a.contiguous(), boxes_b.contiguous(), overlaps_bev)

    # by jly
    overlaps_bev = boxes_overlap_bev(boxes_a, boxes_b)

    max_of_min = torch.max(boxes_a_height_min, boxes_b_height_min)
    min_of_max = torch.min(boxes_a_height_max, boxes_b_height_max)
    overlaps_h = torch.clamp(min_of_max - max_of_min, min=0)

    # 3d iou
    overlaps_3d = overlaps_bev * overlaps_h

    vol_a = (boxes_a[:, 3] * boxes_a[:, 4] * boxes_a[:, 5]).view(-1, 1)
    vol_b = (boxes_b[:, 3] * boxes_b[:, 4] * boxes_b[:, 5]).view(1, -1)

    iou3d = overlaps_3d / torch.clamp(vol_a + vol_b - overlaps_3d, min=1e-6)

    return iou3d


def boxes_aligned_iou3d_gpu(boxes_a, boxes_b):
    """
    Args:
        boxes_a: (N, 7) [x, y, z, dx, dy, dz, heading]
        boxes_b: (N, 7) [x, y, z, dx, dy, dz, heading]

    Returns:
        ans_iou: (N,)
    """
    assert boxes_a.shape[0] == boxes_b.shape[0]
    assert boxes_a.shape[1] == boxes_b.shape[1] == 7

    # transform back to pcdet's coordinate
    # boxes_a = to_pcdet(boxes_a)
    # boxes_b = to_pcdet(boxes_b)

    # height overlap
    boxes_a_height_max = (boxes_a[:, 2] + boxes_a[:, 5] / 2).view(-1, 1)
    boxes_a_height_min = (boxes_a[:, 2] - boxes_a[:, 5] / 2).view(-1, 1)
    boxes_b_height_max = (boxes_b[:, 2] + boxes_b[:, 5] / 2).view(-1, 1)
    boxes_b_height_min = (boxes_b[:, 2] - boxes_b[:, 5] / 2).view(-1, 1)

    # bev overlap
    # Two rows of codes below can be replaced by mmcv.ops.iou3d.boxes_overlap_bev()
    # However, for keeping the shape of the output of this boxes_aligned_iou3d_gpu() function
    # we need to manually select the diagonal of overlapping matrices provided by mmcv.ops.iou3d.boxes_overlap_bev()
    # For example, assuming overlap_matrix below is the output of boxes_overlap_bev()
    # overlap_matrix = torch.rand((N, M))
    # choose the overlapping elements on the diagonal of output
    # overlap_vector = torch.diag(overlap_matrix)
    # This function's inputs, two bboxes, must be in the same numbers of bboxes (shape[0])
    # overlaps_bev = torch.cuda.FloatTensor(
    #     torch.Size((boxes_a.shape[0], 1))).zero_()  # (N, M)
    # iou3d_nms_cuda.boxes_aligned_overlap_bev_gpu(
    #     boxes_a.contiguous(), boxes_b.contiguous(), overlaps_bev)

    # by jly
    overlaps_bev = boxes_overlap_bev(boxes_a, boxes_b)  # (N, N)
    overlaps_bev = torch.diag(overlaps_bev).view(-1, 1)   # (N, 1), by jly

    max_of_min = torch.max(boxes_a_height_min, boxes_b_height_min)
    min_of_max = torch.min(boxes_a_height_max, boxes_b_height_max)
    overlaps_h = torch.clamp(min_of_max - max_of_min, min=0)

    # 3d iou
    overlaps_3d = overlaps_bev * overlaps_h

    vol_a = (boxes_a[:, 3] * boxes_a[:, 4] * boxes_a[:, 5]).view(-1, 1)
    vol_b = (boxes_b[:, 3] * boxes_b[:, 4] * boxes_b[:, 5]).view(-1, 1)

    iou3d = overlaps_3d / torch.clamp(vol_a + vol_b - overlaps_3d, min=1e-6)

    return iou3d


def nms_gpu(boxes, scores, thresh, pre_maxsize=None, **kwargs):
    """
    :param boxes: (N, 7) [x, y, z, dx, dy, dz, heading]
    :param scores: (N)
    :param thresh:
    :return:
    """
    assert boxes.shape[1] == 7
    order = scores.sort(0, descending=True)[1]
    if pre_maxsize is not None:
        order = order[:pre_maxsize]

    boxes = boxes[order].contiguous()
    # keep = torch.LongTensor(boxes.size(0))
    # num_out = iou3d_nms_cuda.nms_gpu(boxes, keep, thresh)

    # by jly
    keep = boxes.new_zeros(boxes.size(0), dtype=torch.long)
    num_out = boxes.new_zeros(size=(), dtype=torch.long)
    ext_module.iou3d_nms3d_forward(
        boxes, keep, num_out, nms_overlap_thresh=thresh)

    return order[keep[:num_out].cuda()].contiguous(), None