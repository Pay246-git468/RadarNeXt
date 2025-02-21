# This is an extension library including two packages, iou3d_nms_cuda and iou3d_nms_utils, to compute the NMS and iou for PillarNeXt
# This extension is built by the setup.py in this directory, which is the source code for establishing the python library
# Two extension packages are named as iou3d_nms_cuda and iou3d_nms_utils
# The iou3d_nms_cuda package provides multiple NMS algorithms implementing by c and c++ source codes in the src sub-directory
# and all the foundational algorthms in formats of c or c++ codes are transferred into CUDA source codes by CUDAExtension in the setup.py
# The iou3d_nms_utils package is created by the iou3d_nms_utils.py, locating at the same directory with setup.py so that being another package
# The iou3d_nms_utils summarized the NMS and IoU algorithms, whose implementations are based on the iou3d_nms_cuda package, for the PillarNeXt calling.
from projects.PillarNeXt.pillarnext.utils.iou3d_nms import iou3d_nms_cuda  # noqa F401