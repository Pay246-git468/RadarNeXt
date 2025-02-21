# PillarNeXt
Here is the reproduced PillarNeXt in MMDetection3D version based on its [Det3D repository](https://github.com/qcraftai/pillarnext).\
In our RadarNeXt, we design our head based on the CenterHead used by PillarNeXt.\

## Training and Testing
For training on View-of-Delft:\
`python tools/train.py projects/PillarNeXt/configs/pillarnext_radar_vod.py`\
For training on TJ4DRadSet:\
`python tools/train.py projects/PillarNeXt/configs/pillarnext_radar_tj4d.py`\
For testing on View-of-Delft val set:\
`python tools/test.py projects/PillarNeXt/configs/pillarnext_radar_vod.py {PATH_TO_WEIGHTS} --samples 1296`\
For testing on TJ4DRadSet test set:\
`python tools/test.py projects/PillarNeXt/configs/pillarnext_radar_tj4d.py {PATH_TO_WEIGHTS} --samples 2040`
