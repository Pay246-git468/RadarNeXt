# PillarNet
Here is the reproduced PillarNeXt in MMDetection3D version based on its [Det3D repository](https://github.com/VISION-SJTU/PillarNet).

## Training and Testing
For training on [View-of-Delft](https://github.com/tudelft-iv/view-of-delft-dataset):\
   For using sparse resnet-18: `python tools/train.py projects/PillarNet/configs/pillarnet_radar_vod.py`;\
   For using sparse resnet-34: `python tools/train.py projects/PillarNet/configs/pillarnet34_radar_vod.py`\
For testing on View-of-Delft val set:\
   For using sparse resnet-18: `python tools/test.py projects/PillarNet/configs/pillarnet_radar_vod.py {PATH_TO_WEIGHTS} --samples 1296`\
   For using sparse resnet-34: `python tools/test.py projects/PillarNet/configs/pillarnet34_radar_vod.py {PATH_TO_WEIGHTS} --samples 1296`\

## Performances
We trained the PillarNeXt on a single Nvidia RTX A4000 GPU
| Methods | Params | VoD mAP | TJ4D mAP | FPS A4000 | FPS Orin |
|:--------:|:--------:|:--------:|:--------:|:--------:|:--------:|
| PillarNet-18 | 10.052M | 35.82 | / | / | 21.20 |
| PillarNet-34 | 7.563M | 45.15 | / | / | 17.90 |
