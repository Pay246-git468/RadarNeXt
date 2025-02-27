# Installation
## Base Environments:
Our RadarNeXt runs on a single Nvidia RTX A4000 GPU with the following environments:\
- cuda=11.3
- torch=1.12.0
- torchvision=0.13.0
- mmengine=l0.10.5
- mmcv=2.1.0
- mmdet=3.3.0
- mmdet3d=1.4.0
### Official Installation:
We suggest following the guidance given by [official MMDetection3D toolbox](https://github.com/open-mmlab/mmdetection3d).
### Custom Installation:
You can also follow these steps to copy our base environments:\
1. Create your virtual environment:\
   `conda create --name {YOUR_ENV_NAME} python=3.8 -y`\
   Then activate the virtual environment:\
   `conda activate {YOUR_ENV_NAME}`
2. Install cudatoolkit, pytorch, and torchvision:\
   We suggest following the [official instructions](https://pytorch.org/get-started/locally/) to download these packages.\
   Or you can use the command below:\
   `conda install pytorch==1.12.0 torchvision==0.13.0 cudatoolkit=11.3 -c pytorch`
3. Install the packages of MMDetection3D:\
   `pip install -U openmim`\
   `mim install 'mmengine==0.10.5'`\
   `mim install 'mmcv==2.1.0'`\
   `mim install 'mmdet==3.3.0'`\
   `mim install 'mmdet3d==1.4.0'`
If you use the commands above to install these packages successfully, you can fully copy our base environments.

## Additional Requirements:
Our RadarNeXt also needs these packages:\
- torch-scatter
- DCNv3
- thop
### torch-scatter:
You can use 'pip' to install torch-scatter:\
`pip install torch-scatter`\
If the 'pip' command is failed, you can install this package by 'conda':\
`conda install pytorch-scatter -c pyg`
### DCNv3:
You can find the .whl file according to your base environment in the [OpenGVLab official website](https://github.com/OpenGVLab/InternImage/releases/tag/whl_files).\
And then use `pip install {FILENAME_YOUR_DOWNLOADED}` to install DCNv3.\
Or you can download the source code from [OpenGVLab InternImage](https://github.com/OpenGVLab/InternImage/tree/master).\
Then build the wheel and install the DCNv3 suitable to your own environment with:\
`cd {PATH_TO_SOURCE_CODE}/InternImage-master/detection/ops_dcnv3/`
`set DISTUTILS_USE_SDK=1`
`python setup.py build install`
### thop:
'thop' is the package to calculate the parameter counts and MACS of the models by '[test_model.py](tools/analysis_tools/test_model.py)'.\
You can install it by running `pip install thop` easily.\
Or look through the [official website](https://github.com/Nobreakfast/UniP) to find the installation guidance.
