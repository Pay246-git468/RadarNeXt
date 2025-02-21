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
- 

