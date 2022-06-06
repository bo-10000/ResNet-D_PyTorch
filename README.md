# ResNet-D_PyTorch
PyTorch implementation of ResNet-D from "Bag of Tricks for Image Classification with Convolutional Neural Networks" (He et al., CVPR2019) 

paper: https://arxiv.org/pdf/1812.01187

summary (KOR): https://bo-10000.tistory.com/133

<img src="https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2FceLiVZ%2FbtrDZItgCfy%2FyFb3QyfZOAhSIDOm1VK0U1%2Fimg.png" height="400px"></img>

</br>

## Models available
**resnetD.py:**
- `resnet18`
- `resnet34`
- `resnet50`
- `resnet101`
- `resnet152`
- `resnet50D`
- `resnet101D`
- `resnet152D`

</br>

**resnetD_3d.py:**
- `resnet3d18`
- `resnet3d34`
- `resnet3d50`
- `resnet3d101`
- `resnet3d152`
- `resnet3d50D`
- `resnet3d101D`
- `resnet3d152D`

## How to use
```
#2D models
from resnetD import *

model = resnet50D()

#3D models
from resnetD_3d import *

model = resnet3d50D()
```

## Model parameters
- `in_channels (int, defauult=3)`: input channel dimension
- `num_classes (int, default=1000)`: number of class labels
- `zero_init_residual (bool, default=False)`: Zero-initialize the last BN in each residual branch, so that the residual branch starts with zeros, and each residual block behaves like an identity.
- `groups (int, default=1)`: number of groups for conv3x3
- `width_per_group (int, default=64)`: to change model width
- `replace_stride_with_dilation (list of bools, default=None)`: each element in the tuple indicates if we should replace the 2x2 stride with a dilated convolution instead
- `norm_layer (nn.Module, default=None)`: norm layer to use

