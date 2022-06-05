# ResNet-D_PyTorch
PyTorch implementation of ResNet-D from "Bag of Tricks for Image Classification with Convolutional Neural Networks" (He et al., CVPR2019) 

paper: https://arxiv.org/pdf/1812.01187

summary (KOR): https://bo-10000.tistory.com/133

<img src="https://s3.us-west-2.amazonaws.com/secure.notion-static.com/35eae783-f8c7-4b2c-a145-e26b00185c38/Untitled.png?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Content-Sha256=UNSIGNED-PAYLOAD&X-Amz-Credential=AKIAT73L2G45EIPT3X45%2F20220605%2Fus-west-2%2Fs3%2Faws4_request&X-Amz-Date=20220605T125246Z&X-Amz-Expires=86400&X-Amz-Signature=a60ed99c2fc3a0b19922ed22837183c96294c514c4916c92185c92615c79b4ae&X-Amz-SignedHeaders=host&response-content-disposition=filename%20%3D%22Untitled.png%22&x-id=GetObject"></img>

## Models available
- `resnet18`
- `resnet34`
- `resnet50`
- `resnet101`
- `resnet152`
- `resnet50D`
- `resnet101D`
- `resnet152D`


## How to use
```
from resnet_d import *

model = resnet50D()
```


## Model parameters
- `num_classes (int, default=1000)`: number of class labels
- `zero_init_residual (bool, default=False)`: Zero-initialize the last BN in each residual branch, so that the residual branch starts with zeros, and each residual block behaves like an identity.
- `groups (int, default=1)`: number of groups for conv3x3
- `width_per_group (int, default=64)`: to change model width
- `replace_stride_with_dilation (list of bools, default=None)`: each element in the tuple indicates if we should replace the 2x2 stride with a dilated convolution instead
- `norm_layer (nn.Module, default=None)`: norm layer to use

