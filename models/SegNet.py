import torch
import torchvision.models as models
from torch import nn









'''
- Encoder-Decoder
E： 普通卷积&下采样 -- 提取抽象特征(低级&高级)
D： 普通卷积&上采样&融合 --  恢复空间维度，端到端
- Pixel-wise的分类层

decoder进行上采样的方式。直接利用与之对应的encoder阶段中，
在经过最大池化时，保留pooling index进行非线性上采样

比较SegNet和FCN等，统筹内存与准确率，SegNet实现了良好的分割效果
模型评估，在Camvid和SUN RGB-D indoor数据集中均有评测
'''