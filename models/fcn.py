'''
fcn.py 的任务
1. 实现双线插值 
2. 实现FCN本人  
(人 •͈ᴗ•͈)  ۶♡♡
'''

import numpy as np
import torch
from torch import nn
from torchvision import models

#双插
def Bilinear_interpolation (src, new_size):
    '''
    使用双线性插值方法放大图像
    params：
        src(np.ndarray)：输入图片
        new_size(tuple)：目标尺寸
    ret：
        dst(np.ndarry):目标图像
    '''
    dst_h, dst_w = new_size  #目标图像的hw
    src_h, src_w = src.shape[:2] #原始图像的hw

    #如果跟需求符合， 就不需要缩放，直接拷贝
    if src_h == dst_h and src_w == dst_w:
        return src.copy()

    scale_x = float(src_w) / dst_w
    scale_y = float(src_H) / dst_h

    #遍历目标图上的每个像素点
    ##构建一张目标大小的空图，遍历差值
    dst = np.zeros((dst_h,dst_w,3),dtype=np.int8)
    ##因为是彩色图，遍历三层： a.rgb三通道 b.height c.width
    for n in range(3):
        for dst_y in range(dst_h):
            for dst_x in range(dst_w):
                
                #目标像素在原图上的坐标 src+0.5 = (dst_x + 0.5) *scale_x
                #加0.5的偏差，可以保证图像缩小时，不会漏掉像素点 详细看：https://www.cnblogs.com/kk17/p/9989984.html
                src_x = (dst_x + 0.5)*scale_x -0.5
                src_y = (dst_y + 0.5)*scale_y -0.5

                #计算在原图某像素点的4个近邻点的位置
                src_x_0 = int(np.floor(src_x)) #*floor()向下取整数 ex: floor(1.2) = 1.0
                src_y_0 = int(np.floor(src_y))
                src_x_1 = min(src_x_0 + 1, src_w - 1  ) #防止出界
                src_y_1 = min(src_y_0 + 1, src_h - 1  )


'''
初始化反卷积核
'''
def bilinear_kernel(in_channels, out_channels, kernel_size):
    """Define a bilinear kernel according to in channels and out channels.
    Returns:
        return a bilinear filter tensor
    """
    factor = (kernel_size + 1) // 2
    if kernel_size % 2 == 1:
        center = factor - 1
    else:
        center = factor - 0.5
    og = np.ogrid[:kernel_size, :kernel_size]
    bilinear_filter = (1 - abs(og[0] - center) / factor) * (1 - abs(og[1] - center) / factor)
    weight = np.zeros((in_channels, out_channels, kernel_size, kernel_size), dtype=np.float32)
    weight[range(in_channels), range(out_channels), :, :] = bilinear_filter
    return torch.from_numpy(weight)

pretrained_net = models.vgg16_bn(pretrained=False)

#FCN本人
class FCN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        self.stage1 = pretrained_net.features[:7]
        self.stage2 = pretrained_net.features[7:14]
        self.stage3 = pretrained_net.features[14:24]
        self.stage4 = pretrained_net.features[24:34]
        self.stage5 = pretrained_net.features[34:]

        self.scores1 = nn.Conv2d(512, num_classes, 1)
        self.scores2 = nn.Conv2d(512, num_classes, 1)
        self.scores3 = nn.Conv2d(128, num_classes, 1)

        self.conv_trans1 = nn.Conv2d(512, 256, 1)
        self.conv_trans2 = nn.Conv2d(256, num_classes, 1)

        self.upsample_8x = nn.ConvTranspose2d(num_classes, num_classes, 16, 8, 4, bias=False)
        self.upsample_8x.weight.data = bilinear_kernel(num_classes, num_classes, 16)

        self.upsample_2x_1 = nn.ConvTranspose2d(512, 512, 4, 2, 1, bias=False)
        self.upsample_2x_1.weight.data = bilinear_kernel(512, 512, 4)

        self.upsample_2x_2 = nn.ConvTranspose2d(256, 256, 4, 2, 1, bias=False)
        self.upsample_2x_2.weight.data = bilinear_kernel(256, 256, 4)

    def forward(self, x):
        s1 = self.stage1(x)
        s2 = self.stage2(s1)
        s3 = self.stage3(s2)
        s4 = self.stage4(s3)
        s5 = self.stage5(s4)

        scores1 = self.scores1(s5)
        s5 = self.upsample_2x_1(s5)
        add1 = s5 + s4

        scores2 = self.scores2(add1)

        add1 = self.conv_trans1(add1)
        add1 = self.upsample_2x_2(add1)
        add2 = add1 + s3

        output = self.conv_trans2(add2)
        output = self.upsample_8x(output)
        return output



if __name__ == "__main__":
    rgb = torch.randn(1, 3, 352, 480)
    net = FCN(12)
    out = net(rgb)
    print('喵喵喵喵喵喵喵喵---------------')
    print(out.shape)