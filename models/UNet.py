'''
对应U-Net.png看  ×××一定要每一步都把特征图打印出来！！不然报错到怀疑人生！！

收缩路径block
    卷积卷积relu --bridge左侧的蓝色小箭头 
    下采样部分 -- bridge左侧的红色小箭头
扩张路径block
    反卷积 --bridge右侧绿色小箭头
    卷积块+relu+batchnorm （跟收缩差不多的组合）--bridge右侧蓝色小箭头
    过度通道（for 拼接） --bridge右侧白蓝相间拼接后的特征图，往下卷的那个小箭头
    前向传播  
        特征图裁剪&拼接 --bridge右侧的灰色小箭头 & 小白块
        torch.cat https://blog.csdn.net/TH_NUM/article/details/83088915
final block 分类1x1卷积block

大类class UNnet

'''
import torch
from torch import nn


def contracting_block(in_channels, out_channels):
    #* torch.nn.Sequential 快速把函数封成模块
    block = torch.nn.Sequential(
                nn.Conv2d(kernel_size=(3,3), in_channels=in_channels, out_channels=out_channels),
                nn.ReLU(),
                nn.BatchNorm2d(out_channels),
                nn.Conv2d(kernel_size=(3,3), in_channels=out_channels, out_channels=out_channels),
                nn.ReLU(),
                nn.BatchNorm2d(out_channels)
            )
    return block


class expansive_block(nn.Module):
    def __init__(self, in_channels, mid_channels, out_channels):
        super(expansive_block, self).__init__()
        
        self.up = nn.ConvTranspose2d(in_channels, in_channels//2, kernel_size=(3, 3), stride=2, padding=1, 
                                     output_padding=1, dilation=1)

        self.block = nn.Sequential(
                    nn.Conv2d(kernel_size=(3,3), in_channels=in_channels, out_channels=mid_channels),
                    nn.ReLU(),
                    nn.BatchNorm2d(mid_channels),
                    nn.Conv2d(kernel_size=(3,3), in_channels=mid_channels, out_channels=out_channels),
                    nn.ReLU(),
                    nn.BatchNorm2d(out_channels)
                    )
        
    def forward(self, e, d): #d和e分别是拼接前的两个元件 分别从收缩路径e来&扩张路径e来的两个
        #举一个例子： 第一个操作路径的e是64*64对x和y位置减去28*28 
        d = self.up(d)
        #concat  **回忆一下tensor的纬度[B,C,H,W] 然后对应位置相减 得出x&y的差值
        diffY = e.size()[2] - d.size()[2]
        diffX = e.size()[3] - d.size()[3]
        e = e[:,:, diffY//2:e.size()[2]-diffY//2, diffX//2:e.size()[3]-diffX//2]
        cat = torch.cat([e, d], dim=1)#在索引1--channel拼接
        out = self.block(cat)
        return out


def final_block(in_channels, out_channels):
    block = nn.Sequential(
            nn.Conv2d(kernel_size=(1,1), in_channels=in_channels, out_channels=out_channels),
            nn.ReLU(),
            nn.BatchNorm2d(out_channels),
            )
    return  block


class UNet(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(UNet, self).__init__()
        #Encode
        self.conv_encode1 = contracting_block(in_channels=in_channel, out_channels=64)
        self.conv_pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv_encode2 = contracting_block(in_channels=64, out_channels=128)
        self.conv_pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv_encode3 = contracting_block(in_channels=128, out_channels=256)
        self.conv_pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv_encode4 = contracting_block(in_channels=256, out_channels=512)
        self.conv_pool4 = nn.MaxPool2d(kernel_size=2, stride=2)
        # Bottleneck #Bottleneck  图中间 最下面两 当做“过渡层”处理 思路会清晰一点 *实际上跟收缩一样
        self.bottleneck = torch.nn.Sequential(
                            nn.Conv2d(kernel_size=3, in_channels=512, out_channels=1024),
                            nn.ReLU(),
                            nn.BatchNorm2d(1024),
                            nn.Conv2d(kernel_size=3, in_channels=1024, out_channels=1024),
                            nn.ReLU(),
                            nn.BatchNorm2d(1024)
                            )
        # Decode
        self.conv_decode4 = expansive_block(1024, 512, 512)
        self.conv_decode3 = expansive_block(512, 256, 256)
        self.conv_decode2 = expansive_block(256, 128, 128)
        self.conv_decode1 = expansive_block(128, 64, 64)
        self.final_layer = final_block(64, out_channel)
    
    def forward(self, x):
        #set_trace()
        # Encode
        encode_block1 = self.conv_encode1(x);print('encode_block1:', encode_block1.size())
        encode_pool1 = self.conv_pool1(encode_block1);print('encode_pool1:', encode_pool1.size())
        encode_block2 = self.conv_encode2(encode_pool1);print('encode_block2:', encode_block2.size())
        encode_pool2 = self.conv_pool2(encode_block2);print('encode_pool2:', encode_pool2.size())
        encode_block3 = self.conv_encode3(encode_pool2);print('encode_block3:', encode_block3.size())
        encode_pool3 = self.conv_pool3(encode_block3);print('encode_pool3:', encode_pool3.size())
        encode_block4 = self.conv_encode4(encode_pool3);print('encode_block4:', encode_block4.size())
        encode_pool4 = self.conv_pool4(encode_block4);print('encode_pool4:', encode_pool4.size())
        
        # Bottleneck
        bottleneck = self.bottleneck(encode_pool4);print('bottleneck:', bottleneck.size())
        
        # Decode
        decode_block4 = self.conv_decode4(encode_block4, bottleneck);print('decode_block4:', decode_block4.size())
        decode_block3 = self.conv_decode3(encode_block3, decode_block4);print('decode_block3:', decode_block3.size())
        decode_block2 = self.conv_decode2(encode_block2, decode_block3);print('decode_block2:', decode_block2.size())
        decode_block1 = self.conv_decode1(encode_block1, decode_block2);print('decode_block1:', decode_block1.size())
        
        final_layer = self.final_layer(decode_block1)
        return final_layer


if __name__ == "__main__":

    rgb = torch.randn(1, 3, 572, 572)
    net = UNet(3, 12)
    out = net(rgb)
    print(out.shape)


'''
Review
镜像折叠保护图像边缘信息
    没有padding 但是后面的fusionnet加入了padding
    review:
    padding的三种模式full/same/vaild 本质是对卷积移动的限制
            https://blog.csdn.net/qq_26552071/article/details/86498227
加权损失函数
    omega c -- 具体看linknet
    后面那一串   d1(x) == 表示图中某一北京像素点与离这个点最近的细胞最近的距离； d2 表示图中某一北京像素点与离这个点第二近的细胞最近的距离
        ×最后的效果就是 细胞边界的权重会给大一些

关于医学图像分割的指标 (medical segmentation index)：
像素误差 Pixel Error: 比较预测的label和实际的label，错误的点除以总数，就是像素误差
兰德误差 Rand error: 评级两个聚类的相似度评价方法
弯曲误差 Warpping error： 衡量分割目标的拓扑形状效果  ×汉明距离
×more on： https://blog.csdn.net/Asun0204/article/details/79002875?utm_source=blogxgwz6
'''