import math
import torch
import torch.nn as nn
import torch.nn.functional as F


#1. CLASS deeplabv3+ 
#2. ASPP模块
#3. 修改后的Xception

class DeepLabv3_plus(nn.Module):
    def __init__(self, nInputChannels=3, n_classes=21, os=16, _print=True):
        if _print:
            print("contructing Deeplabv3+ model")
            print("backbine Xception")

#用并行的方式
class ASPP_module(nn.Module):
    '''Encoder-Decoder with Atrous Separable Convolution for Semantic Image
    Segmentation.

    This head is the implementation of `DeepLabV3+
    <https://arxiv.org/abs/1802.02611>`_.

    Args:
        inplanes(int): The input channels of c1 decoder.
        planes(int):The intermediate channels of c1 decoder
        os(int): output stride / dialations rate

    '''
    def __init__(self, inplanes, planes, os): 
        super(ASPP_module, self).__init__()

        #ASPP  
        '''output  stride  setting, check out deeplabv3 paper part3.3 for more details.
           "In the end, our improved ASPP consists of (a) one 1x1 convolution
            and three 3 x 3 convolutions with rates = (6; 12; 18) when
            output stride = 16 (all with 256 filters and batch normalization),
            and (b) the image-level features, as shown in Fig. 5" 
            *** fig.5 = ASPP.png in dir
        '''
        if os == 16:
            dilations = [1, 6, 12, 18]
        if os == 8:
            dilations = [1, 12, 24, 36]

        #part a: ASPP
        #coz of Parallel computing, all aspp# have the same input cshannel 
        self.aspp1 = nn.Sequential(
            nn.Conv2d(inplanes, planes, kernel_size=1, stride=1,padding=0,dilation=dilations[0],bias=False),
            nn.BatchNorm2d(planes),
            nn.ReLU())
        self.aspp2 = nn.Sequential(
            nn.Conv2d(inplanes, planes, kernel_size=3, stride=1,padding=dilations[1],dilation=dilations[1],bias=False),
            nn.BatchNorm2d(planes),
            nn.ReLU())
        self.aspp3 = nn.Sequential(
            nn.Conv2d(inplanes, planes, kernel_size=3, stride=1,padding=dilations[2],dilation=dilations[2],bias=False),
            nn.BatchNorm2d(planes),
            nn.ReLU())
        self.aspp4 = nn.Sequential(
            nn.Conv2d(inplanes, planes, kernel_size=3, stride=1,padding=dilations[3],dilation=dilations[3],bias=False),
            nn.BatchNorm2d(planes),
            nn.ReLU())

        #part b: global average pooling
        self.glonbal_avg_pool = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)),
                nn.Conv2d(2048, 256, 1, stride=1, bias=False),
                nn.BatchNorm2d(256),
                nn.ReLU())


        #
        '''
        5 channels(aspp1-4+conv1)
        after concat: 1280 -- 256*5
        '''
        self.conv1 = nn.Conv2d(1280, 256, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(256)

        self.__init__weight

    def forward(self, x):
        x1 = self.aspp1(x) ;print(x1.size())
        x2 = self.aspp2(x) ;print(x2.size())
        x3 = self.aspp3(x) ;print(x3.size())
        x4 = self.aspp4(x) ;print(x4.size())
        x5 = self.glonbal_avg_pool(x) ;print(x5.size())
        #插值把pooling结果上采样到跟别的aspp模块一样 就改最后两个值 所以[2：] 选择双线性插值
        x5 = F.interpolate(x5, size=x4.size()[2:], mode='bilinear', align_corners=True) ;print(x5.size())
        
        x = torch.cat((x1,x2,x3,x4,x5), dim=1) ;print(x.size())
        #用1x1卷积转通道数
        x = self.conv1(x) ;print(x.size())
        x = self.bn1(x) ;print(x.size())

        return x

    #卷积核&batchn做初始化  more on:10种pytorch权重初始化的方法 https://www.cnblogs.com/jfdwd/p/11269622.html
    def __init__weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2./n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

##sim data
model =ASPP_module(2048, 256,16)
model.eval()
image = torch.randn(1,2048,176,240)
output=model(image)
print(output.size)
            


