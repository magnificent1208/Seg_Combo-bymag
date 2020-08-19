import math
import torch
import torch.nn as nn
import torch.nn.functional as F


#1. ASPP模块 [Encoder]
#2. 修改后的Xception [Encoder]
    #2.1 depthwise conv
    #2.2 Block 包含repetitive conv & 
#3. CLASS deeplabv3+ [Decode定义处]



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
        self.global_avg_pool = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)),
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

        self._init_weight()

    def forward(self, x):
        x1 = self.aspp1(x) ;print(x1.size())
        x2 = self.aspp2(x) ;print(x2.size())
        x3 = self.aspp3(x) ;print(x3.size())
        x4 = self.aspp4(x) ;print(x4.size())
        x5 = self.global_avg_pool(x) ;print(x5.size())
        #插值把pooling结果上采样到跟别的aspp模块一样 就改最后两个值 所以[2：] 选择双线性插值
        x5 = F.interpolate(x5, size=x4.size()[2:], mode='bilinear', align_corners=True) ;print(x5.size())
        
        x = torch.cat((x1,x2,x3,x4,x5), dim=1) ;print(x.size())
        #用1x1卷积转通道数
        x = self.conv1(x) ;print(x.size())
        x = self.bn1(x) ;print(x.size())

        return x

    #卷积核&batchn做初始化  more on:10种pytorch权重初始化的方法 https://www.cnblogs.com/jfdwd/p/11269622.html
    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
'''
##sim data for aspp testing
model = ASPP_module(2048, 256,16)
model.eval()
image = torch.randn(1,2048,176,240)
output=model(image)
print(output.size)

'''

### 修改后的Xception 组件们 
# Atrous depthwise conv / Depthwise conv / Points conv
# 组卷积 
class SeparableConv2d_same(nn.Module):
    def __init__(self, inplanes, planes, kernel_size=3, stride=1, dilation =1, bias=False):
        super(SeparableConv2d_same, self).__init__()
        self.depth_conv = nn.Conv2d(inplanes, inplanes, kernel_size, stride, padding=0, dilation=1, groups=inplanes, bias=False)
        self.pointwise = nn.Conv2d(inplanes, planes, kernel_size=1, stride=1, padding=0, dilation=1, groups=1, bias=False)
        
    def forward(self, input):
        out = self.depth_conv(input)
        out = self.pointwise(out)
        return out


# block
class Block(nn.Module):
    '''
    Args:
        inplanes(int): The input channels
        planes(int):The intermediate channels
        reps(int): Num of repetitive convolutions (Sep Conv 3x3) at each block
        dilations (int): Dilation rate of each layer.
        start_with_relu(int): some conv doesnt start wif relu 1-19
        is_last(int): for block #20 

    '''
    def __init__(self, inplanes, planes, reps, stride = 1, dilation = 1, start_with_relu=True, grow_first=True, is_last=True):
        super(Block, self).__init__()
        #for skip block
        if planes != inplanes or stride != 1:
            self.skip = nn.Conv2d(inplanes, planes, stride, bias=False)
            self.skipbn = nn.BatchNorm2d(planes)
        #for Middle flow
        else:
            self.skip = None
        
        self.relu = nn.ReLU(inplace=True)
        rep = [] #列表里面返回的元素就是 一个block'的封装

        #后面的模块 通道数 输入输出是一样的，所以有个filters存一下
        filters = planes
        if grow_first:
            rep.append(self.relu)
            rep.append(SeparableConv2d_same(inplanes, planes, 3, stride=1, dilation=dilation)) #适用于所有不带stride的conv
            rep.append(nn.BatchNorm2d(planes))
            filters = planes
        

        #看要将上面4行重复几次
        for i in range(reps -1):
            rep.append(self.relu)
            rep.append(SeparableConv2d_same(filters,filters,3,stride=1, dilation=dilation))
            rep.append(nn.BatchNorm2d(filters))
        
        if not grow_first:
            rep.append(self.relu)
            rep.append(SeparableConv2d_same(inplanes, planes, 3, stride=1, dilation=dilation))
            rep.append(nn.BatchNorm2d(planes))

        #到此为止middle flow的block都定义完了
        if not start_with_relu:
            rep = rep[1:]#就跳过前面的

        if stride != 1: #在Entry flow 和 Exit flow 变小的卷积 就是红色部分
            rep.append(SeparableConv2d_same(planes, planes, 3,stride=2))#net图里面的红色
        
        if stride == 1 and is_last: #block20
            rep.append(SeparableConv2d_same(planes, planes, 3, stride=1))

        self.rep = nn.Sequential(*rep) #*rep形参，可以接收参数，转为元组方式

    def forward(self, inp):
        x = self.rep(inp)

        if self.skip is not None:
            skip = self.skip(inp)
            skip = self.skipbn(skip)
        else:
            skip = inp

        x += skip #融合

        return x
    
class Xception(nn.Module):
    """
    Modified Alighed Xception
    """
    def __init__(self, inplanes=3, os=16):
        super(Xception, self).__init__()

        if os == 16:
            entry_block3_stride = 2
            middle_block_dilation = 1
            exit_block_dilations = (1, 2)
        elif os == 8:
            entry_block3_stride = 1
            middle_block_dilation = 2
            exit_block_dilations = (2, 4)
        else:
            raise NotImplementedError 


        # 对着network图看
        # Entry flow
        self.conv1 = nn.Conv2d(inplanes, 32, 3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(32,64,3,stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(64)

        self.block1 = Block(64, 128, reps=2, stride=2, start_with_relu=False)
        self.block2 = Block(128, 256, reps=2, stride=2, start_with_relu=True, grow_first=True)
        self.block3 = Block(256, 728, reps=2, stride=entry_block3_stride, start_with_relu=True, grow_first=True,
                            is_last=True)

        # Middle Flow  repeat 16 times
        self.block4 = Block(728, 728, reps=3, stride=1, dilation=middle_block_dilation, start_with_relu=True, grow_first=True)
        self.block5 = Block(728, 728, reps=3, stride=1, dilation=middle_block_dilation, start_with_relu=True, grow_first=True)
        self.block6 = Block(728, 728, reps=3, stride=1, dilation=middle_block_dilation, start_with_relu=True, grow_first=True)
        self.block7 = Block(728, 728, reps=3, stride=1, dilation=middle_block_dilation, start_with_relu=True, grow_first=True)
        self.block8 = Block(728, 728, reps=3, stride=1, dilation=middle_block_dilation, start_with_relu=True, grow_first=True)
        self.block9 = Block(728, 728, reps=3, stride=1, dilation=middle_block_dilation, start_with_relu=True, grow_first=True)
        self.block10 = Block(728, 728, reps=3, stride=1, dilation=middle_block_dilation, start_with_relu=True, grow_first=True)
        self.block11 = Block(728, 728, reps=3, stride=1, dilation=middle_block_dilation, start_with_relu=True, grow_first=True)
        self.block12 = Block(728, 728, reps=3, stride=1, dilation=middle_block_dilation, start_with_relu=True, grow_first=True)
        self.block13 = Block(728, 728, reps=3, stride=1, dilation=middle_block_dilation, start_with_relu=True, grow_first=True)
        self.block14 = Block(728, 728, reps=3, stride=1, dilation=middle_block_dilation, start_with_relu=True, grow_first=True)
        self.block15 = Block(728, 728, reps=3, stride=1, dilation=middle_block_dilation, start_with_relu=True, grow_first=True)
        self.block16 = Block(728, 728, reps=3, stride=1, dilation=middle_block_dilation, start_with_relu=True, grow_first=True)
        self.block17 = Block(728, 728, reps=3, stride=1, dilation=middle_block_dilation, start_with_relu=True, grow_first=True)
        self.block18 = Block(728, 728, reps=3, stride=1, dilation=middle_block_dilation, start_with_relu=True, grow_first=True)
        self.block19 = Block(728, 728, reps=3, stride=1, dilation=middle_block_dilation, start_with_relu=True, grow_first=True)


        # Exit flow
        self.block20 = Block(728, 1024, reps=2, stride=1, dilation=exit_block_dilations[0],
                             start_with_relu=True, grow_first=False, is_last=True)

        self.conv3 = SeparableConv2d_same(1024, 1536, 3, stride=1, dilation=exit_block_dilations[1])
        self.bn3 = nn.BatchNorm2d(1536)

        self.conv4 = SeparableConv2d_same(1536, 1536, 3, stride=1, dilation=exit_block_dilations[1])
        self.bn4 = nn.BatchNorm2d(1536)

        self.conv5 = SeparableConv2d_same(1536, 2048, 3, stride=1, dilation=exit_block_dilations[1])
        self.bn5 = nn.BatchNorm2d(2048)

        #后面进入ASPP都是2048

        #init_weights
        self._init_weight()

    def forward(self, x):
        # Entry flow
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        x = self.block1(x)
        low_level_feat = x  #原图的1/4大小的这fm 扔到Decoder那边去
        x = self.block2(x)
        x = self.block3(x)

        # Middle flow
        x = self.block4(x)
        x = self.block5(x)
        x = self.block6(x)
        x = self.block7(x)
        x = self.block8(x)
        x = self.block9(x)
        x = self.block10(x)
        x = self.block11(x)
        x = self.block12(x)
        x = self.block13(x)
        x = self.block14(x)
        x = self.block15(x)
        x = self.block16(x)
        x = self.block17(x)
        x = self.block18(x)
        x = self.block19(x)

        # Exit flow
        x = self.block20(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)

        x = self.conv4(x)
        x = self.bn4(x)
        x = self.relu(x)

        x = self.conv5(x)
        x = self.bn5(x)
        x = self.relu(x)

        #回一下network，传到Decode是两个东西，所以这里要记得返回两个
        return x, low_level_feat


    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

        
#Decoder
class DeepLabv3_plus(nn.Module):
    def __init__(self, nInputChannels=3, n_classes=21, os=16, _print=True):
        if _print:
            print("contructing Deeplabv3+ model")
            print("backbone Xception")
            print("Number of classes: {}".format(n_classes))
            print("Output stride: {}".format(os))
            print("Number of Input Channels: {}".format(nInputChannels))

        super(DeepLabv3_plus, self).__init__()

        # Atrous Convs
        self.xception_features = Xception(nInputChannels, os)

        self.ASPP = ASPP_module(2048, 256, 16)

        self.conv1 = nn.Conv2d(256,256,1,bias=False)
        self.bn1 = nn.BatchNorm2d(256)
        self.relu = nn.ReLU()

        #low level features
        #adopt [1x1, 48] for channel redution.
        self.conv2 = nn.Conv2d(128,48,1,bias=False)
        self.bn2 = nn.BatchNorm2d(48)

        self.last_conv = nn.Sequential(nn.Conv2d(304, 256, kernel_size=3, stride=1, padding=1, bias=False),
                                       nn.BatchNorm2d(256),
                                       nn.ReLU(),
                                       nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
                                       nn.BatchNorm2d(256),
                                       nn.ReLU(),
                                       nn.Conv2d(256, n_classes, kernel_size=1, stride=1))


    def forward(self, input):
        x, low_level_features = self.xception_features(input)
        x = self.ASPP(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = F.interpolate(x, size=(int(math.ceil(input.size()[-2]/4)),
                                int(math.ceil(input.size()[-1]/4))), mode='bilinear', align_corners=True)

        low_level_features = self.conv2(low_level_features)
        low_level_features = self.bn2(low_level_features)
        low_level_features = self.relu(low_level_features)

        x = torch.cat((x, low_level_features), dim=1)
        x = self.last_conv(x)
        x = F.interpolate(x, size=input.size()[2:], mode='bilinear', align_corners=True)

        return x

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


if __name__ == "__main__":
    model = DeepLabv3_plus(nInputChannels=3, n_classes=12, os=16, _print=True)
    model.eval()
    image = torch.randn(1, 3, 352, 480)
    output = model(image)
    print(output.size())