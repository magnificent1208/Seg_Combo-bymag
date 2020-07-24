import os
import torch 
import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import torchvision.transforms.functional as ttf
import cfg

'''
dataset.py 任务
#1 读取数据
#2 按尺寸大小中心裁剪
#3 标签编码
#4 numpy to tensor 
#5 实例化数据类
₍₍ (̨̡ ‾᷄ᗣ‾᷅ )̧̢ ₎₎
'''


class LabelProcessor:
    '''
    对标签图像的编码
    '''
    def __init__(self, file_path):
        self.colormap = self.read_color_map(file_path)
        self.cm21b1 = self.encode_label_pix(self.colormap)

    '''
    静态方法修饰器，可以理解为定义在累中的普通函数，用self.___调用
    静态方法内部，不能实例属性和实列对象，即不可以调用self。相关的内容
    使用的原因是为了程序设计需要：代码简介，封装功能等。
    详细：关于类方法、静态方法和实例方法的详解 可以前后切换知识 http://c.biancheng.net/view/4552.html
    '''

    #camvid比别的要麻烦一点的是，标注是个png图，然后要读取csv获取相应类别
    #而不是直接读取 class=[] &C colormap[]的东西
    @staticmethod
    def read_color_map(file_path):
        pd_label_color = pd.read_csv(file_path, sep=',') #pd.read_csv() https://blog.csdn.net/The_Time_Runner/article/details/86187900
        colormap = []
        for iter in range(len(pd_label_color.index)):
            tmp = pd_label_color.iloc[iter]
            color = [tmp['r'], tmp['g'], tmp['b']]
            colormap.append(color)
        return colormap
    
    #数据处理-数据结构-哈希表 建立映射关系，提高查询效率
    #read_color_map完成了 颜色-标签 对应关系
    #encode_label_pix则完成 color-像素-标签 对应关系
    #参考对PASCAL VOC处理(但少一层映射关系): https://blog.csdn.net/qq_32146369/article/details/106292998
    
    '''
    h希函数：(cm[0]*256 +cm[1]）*256 +cm[2]
    哈希映射：cm2lbl[(cm[0]*256 +cm[1])*256 +cm[2]] = i 
    哈希表：cm2lbl

    举个栗子儿～ 
    一个像素点  P(128,64,128)，由编码函数（P[0]*256 + P[1]）*256 + P[2]转化为整数（8405120）,
    将这个数字作为像素点P在cm2lbl编码表中索引cm2lbl[8405120]
    去查询X像素点P[128,64,128]对应的检测类别i
    '''
    
    #哈希表
    @staticmethod
    def encode_label_pix(colormap):
        cm2lbl = np.zeros(256**3) # 每个像素点有 0 ~ 255 的选择，RGB 三个通道
        for i,cm in enumerate(colormap):
            cm2lbl[(cm[0]*256+cm[1])*256+cm[2]] = i # 建立索引
        return cm2lbl
    #哈希映射
    #矩阵化批量操作像素点的编码 输入像素P（，，）即索引值，在哈希表查找，返回类别
    def encode_label_img(self,img):
        data = np.array(img, dtype='int32')
        idx = data[:, :, 0] * 256 + data[:, :, 1] * 256 + data[:, :, 2]
        return np.array(self.cm21b1[idx], dtype='int64')



#在PyTorch中，数据加载需要自定义数据集类，并用此类来实例化数据对象，
#实现自定义的数据集需要继承torch.utils.data包中的Dataset类。
#参考 class VOCSegDataset() https://zhuanlan.zhihu.com/p/32506912

class CamvidDataset(Dataset):
    def __init__(self, file_path=[], crop_size=None):
        '''
        params:
            files_path(list): 数据&标签路径，列表元素第一个是图片路径，第二个是标签路径
        '''
        
        #1 正确读图片&标签路径
        #判断列表是否有两个元素 -- 图片&标签路径
        if len(file_path) !=2:  
            raise ValueError("同时需要图片和标签文件夹的路径，（图片路径在前）")
            # python的raise用法--python的异常处理机制 详见 http://c.biancheng.net/view/2360.html
        self.img_path = file_path[0]
        self.label_path = file_path[1]
        #2 从路径中取出的图片&标签的文件名保持到两个列表中
        self.imgs = self.read_file(self.img_path)
        self.labels = self.read_file(self.label_path)
        #3 初始化数据处理函数
        self.crop_size=crop_size

    #对原图&标签的处理    
    def __getitem__(self, idx):
        #self.img pytorch的写法,直接调用之前定义的内部变量
        #[idx] 也是pytorch里面写好的，读取列表里面的元素数 
        img = self.imgs[idx]
        label = self.labels[idx]

        #从路径&文件名，读取数据（注意，camvid数据集的图片&标签都是png格式的图像数据哦哦）
        img = Image.open(img)
        label = Image.open(label).convert('RGB')  #统一转一下3通道，免得之后的麻烦

        #中心裁剪 需要输入有 图片、标签、裁剪尺寸
        img, label = self.center_crop(img, label, self.crop_size)
        img, label = self.img_transforms(img, label, self.crop_size)

        sample = {'img': img, 'label': label}
        return sample

    def read_file(self,path):
    #从文件夹中读取数据
    
        files_list = os.listdir(path) #读取文件夹下所有名称列表
        '''
        print(files_list)
        返回
        ['234.png','123.png','345.png', ...]
        '''
        file_path_list =[os.path.join(path,img) for img in files_list] #路径拼接 把path跟img名拼起来
        '''
        print(file_path_list)
        返回
        ['./camvid/train/234/png','./camvid/train/123.png', ...]
        '''
        file_path_list.sort() #路径排序 sort&sorted比较详见 https://www.jianshu.com/p/7be04a3f30cd
        return file_path_list
        '''
      返回
      ['./camvid/train/123.png','./camvid/train/234/png' ...]
     '''

    def center_crop(self, data,label,crop_size):
    #裁剪输入的图片&标签的大小 
        data = ttf.center_crop(data, crop_size)
        label = ttf.center_crop(label, crop_size)
        return data,label

    def img_transform(self, img, label):
    #把label里面的数据值变成整型
        label = np.array(label) #label转np数组
        label = Image.fromarray(label.astype('uint8')) #转int8 再转Imagearray
    #针对原图操作
        transform_img = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        
        img = transform_img(img)
        #针对标签操作
        ##编码
        label = label_processor.encode_label_img(label)
        label = torch.from_numpy(label) #转张量方式 关于torch.from_numpy VS torch.Tensor 详细： https://blog.csdn.net/github_28260175/article/details/105382060
    
        return img, label

class LoadDataset(Dataset):
    def __init__(self, file_path=[], crop_size=None):
        """para:
            file_path(list): 数据和标签路径,列表元素第一个为图片路径，第二个为标签路径
        """
        # 1 正确读入图片和标签路径
        if len(file_path) != 2:
            raise ValueError("同时需要图片和标签文件夹的路径，图片路径在前")
        self.img_path = file_path[0]
        self.label_path = file_path[1]
        # 2 从路径中取出图片和标签数据的文件名保持到两个列表当中（程序中的数据来源）
        self.imgs = self.read_file(self.img_path)
        self.labels = self.read_file(self.label_path)
        # 3 初始化数据处理函数设置
        self.crop_size = crop_size

    #对原图&标签的处理    
    def __getitem__(self, index):
        img = self.imgs[index]
        label = self.labels[index]
        # 从文件名中读取数据（图片和标签都是png格式的图像数据）
        img = Image.open(img)
        label = Image.open(label).convert('RGB')

        img, label = self.center_crop(img, label, self.crop_size)

        img, label = self.img_transform(img, label)
        # print('处理后的图片和标签大小：',img.shape, label.shape)
        sample = {'img': img, 'label': label}

        return sample

    def __len__(self):
        return len(self.imgs)

    def read_file(self, path):
        """从文件夹中读取数据"""
        files_list = os.listdir(path)
        file_path_list = [os.path.join(path, img) for img in files_list]
        file_path_list.sort()
        return file_path_list

    def center_crop(self, data,label,crop_size):
    #裁剪输入的图片&标签的大小 
        data = ttf.center_crop(data, crop_size)
        label = ttf.center_crop(label, crop_size)
        return data,label

    def img_transform(self, img, label):
    #把label里面的数据值变成整型
        label = np.array(label) #label转np数组
        label = Image.fromarray(label.astype('uint8')) #转int8 再转Imagearray
    #针对原图操作
        transform_img = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]
        )
        img = transform_img(img)
        #针对标签操作
        ##编码
        label = label_processor.encode_label_img(label)
        label = torch.from_numpy(label) #转张量方式 关于torch.from_numpy VS torch.Tensor 详细： https://blog.csdn.net/github_28260175/article/details/105382060
    
        return img, label

label_processor = LabelProcessor(cfg.class_dict_path)

 
if __name__ == "__main__":

    Cam_train = CamvidDataset([TRAIN_ROOT, TRAIN_LABEL], crop_size)
    Cam_val = CamvidDataset([VAL_ROOT, VAL_LABEL], crop_size)
    Cam_test = CamvidDataset([TEST_ROOT, TEST_LABEL], crop_size)    

