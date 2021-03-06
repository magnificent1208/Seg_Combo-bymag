
'''
1. 读取数据
2. 加载模型
3. 配置参数
4. 训练&打印结果 各色指标
5. 验证pyt
6. 保存权重
'''

import torch as t
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from datetime import datetime
from dataset import CamvidDataset
from dataset import LoadDataset
from eval_semantic_segmentation import eval_semantic_segmentation
from models.fcn import FCN as FCN 
import cfg

#制定训练的硬件场景咯
device = t.device('cuda') if t.cuda.is_available() else t.device('cpu')
num_class = cfg.DATASET[1]

Load_train = LoadDataset([cfg.TRAIN_ROOT, cfg.TRAIN_LABEL], cfg.crop_size)
Load_val = LoadDataset([cfg.VAL_ROOT, cfg.VAL_LABEL], cfg.crop_size)

train_data = DataLoader(Load_train, batch_size=cfg.BATCH_SIZE, shuffle=True, num_workers=1)
val_data = DataLoader(Load_val, batch_size=cfg.BATCH_SIZE, shuffle=True, num_workers=1)

#FCN模型类实例化
fcn = FCN(num_class)
fcn = fcn.to(device)

#Loss function  因为之后有可能会尝试别的激活函数，所以不用封装好的CEL
#CrossEntropyLoss() 函数联合调用了 nn.LogSoftmax() 和 nn.NLLLoss() More on ：
#https://blog.csdn.net/jasonleesjtu/article/details/89141554?utm_medium=distribute.pc_relevant.none-task-blog-BlogCommendFromMachineLearnPai2-2.nonecase&depth_1-utm_source=distribute.pc_relevant.none-task-blog-BlogCommendFromMachineLearnPai2-2.nonecase
criterion = nn.NLLLoss().to(device) 
#optimizer 经验：一般2D数据集Adam效果比较好 如果是3D数据集SGD会好些
optimizer = optim.Adam(fcn.parameters(), lr=1e-4)  


def train(model):
    best = [0]  #初始化 保存最好的值
    net = model.train() #torch 训练模式
    # 训练轮次 
    for epoch in range(cfg.EPOCH_NUMBER):
        print('Epoch is [{}/{}]'.format(epoch + 1, cfg.EPOCH_NUMBER)) #log
        if epoch % 50 == 0 and epoch != 0:  #每50次降低学习率下降一半
            for group in optimizer.param_groups:
                group['lr'] *= 0.5

        train_loss = 0
        train_acc = 0
        train_miou = 0
        train_class_acc = 0
        # 训练批次

        for i, sample in enumerate(train_data):
            # 载入数据 其实新版不需要Variable 但是习惯写也不报错
            img_data = Variable(sample['img'].to(device))
            img_label = Variable(sample['label'].to(device))
            # 训练
            out = net(img_data)
            out = F.log_softmax(out, dim=1) #因为上面用的NLLloss 所以手动激活下
            loss = criterion(out, img_label)
            optimizer.zero_grad() #手动梯度清零
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

            # 评估
            pre_label = out.max(dim=1)[1].data.cpu().numpy()
            pre_label = [i for i in pre_label]

            true_label = img_label.data.cpu().numpy()
            true_label = [i for i in true_label]

            eval_metrix = eval_semantic_segmentation(pre_label, true_label)
            train_acc += eval_metrix['mean_class_accuracy']
            train_miou += eval_metrix['miou']
            train_class_acc += eval_metrix['class_accuracy']

            print('|batch[{}/{}]|batch_loss {: .8f}|'.format(i + 1, len(train_data), loss.item()))

        metric_description = '|Train Acc|: {:.5f}|Train Mean IU|: {:.5f}\n|Train_class_acc|:{:}'.format(
            train_acc / len(train_data),
            train_miou / len(train_data),
            train_class_acc / len(train_data),
        )

        print(metric_description)
        if max(best) <= train_miou / len(train_data):
            best.append(train_miou / len(train_data))
            t.save(net.state_dict(), './Results/weights/FCN_weight/{}.pth'.format(epoch))


def evaluate(model):
    net = model.eval()
    eval_loss = 0
    eval_acc = 0
    eval_miou = 0
    eval_class_acc = 0

    prec_time = datetime.now()
    for j, sample in enumerate(val_data):
        valImg = Variable(sample['img'].to(device))
        valLabel = Variable(sample['label'].long().to(device))

        out = net(valImg)
        out = F.log_softmax(out, dim=1)
        loss = criterion(out, valLabel)
        eval_loss = loss.item() + eval_loss
        pre_label = out.max(dim=1)[1].data.cpu().numpy()
        pre_label = [i for i in pre_label]

        true_label = valLabel.data.cpu().numpy()
        true_label = [i for i in true_label]

        eval_metrics = eval_semantic_segmentation(pre_label, true_label)
        eval_acc = eval_metrics['mean_class_accuracy'] + eval_acc
        eval_miou = eval_metrics['miou'] + eval_miou

    cur_time = datetime.now()
    h, remainder = divmod((cur_time - prec_time).seconds, 3600)
    m, s = divmod(remainder, 60)
    time_str = 'Time: {:.0f}:{:.0f}:{:.0f}'.format(h, m, s)

    val_str = ('|Valid Loss|: {:.5f} \n|Valid Acc|: {:.5f} \n|Valid Mean IU|: {:.5f} \n|Valid Class Acc|:{:}'.format(
        eval_loss / len(train_data),
        eval_acc / len(val_data),
        eval_miou / len(val_data),
        eval_class_acc / len(val_data)))
    print(val_str)
    print(time_str)


if __name__ == "__main__":
    train(fcn)
