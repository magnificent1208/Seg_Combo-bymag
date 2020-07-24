'''
1. 读取数据
2. 加载权重
3. 测试&打印结果
'''

import torch as t
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import DataLoader
from eval_semantic_segmentation import eval_semantic_segmentation
from dataset import LoadDataset
from models.fcn import FCN as FCN 
import cfg

device = t.device('cuda') if t.cuda.is_available() else t.device('cpu')
num_class = cfg.DATASET[1]

BATCH_SIZE = 4
miou_list = [0]

Load_test = LoadDataset([cfg.TEST_ROOT, cfg.TEST_LABEL], cfg.crop_size)
test_data = DataLoader(Load_test, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

net = FCN(num_class)
net.eval()
net.to(device)
net.load_state_dict(t.load("./Results/weights/FCN_weight/2.pth"))

train_acc = 0
train_miou = 0
train_class_acc = 0
train_mpa = 0
error = 0

for i, sample in enumerate(test_data):
	data = Variable(sample['img']).to(device)
	label = Variable(sample['label']).to(device)
	out = net(data)
	out = F.log_softmax(out, dim=1)

	pre_label = out.max(dim=1)[1].data.cpu().numpy()
	pre_label = [i for i in pre_label]

	true_label = label.data.cpu().numpy()
	true_label = [i for i in true_label]

	eval_metrix = eval_semantic_segmentation(pre_label, true_label)
	train_acc = eval_metrix['mean_class_accuracy'] + train_acc
	train_miou = eval_metrix['miou'] + train_miou
	train_mpa = eval_metrix['pixel_accuracy'] + train_mpa
	if len(eval_metrix['class_accuracy']) < 12:
		eval_metrix['class_accuracy'] = 0
		train_class_acc = train_class_acc + eval_metrix['class_accuracy']
		error += 1
	else:
		train_class_acc = train_class_acc + eval_metrix['class_accuracy']

	print(eval_metrix['class_accuracy'], '================', i)


epoch_str = ('test_acc :{:.5f} ,test_miou:{:.5f}, test_mpa:{:.5f}, test_class_acc :{:}'.format(train_acc /(len(test_data)-error),
															train_miou/(len(test_data)-error), train_mpa/(len(test_data)-error),
															train_class_acc/(len(test_data)-error)))

if train_miou/(len(test_data)-error) > max(miou_list):
	miou_list.append(train_miou/(len(test_data)-error))
	print(epoch_str+'==========last')