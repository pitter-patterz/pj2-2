

from __future__ import  absolute_import
from utils.config import opt
from data.dataset import Dataset,TestDataset,inverse_normalize
from model import FasterRCNNVGG16
from torch.utils import data as data_
from trainer import FasterRCNNTrainer
from eval_new import eval_      

import numpy as np

dataset = Dataset(opt)
print('\nPrepare test data...')
testset = TestDataset(opt)
test_dataloader = data_.DataLoader(testset,batch_size=1,num_workers=opt.test_num_workers,shuffle=True,pin_memory=True)

print('\n...Load the trained faster-RCNN...')
faster_rcnn = FasterRCNNVGG16()
trainer = FasterRCNNTrainer(faster_rcnn).cuda()
trainer.load('./checkpoints/selfnet_10epoch.pth')
opt.caffe_pretrain=False

print('\n Begin test...the test sample size is 200 (by default).')
map,miou,loss = eval_(test_dataloader,trainer.faster_rcnn,trainer,test_num=200)
print('\nAP\n',map)
print('\nIOU\n',miou)

# No sufficient memory to run eval_() on the whole test dataset.

# MAP,MIOU = [],[]
# for s in np.linspace(0,4000,5):
#   map,miou,loss = eval_(test_dataloader,trainer.faster_rcnn,trainer,test_num=1000,start=int(s))
#   print('\nAP\n',map)
#   print('\nIOU\n',miou)
#   MAP.append(map)
#   MIOU.append(miou)

# test_miou = sum(MIOU)/len(MIOU)
# test_ap = np.zeros(shape=20)
# for s in MAP:
#   test_ap += 0.2*s['ap']
# test_map = np.mean(test_ap)

# print('\nAP',test_ap,'\nmAP',test_map,'\nmIOU',test_miou)

# AP [0.68051285 0.70204686 0.6286386  0.49272414 0.43532783 0.76973199
#  0.80180261 0.79055209 0.46761908 0.68072967 0.63535084 0.72763277
#  0.78347192 0.74117988 0.7357755  0.34320405 0.61806328 0.63957931
#  0.73058034 0.66306186] 
# mAP 0.6533792747399396 
# mIOU 0.37819473310147145