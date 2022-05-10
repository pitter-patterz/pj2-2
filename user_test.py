

import torch as t,numpy as np
from utils.config import opt
from model import FasterRCNNVGG16
from trainer import FasterRCNNTrainer
from data.util import  read_image
from utils.vis_tool import vis_bbox
from utils import array_tool as at
from data.dataset import preprocess

import time


print('\n...Load the trained faster-RCNN...')
faster_rcnn = FasterRCNNVGG16()
trainer = FasterRCNNTrainer(faster_rcnn).cuda()
trainer.load(r'.\user\fasterrcnn_10epoch.pth')
opt.caffe_pretrain=False

print('\n...Read images...')
img = read_image(r'.\user\demo1.jpeg')
img = t.from_numpy(img)[None]

trainer.faster_rcnn.rpn.return_proposals = True
_bboxes, _labels, _scores = trainer.faster_rcnn.predict(img,visualize=True)

p1 = vis_bbox(at.tonumpy(img[0]),
         at.tonumpy(_bboxes[0]),
         at.tonumpy(_labels[0]).reshape(-1),
         at.tonumpy(_scores[0]).reshape(-1))

ct1 = str(int(time.time()))
pic1 = p1.get_figure()
pic1.savefig(r'.\user\detect_'+ct1+'.png',dpi=300)


rois = np.loadtxt(r'.\user\rois.txt')
x = img[0] 
size = x.shape[1:]

x = preprocess(at.tonumpy(x))
x = at.totensor(x[None]).float()
scale = x.shape[3] / size[1]

p2 = vis_bbox(at.tonumpy(img[0]),rois[:5]/scale)
ct2 = str(int(time.time()))
pic2 = p2.get_figure()
pic2.savefig(r'.\user\visrpn_'+ct2+'.png',dpi=300)

print('\nTest finished. Two pictures are saved.\n')
print('detect_'+ct1+'.png\n')
print('visrpn_'+ct2+'.png')




