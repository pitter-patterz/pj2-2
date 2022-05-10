
from __future__ import  absolute_import
from tqdm import tqdm
from utils.config import opt
from data.dataset import Dataset, TestDataset, inverse_normalize
from model import FasterRCNNVGG16
from torch.utils import data as data_
from trainer import FasterRCNNTrainer
from utils import array_tool as at
from eval_new import eval_      

import torch


dataset = Dataset(opt)

print('\nPrepare data...')
dataloader = data_.DataLoader(dataset,batch_size=1,shuffle=True,num_workers=opt.num_workers)
testset = TestDataset(opt)
test_dataloader = data_.DataLoader(testset,batch_size=1,num_workers=opt.test_num_workers,shuffle=False,pin_memory=True)

print('\nLoad the pretrained VGG16...')
faster_rcnn = FasterRCNNVGG16()
print('\nSuccessfully implement VGG16.')

trainer = FasterRCNNTrainer(faster_rcnn).cuda()
best_map = 0
lr_ = opt.lr


iteration = 0
opt.test_num = 200

for epoch in range(opt.epoch):
    
    print('\n-----epoch:',epoch)
    trainer.reset_meters()

    record = []

    # record_train: loss & iteration; record_test: mAP & mIOU & total_loss & iteration.

    for ii, (img, bbox_, label_, scale) in tqdm(enumerate(dataloader)):
        
        print('\nIteration:',iteration)
        iteration += 1

        scale = at.scalar(scale)
        img, bbox, label = img.cuda().float(), bbox_.cuda(), label_.cuda()
        trainer.train_step(img, bbox, label, scale)

        # compute loss on the batch (one image)
        loss = trainer.get_meter_data()
        print(loss)
        
        v = list(loss.values())
        v.append(iteration)
        record.append(v)
        
    with torch.no_grad():

        print('\nBegin test...The test sample size is',opt.test_num)

        eval_result, iou_result, loss = eval_(test_dataloader,faster_rcnn,trainer,test_num=opt.test_num)
      
        print('\nAP\n',eval_result)
        print('\nIOU\n',iou_result)
        print('\nLoss\n',loss)

        record.append([eval_result['map'],iou_result,loss,iteration])
        print('\nSave the train records...')
        file = open('record_epoch'+str(epoch)+'.txt','w')
        
        for line in record:
          line = [str(v) for v in line]
          a = ','.join(line)
          file.write(a+'\n')
        file.close()
    
        if eval_result['map'] > best_map:
          best_map = eval_result['map']

# Save the net.
# trainer.save(best_map=best_map) 


# Use tensorboardX to make plots.

# import numpy as np,os
# from tensorboardX import SummaryWriter

# cols1 = ['rpn_loc_loss','rpn_cls_loss','roi_loc_loss','roi_cls_loss','total_loss', 'iteration']

# cols2 = ['mAP','mIOU','total_loss','iteration']

# writer = SummaryWriter('runs')
# path = r'./train_record/'


# for f in os.listdir(path):
#     if f.find('epoch') < 0:
#         continue
#     for line in open(path+f,'r'):
#         line = line.replace('\n','')
#         line = line.split(',')
#         line = [float(l) for l in line]
        
#         if len(line)==4:
#             for i in range(4-1):
#                 writer.add_scalar('test: '+cols2[i],line[i],line[-1])
                
#         if len(line)==6:
#             for i in range(6-1):
#                 writer.add_scalar('train: '+cols1[i],line[i],line[-1])  

# writer.close()
     





