import torch,itertools,six
import numpy as np

from tqdm import tqdm
from utils.eval_tool import eval_detection_voc


def eval_(dataloader, faster_rcnn, trainer, test_num=100, start=0):
    

    pred_bboxes, pred_labels, pred_scores = list(), list(), list()
    gt_bboxes, gt_labels, gt_difficults = list(), list(), list()
    total_loss = 0
    
    for ii, (imgs, sizes, gt_bboxes_, gt_labels_, gt_difficults_, scale_) in tqdm(enumerate(dataloader)):
        
        if ii < start:
            continue
        if ii == start+test_num:
            break
        

        sizes = [sizes[0][0].item(), sizes[1][0].item()]
        pred_bboxes_, pred_labels_, pred_scores_ = faster_rcnn.predict(imgs, [sizes])
        gt_bboxes += list(gt_bboxes_.numpy())
        gt_labels += list(gt_labels_.numpy())
        gt_difficults += list(gt_difficults_.numpy())
        pred_bboxes += pred_bboxes_
        pred_labels += pred_labels_
        pred_scores += pred_scores_
        
        with torch.no_grad():
          img = imgs.cuda()
          bbox = gt_bboxes_.cuda()
          label = gt_labels_.cuda()
          total_loss += trainer.forward(img,bbox,label,scale_.item())[-1]

    ap_result = eval_detection_voc(pred_bboxes, pred_labels,pred_scores,gt_bboxes,gt_labels,gt_difficults,use_07_metric=True)
    iou_result = calc_iou(pred_bboxes,pred_labels,pred_scores,gt_bboxes,gt_labels,gt_difficults)

    return ap_result,iou_result,total_loss.item()/test_num



def combined_iou(pred_bbox,gt_bbox):
    
  h = max(np.max(pred_bbox),np.max(gt_bbox))
  h = int(h)
  pred_map,gt_map = np.zeros(shape=(h,h)),np.zeros(shape=(h,h))

  for i in range(pred_bbox.shape[0]):
    ymin,xmin,ymax,xmax = pred_bbox[i,0],pred_bbox[i,1],pred_bbox[i,2],pred_bbox[i,3]
    ymin,xmin,ymax,xmax = int(ymin),int(xmin),int(ymax),int(xmax)
    pred_map[xmin:xmax+1,ymin:ymax+1] = 1

  for i in range(gt_bbox.shape[0]):
    ymin,xmin,ymax,xmax = gt_bbox[i,0],gt_bbox[i,1],gt_bbox[i,2],gt_bbox[i,3]
    ymin,xmin,ymax,xmax = int(ymin),int(xmin),int(ymax),int(xmax)
    gt_map[xmin:xmax+1,ymin:ymax+1] = 1
  
  inter = np.sum(pred_map*gt_map)
  union = np.sum(pred_map+gt_map>=1)

  if union==0:
    print('\n\n\n...The union equals zero,please check...\n\n\n')

  return inter/(union+1e-6)


def calc_iou(pred_bboxes,pred_labels,pred_scores,gt_bboxes,gt_labels,gt_difficults):
    
    pred_bboxes = iter(pred_bboxes)
    pred_labels = iter(pred_labels)
    pred_scores = iter(pred_scores)
    gt_bboxes = iter(gt_bboxes)
    gt_labels = iter(gt_labels)

    IOUS = []

    for pred_bbox, pred_label, pred_score, gt_bbox, gt_label in six.moves.zip(pred_bboxes, pred_labels, pred_scores, gt_bboxes, gt_labels):
        
        IOU = []
        for l in np.unique(np.concatenate((pred_label, gt_label)).astype(int)):

            pred_mask_l = pred_label == l
            pred_bbox_l = pred_bbox[pred_mask_l]
            pred_score_l = pred_score[pred_mask_l]
            
            # sort by score
            order = pred_score_l.argsort()[::-1]
            pred_bbox_l = pred_bbox_l[order]
            pred_score_l = pred_score_l[order]

            gt_mask_l = gt_label == l
            gt_bbox_l = gt_bbox[gt_mask_l]


            if len(pred_bbox_l)==0 or len(gt_bbox_l)==0:
                IOU.append(0.0)
                continue
            
            pred_bbox_l = pred_bbox_l.copy()
            pred_bbox_l[:, 2:] += 1
            gt_bbox_l = gt_bbox_l.copy()
            gt_bbox_l[:, 2:] += 1

            iou = combined_iou(pred_bbox_l, gt_bbox_l)
            IOU.append(iou)
            
        IOUS.append(sum(IOU)/len(IOU))
        
    return sum(IOUS)/len(IOUS)         