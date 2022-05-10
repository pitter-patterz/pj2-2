# Introduction
This is the second taks of *project 2*. We use Faster-RCNN to do object detection on VOC2007 dataset. This project has referrence to https://github.com/chenyuntc/simple-faster-rcnn-pytorch.


*eval_new.py*: We define functions to compute evaluation metrics of a model on the test data, including total loss, AP, mAP and mIOU.

*train.py*: Train a Faster-RCNN within ten epochs. We use the same hyper parameters (learning rate, batch and the optimizer) as *chenyuntc*. 

*test.py*: Compute the AP, mAP and mIOU of a trained net on the VOC2007 test dataset.

*user_test.py*: For a given image and a trained model, do object detection and visualize the proposal boxes returned by RPN.

Besides, we replace Visdom by TensorBoardX to visualize loss, mAP and etc (see *train_record*). Our model can be downloaded from https://pan.baidu.com/s/1dj9UsXYNCsx5PajrDzgayw?pwd=sjwl (pwd: sjwl).

# Usage

First check the packages via requirements.txt. Then start training.

+ python train.py

The trained net is automatically saved as a .pth file. To test the net,

+ python test.py

note that the default test sample size is 200 due to the limit of memory. One can change it via the parameter *test_num* in *eval_()*. The results in our report are obtained by dividing the whold test dataset into five small batches.

