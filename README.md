# Introduction
This is the second taks of *project 2*. We use Faster-RCNN to do object detection on VOC2007 dataset. This project has referrence to https://github.com/chenyuntc/simple-faster-rcnn-pytorch.


*eval_new.py*: We define functions to compute evaluation metrics of a model on the test data, including total loss, AP, mAP and mIOU.

*train.py*: Train a Faster-RCNN within ten epochs. We use the same hyper parameters (learning rate, batch and the optimizer) as *chenyuntc*. 

*user_test.py*: For a given image (jpg, png or other formats) and a trained model, do object detection and visualize the proposal boxes returned by RPN.
