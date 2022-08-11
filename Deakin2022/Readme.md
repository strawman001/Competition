# Deakin2022 Simpons Challenge

VQA Task

Tensorflow Implement

Based on Bottom-up and Top-down attention
*P. Anderson et al., ‘Bottom-Up and Top-Down Attention for Image Captioning and Visual Question Answering’*(https://arxiv.org/abs/1707.07998) 

# Results(#3)
![image](https://user-images.githubusercontent.com/39436745/183824428-02a66f40-30ba-4cd4-8f45-c01e9712985c.png)

# Implement Details
![image](https://user-images.githubusercontent.com/39436745/183826108-198e9ef4-6fb1-49b0-b92d-ef0cb36705b2.png)
## Visual Detection Head (Bottom-up)

The object detection based on Faster-RCNN by Google(https://tfhub.dev/google/faster_rcnn/openimages_v4/inception_resnet_v2/1)

![image](https://user-images.githubusercontent.com/39436745/183823849-8636152c-2f4a-4f78-8362-677addbdcc1b.png)

For limitation of computation resources, we also provide a simplified visual head, which just split single image to 9 parts as visual features

## Train Data

We use COCO VQA Dataset as train dataset for our model cannot converge on abstract scene dataset.

Object Detection Head:                             
https://nlp.cs.unc.edu/data/lxmert_data/mscoco_imgfeat/train2014_obj36.zip (36 object features)        

Simplified Head:                        
http://images.cocodataset.org/zips/train2014.zip

# Evluation and Improvement
## Evluation
Actually, our solution did not achieve our expectation. The performance showed the model may not learn too much useful features, and still randomly guess.
Through the analysis of the results and model, we though reasons are following:

**1. Gaps between train dataset and test dataset**

Simpons images are very different from natural images, the visual gaps cause bad performance
 
**2. We use simplified visual head**

Object detection head would spend too much computation resources, so we just used simplified head in the competition

## Improvement
If we try to get better performance, we will collect simpons images and make same distribution dataset. Then use the dataset to produce visual features. 
Surely, the BuTd method may not the advancest method, the VQA task has been a part of multimodality. As developing of transformer attention mechanism 
and pretrained large model, there are many available solutions that we can reference and apply in this task.


# Reference
https://github.com/hengyuan-hu/bottom-up-attention-vqa                                        
https://github.com/SatyamGaba/visual_question_answering
