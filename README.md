# Mask R-CNN for Bolt Dataset and Historical Document Image Dataset
This is an implementation of instance segmentation for two challenging datasets: (1) bold dataset and (2) historical document image dataset based on [Mask R-CNN](https://arxiv.org/abs/1703.06870) on Python 3, Keras, and TensorFlow. 

# Requirements
To rerproduce our result, please consider whether your machine is installed with the following requirements:
* Python 3.4
* TensorFlow 1.3
* Keras 2.0.8 
* Other common packages listed in `requirements.txt`.

# Usage
To train a model, please run 
python [deep_voronoi or bolts] train --dataset=/path/to/dataset/ --model=[imagenet or coco]

To evaluate and explore the model, please run the following jupyter notebook:
Pack_Chulwoo_Lee_Jiyoung_Project.ipynb
