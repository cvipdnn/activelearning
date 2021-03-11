# Active Learning on Computer Vision 
Active Learning is a semi-supervised method that allows you to label less data by selecting the most important samples from the learning process. It helps you select a subset of images from a large unlabeled pool of data in such a way that obtaining annotations of those images will result in a maximal increase of model accuracy. I use Mnist Data for evaluation here. 

## Getting Started

### Prerequisites
* Keras 2.4.3
* Tensorflow 2.4.2
* Opencv for python

### Methods to Compare
* Random Labeling (random_sample.py)
* Confidence Score (confidence_sample.py)
* Entropy (entropy_sample.py)

## Usage via command line

python act_lr.py





## References
* https://towardsdatascience.com/active-learning-on-mnist-saving-on-labeling-f3971994c7ba
* https://towardsdatascience.com/active-learning-for-classification-models-de4dc9c3f612
* https://blog.superannotate.com/active-learning-semantic-segmentation
* https://hachmannlab.github.io/chemml/active_model_based.html
* https://github.com/neuropoly/deep-active-learning
* https://github.com/dhaalves/CEAL_keras

