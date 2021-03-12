# Active Learning on Computer Vision 
Active Learning is a semi-supervised method that allows you to label less data by selecting the most important samples from the learning process. It helps you select a subset of images from a large unlabeled pool of data in such a way that obtaining annotations of those images will result in a maximal increase of model accuracy. I use Mnist Data for evaluation here. 

## Getting Started

### Prerequisites
* Keras 2.4.3
* Tensorflow 2.4.2
* Opencv for python

### Methods to Compare
* Random  
* Confidence Score 
* Entropy
* Margin Sampling

## Usage via command line

python activelearning_main.py


## Result
Above figure provides experimental results on mnist dataset. Each chart is the mean of 3 runs with different random seeds. Even with the averaging, the result is still a  little bit sensitive when the number of samples is small. But overall, we can conclude that active learning outperformans random, margin sampling and least confidence outperformances random by a margin of 4.0%-4.5%. Another way is to look at its application: if your goal is 91.6% accuracy, you need 2064 samples to train with random and 1056 samples to train with least score, which reduces the needs of the sample for 50%.  
Then I ran some of the rest of unlabeled data through model evaluation and took some of most confident samples to train the model(people it auto-labeling). I got extra 0.2-0.3% of accuracy. 




## References
* https://towardsdatascience.com/active-learning-on-mnist-saving-on-labeling-f3971994c7ba
* https://towardsdatascience.com/active-learning-for-classification-models-de4dc9c3f612
* https://blog.superannotate.com/active-learning-semantic-segmentation
* https://hachmannlab.github.io/chemml/active_model_based.html
* https://github.com/neuropoly/deep-active-learning
* https://github.com/dhaalves/CEAL_keras

