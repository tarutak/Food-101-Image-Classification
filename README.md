Food-101
==============================

- [Problem Statement](#problem-statement)
  - [Previous SoTA Results](#previous-sota-results)
  - [References](#references)
- [Introduction](#introduction)
- [Fastai Approach](#fastai-approach)
- [Conclusion and Result](#conclusion-and-result)
- [Improvements](#improvements)
- [Test Images Results](#test-images-results)


## Problem Statement

[Food-101](https://www.vision.ee.ethz.ch/datasets_extra/food-101/) is a challenging vision problem, but everyone can relate to it. Recent SoTA is ~80% top-1, 90% top-5.  These approaches rely on lots of TTA, large networks and  even novel architectures.

Train a decent model >85% accuracy for top-1 for the test set, using a ResNet50 or smaller network with a reasonable set of augmentations. 

![sample image from dataset](sample/food-101.jpg "Food-101 sample images")


---

### Previous SoTA Results

Following is a comparison of the previous SoTA classification results for the Food-101 dataset.

| Model                    |  Augmentations           |  Epochs  |  Top-1 Accuracy  % |  Top-5 Accuracy %  |
| ------------------------|----------------------------------| --------------|------------------------------|------------------------------ |
| InceptionV3      | Flip, Rotation, Color, Zoom | 32   |                 88.28           |            96.88                 |
|WISeR                    | Flip, Rotation, Color, Zoom |  ~ 32   |               90.27    |           98.71                   |
| ResNet+fastai   | Optional Transformations |  16   |                 90.52           |            98.34                 |


---

### References

[1] **Inception V3 Approach** Hassannejad, Hamid, et al. [Food image recognition using very deep convolutional networks](https://dl.acm.org/citation.cfm?id=2986042). Proceedings of the 2nd International Workshop on Multimedia Assisted Dietary Management . ACM, 2016.

[2 ] **WISeR Approach** Martinel, Niki, Gian Luca Foresti, and Christian Micheloni. [Wide-slice residual networks for food recognition](https://arxiv.org/pdf/1612.06543.pdf) . Applications of Computer Vision (WACV), 2018 IEEE Winter Conference on . IEEE, 2018.

[3] **ResNet + fastai Approach** [platform.ai](https://platform.ai/blog/page/3/new-food-101-sota-with-fastai-and-platform-ais-fast-augmentation-search/) 


---

I plan to tackle this problem using,

[Fastai](https://docs.fast.ai/)

I have done a few other projects in Fastai and I am more comfortable with its data block api and image augmentations setup is easy. The Platform.ai group who has achieved SoTA also utilizes the same library and demonstrate that we can achieve decent results with ResNets quickly with minimal manual augmentations.

 ---

 ## Introduction

 Our objective is to classify 101,000 food images in 101 categories.

This is very so ImageNet like where we had 1.2  million images to classify into 1000 categories, we have observed that *CNN* are the goto models for such image classification tasks.

---

## Fastai Approach

The Fastai Approach typically follows the following approach for Image Classification tasks:
- Get the data from its source and understand how the data is structured for the problem, i.e., Labels, Train set , valid set, etc
- Use the datablock API of Fastai to make a data bunch by defining how to load the data, how to label the data, how to split it and which types of transformations to apply for augmentations
- Smart image augmentation is key to getting better generalization and making the most out of available data.
- Next we proceed to create a learner object wherein we define the type of Architecture, Data, Loss functions, metrics ,etc. We always start with a pretrained model because low-mid level to feature representations will always be similar for any model.
- In Fastai, The single most important parameter we always tweak is Learning rate. Learning Rate finder is a function which allows us to learn what is the largest LR we can use to train our model without causing our loss to diverge. We can get a plot of Learning rate vs Losses, which helps us decide the optimal LR. We can also tune Weight Decay but that should be secondary.
- Fastai course, Practical Deep learning for coders, introduces various concepts like Cyclical Learning rate, Super Convergence, Progressive resizing, discriminative LR for different layer groups.
- Typical training flow of a model is to train on images for smaller size, inititally with all layers except the last one frozen and then tuning all the layers unfreezed. We move on to repeat the same steps with bigger image size and smaller batch size.
- The library allows us to set different learning rate of early layers, mid layers and last layers of the model which helps to effectively trian the models.
- We have callbacks in Fastai library which allows us to monitor Loss and evaluation metrics as we train the model.
- We use a special ensemble prediction technique called Test Time Augmentations or TTA for final test set predictions. For each image in our test set, we get 8 transformed images on which our model makes predictions and we weight all the prediction to make our final answer.
- Fastai has powerful inference functions which allows us to understand and unravel the specifics of how our model is making a prediction, what are our top losses,etc.


**Steps**
I trained three models
- [1] ResNet 18
- [2] ResNet 50
- [3] ResNet 50 with specific transformations.


What result do we obtain after going through all this? Let's have a look

All results obtained are using Google Cloud Platforms, Nvidia T4 GPU. I have also utilized half precision training which could have resulted in faster training.

*ResNet 18*
|  Phase                       |   Time Taken (hrs)          |  Epochs  |  Top-1 Accuracy  % |  Top-5 Accuracy %  |
| ------------------------     |----------------------------------| --------------|------------------------------|------------------------------ |
| Train on 192 size images(Freeze+ Unfreeze)  |  1.5 | 17   |                 76           |            -                |
|  Train on 384 size images(Freeze+ Unfreeze) | 2.1  |  10   |               82.5    |           -                   |
|  Train on 512 size images(Freeze+ Unfreeze)  | 2.8 |  8  |                 83          |            96.46                 |
 *TTA Final= 83.48%*, *Total Epochs=35*, *Total time=6.4 Hours* 


*ResNet 50- 3 Stage*
|  Phase                       |   Time Taken (hrs)          |  Epochs  |  Top-1 Accuracy  % |  Top-5 Accuracy %  |
| ------------------------     |----------------------------------| --------------|------------------------------|------------------------------ |
| Train on 192 size images(Freeze+ Unfreeze)  |  2.1 | 13   |                 80.5           |            95.46               |
|  Train on 384 size images(Freeze+ Unfreeze) | 4.3  |  8   |               85.38    |           97.29                   |
|  Train on 512 size images(Freeze+ Unfreeze)  | 4.5 |  8  |                 85.76          |            97.34                 |
 *TTA Final= 86.02%*, *Total Epochs=29*, *Total time=11.04 Hours* 


*ResNet 50- 2 Stage*
|  Phase                       |   Time Taken (hrs)          |  Epochs  |  Top-1 Accuracy  % |  Top-5 Accuracy %  |
| ------------------------     |----------------------------------| --------------|------------------------------|------------------------------ |
| Train on 224 size images(Freeze+ Unfreeze)  |  2.5 | 14   |                 83.8           |            96.54               |
|  Train on 512 size images(Freeze+ Unfreeze)  | 6.9 |  12  |                 86.66          |            97.45                 |
 *TTA Final= 87.08%*, *Total Epochs=26*, *Total time=11.04 Hours* 
 
---

**Conclusion**
 - ResNet-50(2 stage) helped me achieve Top-1 Accuracy of 87.08 and Top-5 Accuracy of 97.45 in 25 Epochs, where as the current SoTA on Food-101 dataset is Top-1 Accuracy of 90.52 and Top-5 Accuracy of 98.34 in *16* Epochs.
 - The dataset has a few categories like steak & fillet Minion, chocolate cake & chocolate mousse, Icecream & yogurt which result in the maximum loss. If we dig deeper we realise that it is difficult for humans to make those differences. 
 - ResNet 18 also did a pretty decent job reaching upto 80% accuracy mark but starts to struggle after that, ResNet 50 quickly surpasses ResNet18 but training becomes incredibly slow after 85% mark.
 - I tried to achieve higher accuracy while keeping the number of epochs lesser. We could get higher metrics by prolonged training but Platform.ai's article highlights how they were able to train a less complex model very efficiently.
---

**Improvements**

Results can further be improved
- Exploring optimal transformation for categories which contributes to majority misclassifications.
- Faster GPU will definitely help for quicker expermentations which may result in better results.
