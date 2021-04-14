# CS4240 Reproducibility Project: ImageNet-trained CNNs are biased towards texture

## Authors

 - Just van Stam
 - Daniël van Gelder ([d.vangelder-1@student.tudelft.nl](d.vangelder-1@student.tudelft.nl))

## Introduction

The paper by Geirhos et al. \[1\] provides a novel perspective on the performance on ImageNet-trained CNNs. They argue that while it is commonly thought that Deep CNNs recognise objects by learning abstract representations of their shape, they rather learn to recognize objects by their texture(s). This is in contrast to how humans recognize objects, which is primarily by shape. 

Through a comparative study, the authors demonstrate that many state-of-the-art CNN models (e.g. ResNet, AlexNet, GoogleNet, etc.) trained om ImageNet base their predictions on textures while humans base their respective predictions on shape. This result is achieved by performing a "style transfer" on some of the images in ImageNet. The style transfer is performed by extracting the texture of a _style_ image and applying that on a _content_ image. See the figure below for an example provided by the authors:

![Style transfer example](img/style-transfer-example.png)
_Figure 1: Style transfer example where the leftmost image shows a picture of elephant skin used as the style image, the middle picture shows a picture of a cat used as the content image and the rightmost picture showing the style transfered result._

When a model predicts the class of the image as its texture's class rather than its content's class, this is called a "cue conflict".

After demonstrating that ImageNet trained CNNs are biased towards texture, the authors propose a solution to this phenomenon. A novel dataset is proposed built on top of ImageNet which applies style transfers to all images in ImageNet. This dataset is called "Stylized-Imagenet". 

Using the AdaIN style transfer approach by Huang and Belongie \[2\], images from ImageNet are converted to stylized images using random paintings (in this case, the [Kaggle Painter By Numbers Dataset](https://www.kaggle.com/c/painter-by-numbers/data)). The authors hypothesize that when models like ResNet-50 are trained on Stylized ImageNet, the frequency of cue conflicts is significantly reduced.

![Stylized ImageNet Examples](img/stylized-imagenet.png)
_Figure 2: Style transfer examples where the content image is the picture on the left and its corresponding stylized versions are the ten images to the right._

The results show that when ResNet-50 is trained on the Stylized-ImageNet the cue conflicts dramatically decrease. However, performance on the original ImageNet validation set is somewhat reduced. To combat this, the authors propose a model called Shape-ResNet which is trained on both ImageNet and Stylized-ImageNet and consequently fine-tuned on ImageNet again. This final model achieves a better score on the ImageNet task than the original ResNet-50 model.


## Paper Results 
<!-- Just -->

## Replication
<!-- Just -->

## Additional Dataset
<!-- Daniel -->
Besides replicating figure 5 using the pre-trained weights provided by the authors, we wanted to replicate it by retraining the model on the Stylized-ImageNet dataset. However, due to the licensing of the ImageNet dataset, the authors were not allowed to share the dataset used. Therefore, we had to create it on our own. However, due to the sheer size of the ImageNet dataset and the limited hardware available for the style transfers, we opted to recreate a downsampled version of Stylized-ImageNet with a resolution of 64x64. It turned out to not be possible within the limit of the budget to train the ResNet-50 model on this dataset, therefore we opted to only  create the dataset.

Creating this dataset was non-trivial as it requires sophisticated compute hardware. We initially attempted to run this on a Google Cloud VM could not create an instance with a fast enough VM to create the dataset in a reasonable amount of time. Thus, we tried to create the dataset in Colab Notebooks using Colab Pro. This turned out to be possible. We created the dataset in 10 batches taking approximately ~2.5 hours per batch. 



## Conclusion
<!-- Daniel -->
<!-- Future work -->
With this reproduction we have replicated a portion of the results in the paper by Geirhos et al. Our work indicates that ImageNet trained CNNs indeed are biased towards textures. As future work, we propose that, using the downsampled dataset that was created with this project, the ResNet-50 model is re-trained using that dataset to further confirm the results put forward by the authors. 

## References
- \[1\]: Geirhos, R., Rubisch, P., Michaelis, C., Bethge, M., Wichmann, F. A., & Brendel, W. (2018). ImageNet-trained CNNs are biased towards texture; increasing shape bias improves accuracy and robustness. _arXiv preprint arXiv:1811.12231_

- \[2\]: Huang, X., & Belongie, S. (2017). Arbitrary style transfer in real-time with adaptive instance normalization. In _Proceedings of the IEEE International Conference on Computer Vision_ (pp. 1501-1510).
Chicago	


# Using this Repository

## Replicating the figures
In order to create the data to create the figures from this report, run the script `code/evaluate_models.pipeline.py` with the following arguments:

1. model name
2. stimuli directory (in our case: `stimuli/style-transfer-preprocessed-512`) 
3. output csv file name
4. if using stylized pretrained models, add `-s`

Create the figures in the notebook `code/ReproduceFigures.ipynb`

## Creating the stylized dataset

The code to create the stylized dataset can be found in the repository by the original authors: 

https://github.com/bethgelab/stylize-datasets