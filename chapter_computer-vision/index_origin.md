# Computer Vision
:label:`chap_cv`

Whether it is medical diagnosis, self-driving vehicles, camera monitoring, or smart filters, many applications in the field of computer vision are closely related to our current and future lives. 
In recent years, deep learning has been
the transformative power for advancing the performance of computer vision systems.
It can be said that the most advanced computer vision applications are almost inseparable from deep learning.
In view of this, this chapter will focus on the field of computer vision, and investigate methods and applications that have recently been influential in academia and industry.


In :numref:`chap_cnn` and :numref:`chap_modern_cnn`, we studied various convolutional neural networks that are
commonly used in computer vision, and applied them
to simple image classification tasks. 
At the beginning of this chapter, we will describe
two methods that 
may improve model generalization, namely *image augmentation* and *fine-tuning*,
and apply them to image classification. 
Since deep neural networks can effectively represent images in multiple levels, 
such layerwise representations have been successfully 
used in various computer vision tasks such as *object detection*, *semantic segmentation*, and *style transfer*. 
Following the key idea of leveraging layerwise representations in computer vision,
we will begin with major components and techniques for object detection. Next, we will show how to use *fully convolutional networks* for semantic segmentation of images. Then we will explain how to use style transfer techniques to generate images like the cover of this book.
In the end, we conclude this chapter
by applying the materials of this chapter and several previous chapters on two popular computer vision benchmark datasets.

```toc
:maxdepth: 2

image-augmentation
fine-tuning
bounding-box
anchor
multiscale-object-detection
object-detection-dataset
ssd
rcnn
semantic-segmentation-and-dataset
transposed-conv
fcn
neural-style
kaggle-cifar10
kaggle-dog
```

