# Convolutional Neural Networks
:label:`chap_cnn`

In earlier chapters, we came up against image data,
for which each example consists of a two-dimensional grid of pixels.
Depending on whether we are handling black-and-white or color images,
each pixel location might be associated with either
one or multiple numerical values, respectively.
Until now, our way of dealing with this rich structure
was deeply unsatisfying.
We simply discarded each image's spatial structure
by flattening them into one-dimensional vectors, feeding them 
through a fully-connected MLP.
Because these networks are invariant to the order
of the features,
we could get similar results
regardless of whether we preserve an order 
corresponding to the spatial structure of the pixels
or if we permute the columns of our design matrix
before fitting the MLP's parameters.
Preferably, we would leverage our prior knowledge
that nearby pixels are typically related to each other,
to build efficient models for learning from image data. 

This chapter introduces *convolutional neural networks* (CNNs),
a powerful family of neural networks
that are designed for precisely this purpose.
CNN-based architectures are now ubiquitous
in the field of computer vision, 
and have become so dominant
that hardly anyone today would develop
a commercial application or enter a competition
related to image recognition, object detection,
or semantic segmentation,
without building off of this approach.

Modern CNNs, as they are called colloquially
owe their design to inspirations from biology, group theory,
and a healthy dose of experimental tinkering.
In addition to their sample efficiency in achieving accurate models,
CNNs tend to be computationally efficient,
both because they require fewer parameters than fully-connected architectures
and because convolutions are easy to parallelize across GPU cores.
Consequently, practitioners often apply CNNs whenever possible,
and increasingly they have emerged as credible competitors
even on tasks with a one-dimensional sequence structure,
such as audio, text, and time series analysis,
where recurrent neural networks are conventionally used.
Some clever adaptations of CNNs have also brought them to bear
on graph-structured data and in recommender systems.

First, we will walk through the basic operations
that comprise the backbone of all convolutional networks.
These include the convolutional layers themselves,
nitty-gritty details including padding and stride,
the pooling layers used to aggregate information
across adjacent spatial regions,
the use of multiple channels  at each layer,
and a careful discussion of the structure of modern architectures.
We will conclude the chapter with a full working example of LeNet,
the first convolutional network successfully deployed,
long before the rise of modern deep learning.
In the next chapter, we will dive into full implementations
of some popular and comparatively recent CNN architectures
whose designs represent most of the techniques
commonly used by modern practitioners.

```toc
:maxdepth: 2

why-conv
conv-layer
padding-and-strides
channels
pooling
lenet
```

