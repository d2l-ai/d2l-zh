# Modern Convolutional Neural Networks
:label:`chap_modern_cnn`

Now that we understand the basics of wiring together CNNs, 
we will take you through a tour of modern CNN architectures.
In this chapter, each section corresponds 
to a significant CNN architecture that was 
at some point (or currently) the base model
upon which many research projects and deployed systems were built.
Each of these networks was briefly a dominant architecture 
and many were winners or runners-up in the ImageNet competition,
which has served as a barometer of progress
on supervised learning in computer vision since 2010.

These models include AlexNet, the first large-scale network deployed 
to beat conventional computer vision methods on a large-scale vision challenge;
the VGG network, which makes use of a number of repeating blocks of elements; the network in network (NiN) which convolves 
whole neural networks patch-wise over inputs; 
GoogLeNet, which uses networks with parallel concatenations;
residual networks (ResNet), which remain the most popular 
off-the-shelf architecture in computer vision;
and densely connected networks (DenseNet), 
which are expensive to compute but have set some recent benchmarks.

While the idea of *deep* neural networks is quite simple
(stack together a bunch of layers),
performance can vary wildly across architectures and hyperparameter choices.
The neural networks described in this chapter
are the product of intuition, a few mathematical insights,
and a whole lot of trial and error. 
We present these models in chronological order,
partly to convey a sense of the history
so that you can form your own intuitions 
about where the field is heading 
and perhaps develop your own architectures.
For instance,
batch normalization and residual connections described in this chapter have offered two popular ideas for training and designing deep models.

```toc
:maxdepth: 2

alexnet
vgg
nin
googlenet
batch-norm
resnet
densenet
```

