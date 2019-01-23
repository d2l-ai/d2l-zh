# 卷积神经网络

本章将介绍卷积神经网络。它是近年来深度学习能在计算机视觉领域取得突破性成果的基石。它也逐渐在被其他诸如自然语言处理、推荐系统和语音识别等领域广泛使用。我们将先描述卷积神经网络中卷积层和池化层的工作原理，并解释填充、步幅、输入通道和输出通道的含义。在掌握了这些基础知识以后，我们将探究数个具有代表性的深度卷积神经网络的设计思路。这些模型包括最早提出的AlexNet，以及后来的使用重复元素的网络（VGG）、网络中的网络（NiN）、含并行连结的网络（GoogLeNet）、残差网络（ResNet）和稠密连接网络（DenseNet）。它们中有不少在过去几年的ImageNet比赛（一个著名的计算机视觉竞赛）中大放异彩。虽然深度模型看上去只是具有很多层的神经网络，然而获得有效的深度模型并不容易。有幸的是，本章阐述的批量归一化和残差网络为训练和设计深度模型提供了两类重要思路。

```eval_rst

.. toctree::
   :maxdepth: 2

   conv-layer
   padding-and-strides
   channels
   pooling
   lenet
   alexnet
   vgg
   nin
   googlenet
   batch-norm
   resnet
   densenet
```




