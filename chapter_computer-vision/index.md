# 计算机视觉
:label:`chap_cv`

近年来，深度学习一直是提高计算机视觉系统性能的变革力量。
无论是医疗诊断、自动驾驶，还是智能滤波器、摄像头监控，许多计算机视觉领域的应用都与我们当前和未来的生活密切相关。
可以说，最先进的计算机视觉应用与深度学习几乎是不可分割的。
有鉴于此，本章将重点介绍计算机视觉领域，并探讨最近在学术界和行业中具有影响力的方法和应用。

在 :numref:`chap_cnn`和 :numref:`chap_modern_cnn`中，我们研究了计算机视觉中常用的各种卷积神经网络，并将它们应用到简单的图像分类任务中。
本章开头，我们将介绍两种可以改进模型泛化的方法，即*图像增广*和*微调*，并将它们应用于图像分类。
由于深度神经网络可以有效地表示多个层次的图像，因此这种分层表示已成功用于各种计算机视觉任务，例如*目标检测*（object detection）、*语义分割*（semantic segmentation）和*样式迁移*（style transfer）。
秉承计算机视觉中利用分层表示的关键思想，我们将从物体检测的主要组件和技术开始，继而展示如何使用*完全卷积网络*对图像进行语义分割，然后我们将解释如何使用样式迁移技术来生成像本书封面一样的图像。
最后在结束本章时，我们将本章和前几章的知识应用于两个流行的计算机视觉基准数据集。

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
