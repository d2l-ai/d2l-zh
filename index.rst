动手学深度学习
========================

这是一个深度学习的教学项目。我们将使用 `Apache MXNet (incubating) <https://github.com/apache/incubator-mxnet>`_ 的最新 gluon 接口来演示如何从0开始实现深度学习的各个算法。我们的将利用 `Jupyter notebook <http://jupyter.org/>`_ 能将文档，代码，公式和图形统一在一起的优势，提供一个交互式的学习体验。这个项目可以作为一本书，上课用的材料，现场演示的案例，和一个可以尽情拷贝的代码库。据我们所知，目前并没有哪个项目能既覆盖全面深度学习，又提供交互式的可执行代码。我们将尝试弥补这个空白。

Github源代码在 `https://github.com/mli/gluon-tutorials-zh <https://github.com/mli/gluon-tutorials-zh>`_

欢迎使用 `http://discuss.gluon.ai/ <http://discuss.gluon.ai/>`_ 来进行讨论

.. toctree::
   :maxdepth: 1
   :caption: 前言

   why
   preface
   install

第一部分：深度学习介绍
--------------------

.. toctree::
   :maxdepth: 1
   :caption: 预备知识

   introduction
   ndarray
   autograd

.. toctree::
   :maxdepth: 1
   :caption: 监督学习

   linear-regression-scratch
   linear-regression-gluon
   softmax-regression-scratch
   softmax-regression-gluon
   mlp-scratch
   mlp-gluon
   underfit-overfit
   reg-scratch
   reg-gluon

.. toctree::
   :maxdepth: 1
   :caption: Gluon基础

   block
   parameters
   serialization
   custom-layer
   use-gpu

.. toctree::
   :maxdepth: 1
   :caption: 卷积神经网络

   cnn-scratch
   cnn-gluon

.. toctree::
   :maxdepth: 1
   :caption: 循环神经网络

   rnn-scratch

我们将持续的加入新的内容。如果想提前了解，可以参见 `英文版本 <http://gluon.mxnet.io/>`_ （注意：中文版本根据社区的反馈做了比较大的更改，我们还在努力的将改动同步到英文版）
