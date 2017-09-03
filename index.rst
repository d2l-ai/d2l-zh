动手学深度学习
========================

这是一个深度学习的教学项目。我们将使用 `Apache MXNet (incubating) <https://github.com/apache/incubator-mxnet>`_ 的最新 gluon 接口来演示如何从0开始实现深度学习的各个算法。我们的将利用 `Jupyter notebook <http://jupyter.org/>`_ 能将文档，代码，公式和图形统一在一起的优势，提供一个交互式的学习体验。这个项目可以作为一本书，上课用的材料，现场演示的案例，和一个可以尽情拷贝的代码库。据我们所知，目前并没有哪个项目能既覆盖全面深度学习，又提供交互式的可执行代码。我们将尝试弥补这个空白。

另外一个独特的地方在于它的编写流程。我们公开整个项目的方方面面，并且陈诺一直免费。虽然它主要由几个人撰写，但我们非常欢迎大家来一起写作和讨论。


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

.. toctree::
   :maxdepth: 1
   :caption: 神经网络

   mlp-scratch
   mlp-gluon
