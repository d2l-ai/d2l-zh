# `d2l` API 文档
:label:`sec_d2l`

`d2l`包以下成员的实现及其定义和解释部分可在[源文件](https://github.com/d2l-ai/d2l-en/tree/master/d2l)中找到。


:begin_tab:`mxnet`
```eval_rst
.. currentmodule:: d2l.mxnet
```
:end_tab:

:begin_tab:`pytorch`
```eval_rst
.. currentmodule:: d2l.torch
```
:end_tab:

:begin_tab:`tensorflow`
```eval_rst
.. currentmodule:: d2l.torch
```
:end_tab:

:begin_tab:`paddle`
```eval_rst
.. currentmodule:: d2l.paddle
```
:end_tab:

## 模型

```eval_rst
.. autoclass:: Module
   :members:

.. autoclass:: LinearRegressionScratch
   :members:

.. autoclass:: LinearRegression
   :members:

.. autoclass:: Classification
   :members:
```

## 数据

```eval_rst
.. autoclass:: DataModule
   :members:

.. autoclass:: SyntheticRegressionData
   :members:

.. autoclass:: FashionMNIST
   :members:
```

## 训练

```eval_rst
.. autoclass:: Trainer
   :members:

.. autoclass:: SGD
   :members:
```

## 公用

```eval_rst
.. autofunction:: add_to_class

.. autofunction:: cpu

.. autofunction:: gpu

.. autofunction:: num_gpus

.. autoclass:: ProgressBoard
   :members:

.. autoclass:: HyperParameters
   :members:
```
