# `d2l` API 文档
:label:`sec_d2l`

`d2l` 软件包中以下成员的实现以及在其中定义和解释这些成员的部分可以在 [source file](https://github.com/d2l-ai/d2l-en/tree/master/d2l) 中找到。

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

## 資料

```eval_rst 
.. autoclass:: DataModule
   :members: 

.. autoclass:: SyntheticRegressionData
   :members: 

.. autoclass:: FashionMNIST
   :members:
```

## 培训师

```eval_rst 
.. autoclass:: Trainer
   :members: 

.. autoclass:: SGD
   :members:
```

## 实用

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
