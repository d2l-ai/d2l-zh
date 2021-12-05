# `d2l` API Document
:label:`sec_d2l`

The implementations of the following members of the `d2l` package and sections where they are defined and explained can be found in the [source file](https://github.com/d2l-ai/d2l-en/tree/master/d2l).


:begin_tab:`mxnet`

```eval_rst

.. currentmodule:: d2l.mxnet

```

:end_tab:

:begin_tab:`pytorch`

```eval_rst

.. currentmodule:: d2l.torch

```

:begin_tab:`tensorflow`

```eval_rst

.. currentmodule:: d2l.torch

```

:end_tab:

## Models

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

## Data

```eval_rst 

.. autoclass:: DataModule
   :members: 

.. autoclass:: SyntheticRegressionData
   :members: 

.. autoclass:: FashionMNIST
   :members: 

```

## Trainer

```eval_rst 

.. autoclass:: Trainer
   :members: 

.. autoclass:: SGD
   :members: 

```

## Utilities

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
