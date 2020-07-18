# Multilayer Perceptrons
:label:`chap_perceptrons`

In this chapter, we will introduce your first truly *deep* network.
The simplest deep networks are called multilayer perceptrons,
and they consist of multiple layers of neurons
each fully connected to those in the layer below
(from which they receive input)
and those above (which they, in turn, influence).
When we train high-capacity models we run the risk of overfitting.
Thus, we will need to provide your first rigorous introduction
to the notions of overfitting, underfitting, and model selection.
To help you combat these problems,
we will introduce regularization techniques such as weight decay and dropout.
We will also discuss issues relating to numerical stability and parameter initialization
that are key to successfully training deep networks.
Throughout, we aim to give you a firm grasp not just of the concepts
but also of the practice of using deep networks.
At the end of this chapter,
we apply what we have introduced so far to a real case: house price prediction.
We punt matters relating to the computational performance,
scalability, and efficiency of our models to subsequent chapters.

```toc
:maxdepth: 2

mlp
mlp-scratch
mlp-concise
underfit-overfit
weight-decay
dropout
backprop
numerical-stability-and-init
environment
kaggle-house-price
```

