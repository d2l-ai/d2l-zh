# Optimization Algorithms
:label:`chap_optimization`

If you read the book in sequence up to this point you already used a number of optimization algorithms to train deep learning models.
They were the tools that allowed us to continue updating model parameters and to minimize the value of the loss function, as evaluated on the training set. Indeed, anyone content with treating optimization as a black box device to minimize objective functions in a simple setting might well content oneself with the knowledge that there exists an array of incantations of such a procedure (with names such as "SGD" and "Adam").

To do well, however, some deeper knowledge is required.
Optimization algorithms are important for deep learning.
On one hand, training a complex deep learning model can take hours, days, or even weeks.
The performance of the optimization algorithm directly affects the model's training efficiency.
On the other hand, understanding the principles of different optimization algorithms and the role of their hyperparameters
will enable us to tune the hyperparameters in a targeted manner to improve the performance of deep learning models.

In this chapter, we explore common deep learning optimization algorithms in depth.
Almost all optimization problems arising in deep learning are *nonconvex*.
Nonetheless, the design and analysis of algorithms in the context of *convex* problems have proven to be very instructive.
It is for that reason that this chapter includes a primer on convex optimization and the proof for a very simple stochastic gradient descent algorithm on a convex objective function.

```toc
:maxdepth: 2

optimization-intro
convexity
gd
sgd
minibatch-sgd
momentum
adagrad
rmsprop
adadelta
adam
lr-scheduler
```

