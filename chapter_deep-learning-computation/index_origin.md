# Deep Learning Computation
:label:`chap_computation`

Alongside giant datasets and powerful hardware,
great software tools have played an indispensable role
in the rapid progress of deep learning.
Starting with the pathbreaking Theano library released in 2007,
flexible open-source tools have enabled researchers
to rapidly prototype models, avoiding repetitive work
when recycling standard components
while still maintaining the ability to make low-level modifications.
Over time, deep learning's libraries have evolved
to offer increasingly coarse abstractions.
Just as semiconductor designers went from specifying transistors
to logical circuits to writing code,
neural networks researchers have moved from thinking about
the behavior of individual artificial neurons
to conceiving of networks in terms of whole layers,
and now often design architectures with far coarser *blocks* in mind.


So far, we have introduced some basic machine learning concepts,
ramping up to fully-functional deep learning models.
In the last chapter,
we implemented each component of an MLP from scratch
and even showed how to leverage high-level APIs
to roll out the same models effortlessly.
To get you that far that fast, we *called upon* the libraries,
but skipped over more advanced details about *how they work*.
In this chapter, we will peel back the curtain,
digging deeper into the key components of deep learning computation,
namely model construction, parameter access and initialization,
designing custom layers and blocks, reading and writing models to disk,
and leveraging GPUs to achieve dramatic speedups.
These insights will move you from *end user* to *power user*,
giving you the tools needed to reap the benefits
of a mature deep learning library while retaining the flexibility
to implement more complex models, including those you invent yourself!
While this chapter does not introduce any new models or datasets,
the advanced modeling chapters that follow rely heavily on these techniques.

```toc
:maxdepth: 2

model-construction
parameters
deferred-init
custom-layer
read-write
use-gpu
```

