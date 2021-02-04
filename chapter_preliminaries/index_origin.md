#  Preliminaries
:label:`chap_preliminaries`

To get started with deep learning,
we will need to develop a few basic skills.
All machine learning is concerned
with extracting information from data.
So we will begin by learning the practical skills
for storing, manipulating, and preprocessing data.

Moreover, machine learning typically requires
working with large datasets, which we can think of as tables,
where the rows correspond to examples
and the columns correspond to attributes.
Linear algebra gives us a powerful set of techniques
for working with tabular data.
We will not go too far into the weeds but rather focus on the basic
of matrix operations and their implementation.

Additionally, deep learning is all about optimization.
We have a model with some parameters and
we want to find those that fit our data *the best*.
Determining which way to move each parameter at each step of an algorithm
requires a little bit of calculus, which will be briefly introduced.
Fortunately, the `autograd` package automatically computes differentiation for us,
and we will cover it next.

Next, machine learning is concerned with making predictions:
what is the likely value of some unknown attribute,
given the information that we observe?
To reason rigorously under uncertainty
we will need to invoke the language of probability.

In the end, the official documentation provides
plenty of descriptions and examples that are beyond this book.
To conclude the chapter, we will show you how to look up documentation for
the needed information.

This book has kept the mathematical content to the minimum necessary
to get a proper understanding of deep learning.
However, it does not mean that
this book is mathematics free.
Thus, this chapter provides a rapid introduction to
basic and frequently-used mathematics to allow anyone to understand
at least *most* of the mathematical content of the book.
If you wish to understand *all* of the mathematical content,
further reviewing the [online appendix on mathematics](https://d2l.ai/chapter_appendix-mathematics-for-deep-learning/index.html) should be sufficient.

```toc
:maxdepth: 2

ndarray
pandas
linear-algebra
calculus
autograd
probability
lookup-api
```

