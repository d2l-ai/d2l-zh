# Data Manipulation
:label:`sec_ndarray`

In order to get anything done, we need some way to store and manipulate data.
Generally, there are two important things we need to do with data: (i) acquire
them; and (ii) process them once they are inside the computer.  There is no
point in acquiring data without some way to store it, so let us get our hands
dirty first by playing with synthetic data.  To start, we introduce the
$n$-dimensional array, which is also called the *tensor*.

If you have worked with NumPy, the most widely-used
scientific computing package in Python,
then you will find this section familiar.
No matter which framework you use,
its *tensor class* (`ndarray` in MXNet,
`Tensor` in both PyTorch and TensorFlow) is similar to NumPy's `ndarray` with
a few killer features.
First, GPU is well-supported to accelerate the computation
whereas NumPy only supports CPU computation.
Second, the tensor class
supports automatic differentiation.
These properties make the tensor class suitable for deep learning.
Throughout the book, when we say tensors,
we are referring to instances of the tensor class unless otherwise stated.

## Getting Started

In this section, we aim to get you up and running,
equipping you with the basic math and numerical computing tools
that you will build on as you progress through the book.
Do not worry if you struggle to grok some of
the mathematical concepts or library functions.
The following sections will revisit this material
in the context of practical examples and it will sink.
On the other hand, if you already have some background
and want to go deeper into the mathematical content, just skip this section.

:begin_tab:`mxnet`
To start, we import the `np` (`numpy`) and
`npx` (`numpy_extension`) modules from MXNet.
Here, the `np` module includes functions supported by NumPy,
while the `npx` module contains a set of extensions
developed to empower deep learning within a NumPy-like environment.
When using tensors, we almost always invoke the `set_np` function:
this is for compatibility of tensor processing by other components of MXNet.
:end_tab:

:begin_tab:`pytorch`
(**To start, we import `torch`. Note that though it's called PyTorch, we should
import `torch` instead of `pytorch`.**)
:end_tab:

:begin_tab:`tensorflow`
To start, we import `tensorflow`. As the name is a little long, we often import
it with a short alias `tf`.
:end_tab:

```{.python .input}
from mxnet import np, npx
npx.set_np()
```

```{.python .input}
#@tab pytorch
import torch
```

```{.python .input}
#@tab tensorflow
import tensorflow as tf
```

[**A tensor represents a (possibly multi-dimensional) array of numerical values.**]
With one axis, a tensor corresponds (in math) to a *vector*.
With two axes, a tensor corresponds to a *matrix*.
Tensors with more than two axes do not have special
mathematical names.

To start, we can use `arange` to create a row vector `x`
containing the first 12 integers starting with 0,
though they are created as floats by default.
Each of the values in a tensor is called an *element* of the tensor.
For instance, there are 12 elements in the tensor `x`.
Unless otherwise specified, a new tensor
will be stored in main memory and designated for CPU-based computation.


```{.python .input}
x = np.arange(12)
x
```

```{.python .input}
#@tab pytorch
x = torch.arange(12)
x
```

```{.python .input}
#@tab tensorflow
x = tf.range(12)
x
```

(**We can access a tensor's *shape***) (~~and the total number of elements~~) (the length along each axis)
by inspecting its `shape` property.

```{.python .input}
#@tab all
x.shape
```

If we just want to know the total number of elements in a tensor,
i.e., the product of all of the shape elements,
we can inspect its size.
Because we are dealing with a vector here,
the single element of its `shape` is identical to its size.

```{.python .input}
x.size
```

```{.python .input}
#@tab pytorch
x.numel()
```

```{.python .input}
#@tab tensorflow
tf.size(x)
```

To [**change the shape of a tensor without altering
either the number of elements or their values**],
we can invoke the `reshape` function.
For example, we can transform our tensor, `x`,
from a row vector with shape (12,) to a matrix with shape (3, 4).
This new tensor contains the exact same values,
but views them as a matrix organized as 3 rows and 4 columns.
To reiterate, although the shape has changed,
the elements have not.
Note that the size is unaltered by reshaping.


```{.python .input}
#@tab mxnet, pytorch
X = x.reshape(3, 4)
X
```

```{.python .input}
#@tab tensorflow
X = tf.reshape(x, (3, 4))
X
```

Reshaping by manually specifying every dimension is unnecessary.
If our target shape is a matrix with shape (height, width),
then after we know the width, the height is given implicitly.
Why should we have to perform the division ourselves?
In the example above, to get a matrix with 3 rows,
we specified both that it should have 3 rows and 4 columns.
Fortunately, tensors can automatically work out one dimension given the rest.
We invoke this capability by placing `-1` for the dimension
that we would like tensors to automatically infer.
In our case, instead of calling `x.reshape(3, 4)`,
we could have equivalently called `x.reshape(-1, 4)` or `x.reshape(3, -1)`.

Typically, we will want our matrices initialized
either with zeros, ones, some other constants,
or numbers randomly sampled from a specific distribution.
[**We can create a tensor representing a tensor with all elements
set to 0**] (~~or 1~~)
and a shape of (2, 3, 4) as follows:


```{.python .input}
np.zeros((2, 3, 4))
```

```{.python .input}
#@tab pytorch
torch.zeros((2, 3, 4))
```

```{.python .input}
#@tab tensorflow
tf.zeros((2, 3, 4))
```

Similarly, we can create tensors with each element set to 1 as follows:

```{.python .input}
np.ones((2, 3, 4))
```

```{.python .input}
#@tab pytorch
torch.ones((2, 3, 4))
```

```{.python .input}
#@tab tensorflow
tf.ones((2, 3, 4))
```

Often, we want to [**randomly sample the values
for each element in a tensor**]
from some probability distribution.
For example, when we construct arrays to serve
as parameters in a neural network, we will
typically initialize their values randomly.
The following snippet creates a tensor with shape (3, 4).
Each of its elements is randomly sampled
from a standard Gaussian (normal) distribution
with a mean of 0 and a standard deviation of 1.


```{.python .input}
np.random.normal(0, 1, size=(3, 4))
```

```{.python .input}
#@tab pytorch
torch.randn(3, 4)
```

```{.python .input}
#@tab tensorflow
tf.random.normal(shape=[3, 4])
```

We can also [**specify the exact values for each element**] in the desired tensor
by supplying a Python list (or list of lists) containing the numerical values.
Here, the outermost list corresponds to axis 0, and the inner list to axis 1.


```{.python .input}
np.array([[2, 1, 4, 3], [1, 2, 3, 4], [4, 3, 2, 1]])
```

```{.python .input}
#@tab pytorch
torch.tensor([[2, 1, 4, 3], [1, 2, 3, 4], [4, 3, 2, 1]])
```

```{.python .input}
#@tab tensorflow
tf.constant([[2, 1, 4, 3], [1, 2, 3, 4], [4, 3, 2, 1]])
```

## Operations

This book is not about software engineering.
Our interests are not limited to simply
reading and writing data from/to arrays.
We want to perform mathematical operations on those arrays.
Some of the simplest and most useful operations
are the *elementwise* operations.
These apply a standard scalar operation
to each element of an array.
For functions that take two arrays as inputs,
elementwise operations apply some standard binary operator
on each pair of corresponding elements from the two arrays.
We can create an elementwise function from any function
that maps from a scalar to a scalar.

In mathematical notation, we would denote such
a *unary* scalar operator (taking one input)
by the signature $f: \mathbb{R} \rightarrow \mathbb{R}$.
This just means that the function is mapping
from any real number ($\mathbb{R}$) onto another.
Likewise, we denote a *binary* scalar operator
(taking two real inputs, and yielding one output)
by the signature $f: \mathbb{R}, \mathbb{R} \rightarrow \mathbb{R}$.
Given any two vectors $\mathbf{u}$ and $\mathbf{v}$ *of the same shape*,
and a binary operator $f$, we can produce a vector
$\mathbf{c} = F(\mathbf{u},\mathbf{v})$
by setting $c_i \gets f(u_i, v_i)$ for all $i$,
where $c_i, u_i$, and $v_i$ are the $i^\mathrm{th}$ elements
of vectors $\mathbf{c}, \mathbf{u}$, and $\mathbf{v}$.
Here, we produced the vector-valued
$F: \mathbb{R}^d, \mathbb{R}^d \rightarrow \mathbb{R}^d$
by *lifting* the scalar function to an elementwise vector operation.

The common standard arithmetic operators
(`+`, `-`, `*`, `/`, and `**`)
have all been *lifted* to elementwise operations
for any identically-shaped tensors of arbitrary shape.
We can call elementwise operations on any two tensors of the same shape.
In the following example, we use commas to formulate a 5-element tuple,
where each element is the result of an elementwise operation.

### Operations

[**The common standard arithmetic operators
(`+`, `-`, `*`, `/`, and `**`)
have all been *lifted* to elementwise operations.**]

```{.python .input}
x = np.array([1, 2, 4, 8])
y = np.array([2, 2, 2, 2])
x + y, x - y, x * y, x / y, x ** y  # The ** operator is exponentiation
```

```{.python .input}
#@tab pytorch
x = torch.tensor([1.0, 2, 4, 8])
y = torch.tensor([2, 2, 2, 2])
x + y, x - y, x * y, x / y, x ** y  # The ** operator is exponentiation
```

```{.python .input}
#@tab tensorflow
x = tf.constant([1.0, 2, 4, 8])
y = tf.constant([2.0, 2, 2, 2])
x + y, x - y, x * y, x / y, x ** y  # The ** operator is exponentiation
```

Many (**more operations can be applied elementwise**),
including unary operators like exponentiation.

```{.python .input}
np.exp(x)
```

```{.python .input}
#@tab pytorch
torch.exp(x)
```

```{.python .input}
#@tab tensorflow
tf.exp(x)
```

In addition to elementwise computations,
we can also perform linear algebra operations,
including vector dot products and matrix multiplication.
We will explain the crucial bits of linear algebra
(with no assumed prior knowledge) in :numref:`sec_linear-algebra`.

We can also [***concatenate* multiple tensors together,**]
stacking them end-to-end to form a larger tensor.
We just need to provide a list of tensors
and tell the system along which axis to concatenate.
The example below shows what happens when we concatenate
two matrices along rows (axis 0, the first element of the shape)
vs. columns (axis 1, the second element of the shape).
We can see that the first output tensor's axis-0 length ($6$)
is the sum of the two input tensors' axis-0 lengths ($3 + 3$);
while the second output tensor's axis-1 length ($8$)
is the sum of the two input tensors' axis-1 lengths ($4 + 4$).

```{.python .input}
X = np.arange(12).reshape(3, 4)
Y = np.array([[2, 1, 4, 3], [1, 2, 3, 4], [4, 3, 2, 1]])
np.concatenate([X, Y], axis=0), np.concatenate([X, Y], axis=1)
```

```{.python .input}
#@tab pytorch
X = torch.arange(12, dtype=torch.float32).reshape((3,4))
Y = torch.tensor([[2.0, 1, 4, 3], [1, 2, 3, 4], [4, 3, 2, 1]])
torch.cat((X, Y), dim=0), torch.cat((X, Y), dim=1)
```

```{.python .input}
#@tab tensorflow
X = tf.reshape(tf.range(12, dtype=tf.float32), (3, 4))
Y = tf.constant([[2.0, 1, 4, 3], [1, 2, 3, 4], [4, 3, 2, 1]])
tf.concat([X, Y], axis=0), tf.concat([X, Y], axis=1)
```


Sometimes, we want to [**construct a binary tensor via *logical statements*.**]
Take `X == Y` as an example.
For each position, if `X` and `Y` are equal at that position,
the corresponding entry in the new tensor takes a value of 1,
meaning that the logical statement `X == Y` is true at that position;
otherwise that position takes 0.

```{.python .input}
#@tab all
X == Y
```

[**Summing all the elements in the tensor**] yields a tensor with only one element.

```{.python .input}
#@tab mxnet, pytorch
X.sum()
```

```{.python .input}
#@tab tensorflow
tf.reduce_sum(X)
```

## Broadcasting Mechanism
:label:`subsec_broadcasting`

In the above section, we saw how to perform elementwise operations
on two tensors of the same shape. Under certain conditions,
even when shapes differ, we can still [**perform elementwise operations
by invoking the *broadcasting mechanism*.**]
This mechanism works in the following way:
First, expand one or both arrays
by copying elements appropriately
so that after this transformation,
the two tensors have the same shape.
Second, carry out the elementwise operations
on the resulting arrays.

In most cases, we broadcast along an axis where an array
initially only has length 1, such as in the following example:


```{.python .input}
a = np.arange(3).reshape(3, 1)
b = np.arange(2).reshape(1, 2)
a, b
```

```{.python .input}
#@tab pytorch
a = torch.arange(3).reshape((3, 1))
b = torch.arange(2).reshape((1, 2))
a, b
```

```{.python .input}
#@tab tensorflow
a = tf.reshape(tf.range(3), (3, 1))
b = tf.reshape(tf.range(2), (1, 2))
a, b
```

Since `a` and `b` are $3\times1$ and $1\times2$ matrices respectively,
their shapes do not match up if we want to add them.
We *broadcast* the entries of both matrices into a larger $3\times2$ matrix as follows:
for matrix `a` it replicates the columns
and for matrix `b` it replicates the rows
before adding up both elementwise.


```{.python .input}
#@tab all
a + b
```

## Indexing and Slicing

Just as in any other Python array, elements in a tensor can be accessed by index.
As in any Python array, the first element has index 0
and ranges are specified to include the first but *before* the last element.
As in standard Python lists, we can access elements
according to their relative position to the end of the list
by using negative indices.

Thus, [**`[-1]` selects the last element and `[1:3]`
selects the second and the third elements**] as follows:


```{.python .input}
#@tab all
X[-1], X[1:3]
```

:begin_tab:`mxnet, pytorch`
Beyond reading, (**we can also write elements of a matrix by specifying indices.**)
:end_tab:

:begin_tab:`tensorflow`
`Tensors` in TensorFlow are immutable, and cannot be assigned to.
`Variables` in TensorFlow are mutable containers of state that support
assignments. Keep in mind that gradients in TensorFlow do not flow backwards
through `Variable` assignments.

Beyond assigning a value to the entire `Variable`, we can write elements of a
`Variable` by specifying indices.
:end_tab:

```{.python .input}
#@tab mxnet, pytorch
X[1, 2] = 9
X
```

```{.python .input}
#@tab tensorflow
X_var = tf.Variable(X)
X_var[1, 2].assign(9)
X_var
```


If we want [**to assign multiple elements the same value,
we simply index all of them and then assign them the value.**]
For instance, `[0:2, :]` accesses the first and second rows,
where `:` takes all the elements along axis 1 (column).
While we discussed indexing for matrices,
this obviously also works for vectors
and for tensors of more than 2 dimensions.

```{.python .input}
#@tab mxnet, pytorch
X[0:2, :] = 12
X
```

```{.python .input}
#@tab tensorflow
X_var = tf.Variable(X)
X_var[0:2, :].assign(tf.ones(X_var[0:2,:].shape, dtype = tf.float32) * 12)
X_var
```

## Saving Memory

[**Running operations can cause new memory to be
allocated to host results.**]
For example, if we write `Y = X + Y`,
we will dereference the tensor that `Y` used to point to
and instead point `Y` at the newly allocated memory.
In the following example, we demonstrate this with Python's `id()` function,
which gives us the exact address of the referenced object in memory.
After running `Y = Y + X`, we will find that `id(Y)` points to a different location.
That is because Python first evaluates `Y + X`,
allocating new memory for the result and then makes `Y`
point to this new location in memory.

```{.python .input}
#@tab all
before = id(Y)
Y = Y + X
id(Y) == before
```

This might be undesirable for two reasons.
First, we do not want to run around
allocating memory unnecessarily all the time.
In machine learning, we might have
hundreds of megabytes of parameters
and update all of them multiple times per second.
Typically, we will want to perform these updates *in place*.
Second, we might point at the same parameters from multiple variables.
If we do not update in place, other references will still point to
the old memory location, making it possible for parts of our code
to inadvertently reference stale parameters.

:begin_tab:`mxnet, pytorch`
Fortunately, (**performing in-place operations**) is easy.
We can assign the result of an operation
to a previously allocated array with slice notation,
e.g., `Y[:] = <expression>`.
To illustrate this concept, we first create a new matrix `Z`
with the same shape as another `Y`,
using `zeros_like` to allocate a block of $0$ entries.
:end_tab:

:begin_tab:`tensorflow`
`Variables` are mutable containers of state in TensorFlow. They provide
a way to store your model parameters.
We can assign the result of an operation
to a `Variable` with `assign`.
To illustrate this concept, we create a `Variable` `Z`
with the same shape as another tensor `Y`,
using `zeros_like` to allocate a block of $0$ entries.
:end_tab:

```{.python .input}
Z = np.zeros_like(Y)
print('id(Z):', id(Z))
Z[:] = X + Y
print('id(Z):', id(Z))
```

```{.python .input}
#@tab pytorch
Z = torch.zeros_like(Y)
print('id(Z):', id(Z))
Z[:] = X + Y
print('id(Z):', id(Z))
```

```{.python .input}
#@tab tensorflow
Z = tf.Variable(tf.zeros_like(Y))
print('id(Z):', id(Z))
Z.assign(X + Y)
print('id(Z):', id(Z))
```

:begin_tab:`mxnet, pytorch`
[**If the value of `X` is not reused in subsequent computations,
we can also use `X[:] = X + Y` or `X += Y`
to reduce the memory overhead of the operation.**]
:end_tab:

:begin_tab:`tensorflow`
Even once you store state persistently in a `Variable`, you
may want to reduce your memory usage further by avoiding excess
allocations for tensors that are not your model parameters.

Because TensorFlow `Tensors` are immutable and gradients do not flow through
`Variable` assignments, TensorFlow does not provide an explicit way to run
an individual operation in-place.

However, TensorFlow provides the `tf.function` decorator to wrap computation
inside of a TensorFlow graph that gets compiled and optimized before running.
This allows TensorFlow to prune unused values, and to re-use
prior allocations that are no longer needed. This minimizes the memory
overhead of TensorFlow computations.
:end_tab:

```{.python .input}
#@tab mxnet, pytorch
before = id(X)
X += Y
id(X) == before
```

```{.python .input}
#@tab tensorflow
@tf.function
def computation(X, Y):
    Z = tf.zeros_like(Y)  # This unused value will be pruned out
    A = X + Y  # Allocations will be re-used when no longer needed
    B = A + Y
    C = B + Y
    return C + Y

computation(X, Y)
```


## Conversion to Other Python Objects

[**Converting to a NumPy tensor**], or vice versa, is easy.
The converted result does not share memory.
This minor inconvenience is actually quite important:
when you perform operations on the CPU or on GPUs,
you do not want to halt computation, waiting to see
whether the NumPy package of Python might want to be doing something else
with the same chunk of memory.


```{.python .input}
A = X.asnumpy()
B = np.array(A)
type(A), type(B)
```

```{.python .input}
#@tab pytorch
A = X.numpy()
B = torch.tensor(A)
type(A), type(B)
```

```{.python .input}
#@tab tensorflow
A = X.numpy()
B = tf.constant(A)
type(A), type(B)
```

To (**convert a size-1 tensor to a Python scalar**),
we can invoke the `item` function or Python's built-in functions.


```{.python .input}
a = np.array([3.5])
a, a.item(), float(a), int(a)
```

```{.python .input}
#@tab pytorch
a = torch.tensor([3.5])
a, a.item(), float(a), int(a)
```

```{.python .input}
#@tab tensorflow
a = tf.constant([3.5]).numpy()
a, a.item(), float(a), int(a)
```

## Summary

* The main interface to store and manipulate data for deep learning is the tensor ($n$-dimensional array). It provides a variety of functionalities including basic mathematics operations, broadcasting, indexing, slicing, memory saving, and conversion to other Python objects.


## Exercises

1. Run the code in this section. Change the conditional statement `X == Y` in this section to `X < Y` or `X > Y`, and then see what kind of tensor you can get.
1. Replace the two tensors that operate by element in the broadcasting mechanism with other shapes, e.g., 3-dimensional tensors. Is the result the same as expected?

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/26)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/27)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/187)
:end_tab:
