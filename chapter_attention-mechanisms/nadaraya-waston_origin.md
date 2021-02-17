# Attention Pooling: Nadaraya-Watson Kernel Regression
:label:`sec_nadaraya-waston`

Now you know the major components of attention mechanisms under the framework in :numref:`fig_qkv`.
To recapitulate,
the interactions between
queries (volitional cues) and keys (nonvolitional cues)
result in *attention pooling*.
The attention pooling selectively aggregates values (sensory inputs) to produce the output.
In this section,
we will describe attention pooling in greater detail
to give you a high-level view of
how attention mechanisms work in practice.
Specifically,
the Nadaraya-Watson kernel regression model
proposed in 1964
is a simple yet complete example
for demonstrating machine learning with attention mechanisms.

```{.python .input}
from d2l import mxnet as d2l
from mxnet import autograd, gluon, np, npx
from mxnet.gluon import nn

npx.set_np()
```

```{.python .input}
#@tab pytorch
from d2l import torch as d2l
import torch
from torch import nn
```

## Generating the Dataset

To keep things simple,
let us consider the following regression problem:
given a dataset of input-output pairs $\{(x_1, y_1), \ldots, (x_n, y_n)\}$,
how to learn $f$ to predict the output $\hat{y} = f(x)$ for any new input $x$?

Here we generate an artificial dataset according to the following nonlinear function with the noise term $\epsilon$:

$$y_i = 2\sin(x_i) + x_i^{0.8} + \epsilon,$$

where $\epsilon$ obeys a normal distribution with zero mean and standard deviation 0.5.
Both 50 training examples and 50 testing examples
are generated.
To better visualize the pattern of attention later, the training inputs are sorted.

```{.python .input}
n_train = 50  # No. of training examples
x_train = np.sort(d2l.rand(n_train) * 5)   # Training inputs
```

```{.python .input}
#@tab pytorch
n_train = 50  # No. of training examples
x_train, _ = torch.sort(d2l.rand(n_train) * 5)   # Training inputs
```

```{.python .input}
#@tab all
def f(x):
    return 2 * d2l.sin(x) + x**0.8

y_train = f(x_train) + d2l.normal(0.0, 0.5, (n_train,))  # Training outputs
x_test = d2l.arange(0, 5, 0.1)  # Testing examples
y_truth = f(x_test)  # Ground-truth outputs for the testing examples
n_test = len(x_test)  # No. of testing examples
n_test
```

The following function plots all the training examples (represented by circles),
the ground-truth data generation function `f` without the noise term (labeled by "Truth"), and the learned prediction function (labeled by "Pred").

```{.python .input}
#@tab all
def plot_kernel_reg(y_hat):
    d2l.plot(x_test, [y_truth, y_hat], 'x', 'y', legend=['Truth', 'Pred'],
             xlim=[0, 5], ylim=[-1, 5])
    d2l.plt.plot(x_train, y_train, 'o', alpha=0.5);
```

## Average Pooling

We begin with perhaps the world's "dumbest" estimator for this regression problem:
using average pooling to average over all the training outputs:

$$f(x) = \frac{1}{n}\sum_{i=1}^n y_i,$$
:eqlabel:`eq_avg-pooling`

which is plotted below. As we can see, this estimator is indeed not so smart.

```{.python .input}
y_hat = y_train.mean().repeat(n_test)
plot_kernel_reg(y_hat)
```

```{.python .input}
#@tab pytorch
y_hat = torch.repeat_interleave(y_train.mean(), n_test)
plot_kernel_reg(y_hat)
```

## Nonparametric Attention Pooling

Obviously,
average pooling omits the inputs $x_i$.
A better idea was proposed
by Nadaraya :cite:`Nadaraya.1964`
and Waston :cite:`Watson.1964`
to weigh the outputs $y_i$ according to their input locations:

$$f(x) = \sum_{i=1}^n \frac{K(x - x_i)}{\sum_{j=1}^n K(x - x_j)} y_i,$$
:eqlabel:`eq_nadaraya-waston`

where $K$ is a *kernel*.
The estimator in :eqref:`eq_nadaraya-waston`
is called *Nadaraya-Watson kernel regression*.
Here we will not dive into details of kernels.
Recall the framework of attention mechanisms in :numref:`fig_qkv`.
From the perspective of attention,
we can rewrite :eqref:`eq_nadaraya-waston`
in a more generalized form of *attention pooling*:

$$f(x) = \sum_{i=1}^n \alpha(x, x_i) y_i,$$
:eqlabel:`eq_attn-pooling`


where $x$ is the query and $(x_i, y_i)$ is the key-value pair.
Comparing :eqref:`eq_attn-pooling` and :eqref:`eq_avg-pooling`,
the attention pooling here
is a weighted average of values $y_i$.
The *attention weight* $\alpha(x, x_i)$
in :eqref:`eq_attn-pooling`
is assigned to the corresponding value $y_i$
based on the interaction
between the query $x$ and the key $x_i$
modeled by $\alpha$.
For any query, its attention weights over all the key-value pairs are a valid probability distribution: they are non-negative and sum up to one.

To gain intuitions of attention pooling,
just consider a *Gaussian kernel* defined as

$$
K(u) = \frac{1}{\sqrt{2\pi}} \exp(-\frac{u^2}{2}).
$$


Plugging the Gaussian kernel into
:eqref:`eq_attn-pooling` and
:eqref:`eq_nadaraya-waston` gives

$$\begin{aligned} f(x) &=\sum_{i=1}^n \alpha(x, x_i) y_i\\ &= \sum_{i=1}^n \frac{\exp\left(-\frac{1}{2}(x - x_i)^2\right)}{\sum_{j=1}^n \exp\left(-\frac{1}{2}(x - x_j)^2\right)} y_i \\&= \sum_{i=1}^n \mathrm{softmax}\left(-\frac{1}{2}(x - x_i)^2\right) y_i. \end{aligned}$$
:eqlabel:`eq_nadaraya-waston-gaussian`

In :eqref:`eq_nadaraya-waston-gaussian`,
a key $x_i$ that is closer to the given query $x$ will get
*more attention* via a *larger attention weight* assigned to the key's corresponding value $y_i$.

Notably, Nadaraya-Watson kernel regression is a nonparametric model;
thus :eqref:`eq_nadaraya-waston-gaussian`
is an example of *nonparametric attention pooling*.
In the following, we plot the prediction based on this
nonparametric attention model.
The predicted line is smooth and closer to the ground-truth than that produced by average pooling.

```{.python .input}
# Shape of `X_repeat`: (`n_test`, `n_train`), where each row contains the
# same testing inputs (i.e., same queries)
X_repeat = d2l.reshape(x_test.repeat(n_train), (-1, n_train))
# Note that `x_train` contains the keys. Shape of `attention_weights`:
# (`n_test`, `n_train`), where each row contains attention weights to be
# assigned among the values (`y_train`) given each query
attention_weights = npx.softmax(-(X_repeat - x_train)**2 / 2)
# Each element of `y_hat` is weighted average of values, where weights are
# attention weights
y_hat = d2l.matmul(attention_weights, y_train)
plot_kernel_reg(y_hat)
```

```{.python .input}
#@tab pytorch
# Shape of `X_repeat`: (`n_test`, `n_train`), where each row contains the
# same testing inputs (i.e., same queries)
X_repeat = d2l.reshape(x_test.repeat_interleave(n_train), (-1, n_train))
# Note that `x_train` contains the keys. Shape of `attention_weights`:
# (`n_test`, `n_train`), where each row contains attention weights to be
# assigned among the values (`y_train`) given each query
attention_weights = nn.functional.softmax(-(X_repeat - x_train)**2 / 2, dim=1)
# Each element of `y_hat` is weighted average of values, where weights are
# attention weights
y_hat = d2l.matmul(attention_weights, y_train)
plot_kernel_reg(y_hat)
```

Now let us take a look at the attention weights.
Here testing inputs are queries while training inputs are keys.
Since both inputs are sorted,
we can see that the closer the query-key pair is,
the higher attention weight is in the attention pooling.

```{.python .input}
d2l.show_heatmaps(np.expand_dims(np.expand_dims(attention_weights, 0), 0),
                  xlabel='Sorted training inputs',
                  ylabel='Sorted testing inputs')
```

```{.python .input}
#@tab pytorch
d2l.show_heatmaps(attention_weights.unsqueeze(0).unsqueeze(0),
                  xlabel='Sorted training inputs',
                  ylabel='Sorted testing inputs')
```

## Parametric Attention Pooling

Nonparametric Nadaraya-Watson kernel regression
enjoys the *consistency* benefit:
given enough data this model converges to the optimal solution.
Nonetheless,
we can easily integrate learnable parameters into attention pooling.

As an example, slightly different from :eqref:`eq_nadaraya-waston-gaussian`,
in the following
the distance between the query $x$ and the key $x_i$
is multiplied a learnable parameter $w$:


$$\begin{aligned}f(x) &= \sum_{i=1}^n \alpha(x, x_i) y_i \\&= \sum_{i=1}^n \frac{\exp\left(-\frac{1}{2}((x - x_i)w)^2\right)}{\sum_{j=1}^n \exp\left(-\frac{1}{2}((x - x_i)w)^2\right)} y_i \\&= \sum_{i=1}^n \mathrm{softmax}\left(-\frac{1}{2}((x - x_i)w)^2\right) y_i.\end{aligned}$$
:eqlabel:`eq_nadaraya-waston-gaussian-para`

In the rest of the section,
we will train this model by learning the parameter of
the attention pooling in :eqref:`eq_nadaraya-waston-gaussian-para`.


### Batch Matrix Multiplication
:label:`subsec_batch_dot`

To more efficiently compute attention
for minibatches,
we can leverage batch matrix multiplication utilities
provided by deep learning frameworks.


Suppose that the first minibatch contains $n$ matrices $\mathbf{X}_1, \ldots, \mathbf{X}_n$ of shape $a\times b$, and the second minibatch contains $n$ matrices $\mathbf{Y}_1, \ldots, \mathbf{Y}_n$ of shape $b\times c$. Their batch matrix multiplication
results in
$n$ matrices $\mathbf{X}_1\mathbf{Y}_1, \ldots, \mathbf{X}_n\mathbf{Y}_n$ of shape $a\times c$. Therefore, given two tensors of shape ($n$, $a$, $b$) and ($n$, $b$, $c$), the shape of their batch matrix multiplication output is ($n$, $a$, $c$).

```{.python .input}
X = d2l.ones((2, 1, 4))
Y = d2l.ones((2, 4, 6))
npx.batch_dot(X, Y).shape
```

```{.python .input}
#@tab pytorch
X = d2l.ones((2, 1, 4))
Y = d2l.ones((2, 4, 6))
torch.bmm(X, Y).shape
```

In the context of attention mechanisms, we can use minibatch matrix multiplication to compute weighted averages of values in a minibatch.

```{.python .input}
weights = d2l.ones((2, 10)) * 0.1
values = d2l.reshape(d2l.arange(20), (2, 10))
npx.batch_dot(np.expand_dims(weights, 1), np.expand_dims(values, -1))
```

```{.python .input}
#@tab pytorch
weights = d2l.ones((2, 10)) * 0.1
values = d2l.reshape(d2l.arange(20.0), (2, 10))
torch.bmm(weights.unsqueeze(1), values.unsqueeze(-1))
```

### Defining the Model

Using minibatch matrix multiplication,
below we define the parametric version
of Nadaraya-Watson kernel regression
based on the parametric attention pooling in
:eqref:`eq_nadaraya-waston-gaussian-para`.

```{.python .input}
class NWKernelRegression(nn.Block):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.w = self.params.get('w', shape=(1,))

    def forward(self, queries, keys, values):
        # Shape of the output `queries` and `attention_weights`:
        # (no. of queries, no. of key-value pairs)
        queries = d2l.reshape(
            queries.repeat(keys.shape[1]), (-1, keys.shape[1]))
        self.attention_weights = npx.softmax(
            -((queries - keys) * self.w.data())**2 / 2)
        # Shape of `values`: (no. of queries, no. of key-value pairs)
        return npx.batch_dot(np.expand_dims(self.attention_weights, 1),
                             np.expand_dims(values, -1)).reshape(-1)
```

```{.python .input}
#@tab pytorch
class NWKernelRegression(nn.Module):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.w = nn.Parameter(torch.rand((1,), requires_grad=True))

    def forward(self, queries, keys, values):
        # Shape of the output `queries` and `attention_weights`:
        # (no. of queries, no. of key-value pairs)
        queries = d2l.reshape(
            queries.repeat_interleave(keys.shape[1]), (-1, keys.shape[1]))
        self.attention_weights = nn.functional.softmax(
            -((queries - keys) * self.w)**2 / 2, dim=1)
        # Shape of `values`: (no. of queries, no. of key-value pairs)
        return torch.bmm(self.attention_weights.unsqueeze(1),
                         values.unsqueeze(-1)).reshape(-1)
```

### Training

In the following, we transform the training dataset
to keys and values to train the attention model.
In the parametric attention pooling,
any training input takes key-value pairs from all the training examples except for itself to predict its output.

```{.python .input}
# Shape of `X_tile`: (`n_train`, `n_train`), where each column contains the
# same training inputs
X_tile = np.tile(x_train, (n_train, 1))
# Shape of `Y_tile`: (`n_train`, `n_train`), where each column contains the
# same training outputs
Y_tile = np.tile(y_train, (n_train, 1))
# Shape of `keys`: ('n_train', 'n_train' - 1)
keys = d2l.reshape(X_tile[(1 - d2l.eye(n_train)).astype('bool')],
                   (n_train, -1))
# Shape of `values`: ('n_train', 'n_train' - 1)
values = d2l.reshape(Y_tile[(1 - d2l.eye(n_train)).astype('bool')],
                     (n_train, -1))
```

```{.python .input}
#@tab pytorch
# Shape of `X_tile`: (`n_train`, `n_train`), where each column contains the
# same training inputs
X_tile = x_train.repeat((n_train, 1))
# Shape of `Y_tile`: (`n_train`, `n_train`), where each column contains the
# same training outputs
Y_tile = y_train.repeat((n_train, 1))
# Shape of `keys`: ('n_train', 'n_train' - 1)
keys = d2l.reshape(X_tile[(1 - d2l.eye(n_train)).type(torch.bool)],
                   (n_train, -1))
# Shape of `values`: ('n_train', 'n_train' - 1)
values = d2l.reshape(Y_tile[(1 - d2l.eye(n_train)).type(torch.bool)],
                     (n_train, -1))
```

Using the squared loss and stochastic gradient descent,
we train the parametric attention model.

```{.python .input}
net = NWKernelRegression()
net.initialize()
loss = gluon.loss.L2Loss()
trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': 0.5})
animator = d2l.Animator(xlabel='epoch', ylabel='loss', xlim=[1, 5])

for epoch in range(5):
    with autograd.record():
        l = loss(net(x_train, keys, values), y_train)
    l.backward()
    trainer.step(1)
    print(f'epoch {epoch + 1}, loss {float(l.sum()):.6f}')
    animator.add(epoch + 1, float(l.sum()))
```

```{.python .input}
#@tab pytorch
net = NWKernelRegression()
loss = nn.MSELoss(reduction='none')
trainer = torch.optim.SGD(net.parameters(), lr=0.5)
animator = d2l.Animator(xlabel='epoch', ylabel='loss', xlim=[1, 5])

for epoch in range(5):
    trainer.zero_grad()
    # Note: L2 Loss = 1/2 * MSE Loss. PyTorch has MSE Loss which is slightly
    # different from MXNet's L2Loss by a factor of 2. Hence we halve the loss
    l = loss(net(x_train, keys, values), y_train) / 2
    l.sum().backward()
    trainer.step()
    print(f'epoch {epoch + 1}, loss {float(l.sum()):.6f}')
    animator.add(epoch + 1, float(l.sum()))
```

After training the parametric attention model,
we can plot its prediction.
Trying to fit the training dataset with noise,
the predicted line is less smooth
than its nonparametric counterpart that was plotted earlier.

```{.python .input}
# Shape of `keys`: (`n_test`, `n_train`), where each column contains the same
# training inputs (i.e., same keys)
keys = np.tile(x_train, (n_test, 1))
# Shape of `value`: (`n_test`, `n_train`)
values = np.tile(y_train, (n_test, 1))
y_hat = net(x_test, keys, values)
plot_kernel_reg(y_hat)
```

```{.python .input}
#@tab pytorch
# Shape of `keys`: (`n_test`, `n_train`), where each column contains the same
# training inputs (i.e., same keys)
keys = x_train.repeat((n_test, 1))
# Shape of `value`: (`n_test`, `n_train`)
values = y_train.repeat((n_test, 1))
y_hat = net(x_test, keys, values).unsqueeze(1).detach()
plot_kernel_reg(y_hat)
```

Comparing with nonparametric attention pooling,
the region with large attention weights becomes sharper
in the learnable and parametric setting.

```{.python .input}
d2l.show_heatmaps(np.expand_dims(np.expand_dims(net.attention_weights, 0), 0),
                  xlabel='Sorted training inputs',
                  ylabel='Sorted testing inputs')
```

```{.python .input}
#@tab pytorch
d2l.show_heatmaps(net.attention_weights.unsqueeze(0).unsqueeze(0),
                  xlabel='Sorted training inputs',
                  ylabel='Sorted testing inputs')
```

## Summary

* Nadaraya-Watson kernel regression is an example of machine learning with attention mechanisms.
* The attention pooling of Nadaraya-Watson kernel regression is a weighted average of the training outputs. From the attention perspective, the attention weight is assigned to a value based on a function of a query and the key that is paired with the value.
* Attention pooling can be either nonparametric or parametric.


## Exercises

1. Increase the number of training examples. Can you learn  nonparametric Nadaraya-Watson kernel regression better?
1. What is the value of our learned $w$ in the parametric attention pooling experiment? Why does it make the weighted region sharper when visualizing the attention weights?
1. How can we add hyperparameters to nonparametric Nadaraya-Watson kernel regression to predict better?
1. Design another parametric attention pooling for the kernel regression of this section. Train this new model and visualize its attention weights.



:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/1598)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/1599)
:end_tab:
