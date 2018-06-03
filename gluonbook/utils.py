import random
from time import time

from IPython.display import set_matplotlib_formats
from matplotlib import pyplot as plt
import mxnet as mx
from mxnet import autograd, gluon, image, nd
from mxnet.gluon import nn, data as gdata, loss as gloss, utils as gutils
import numpy as np

# set default figure size
set_matplotlib_formats('retina')
plt.rcParams['figure.figsize'] = (3.5, 2.5)

class DataLoader(object):
    """similiar to gluon.data.DataLoader, but might be faster.

    The main difference this data loader tries to read more exmaples each
    time. But the limits are 1) all examples in dataset have the same shape, 2)
    data transfomer needs to process multiple examples at each time
    """
    def __init__(self, dataset, batch_size, shuffle, transform=None):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.transform = transform

    def __iter__(self):
        data = self.dataset[:]
        X = data[0]
        y = nd.array(data[1])
        n = X.shape[0]
        if self.shuffle:
            idx = np.arange(n)
            np.random.shuffle(idx)
            X = nd.array(X.asnumpy()[idx])
            y = nd.array(y.asnumpy()[idx])

        for i in range(n//self.batch_size):
            if self.transform is not None:
                yield self.transform(X[i*self.batch_size:(i+1)*self.batch_size],
                                     y[i*self.batch_size:(i+1)*self.batch_size])
            else:
                yield (X[i*self.batch_size:(i+1)*self.batch_size],
                       y[i*self.batch_size:(i+1)*self.batch_size])

    def __len__(self):
        return len(self.dataset)//self.batch_size


def load_data_fashion_mnist(batch_size, resize=None, root="~/.mxnet/datasets/fashion-mnist"):
    """download the fashion mnist dataest and then load into memory"""
    def transform_mnist(data, label):
        # Transform a batch of examples.
        if resize:
            n = data.shape[0]
            new_data = nd.zeros((n, resize, resize, data.shape[3]))
            for i in range(n):
                new_data[i] = image.imresize(data[i], resize, resize)
            data = new_data
        # change data from batch x height x width x channel to batch x channel x height x width
        return nd.transpose(data.astype('float32'), (0,3,1,2))/255, label.astype('float32')

    mnist_train = gluon.data.vision.FashionMNIST(root=root, train=True, transform=None)
    mnist_test = gluon.data.vision.FashionMNIST(root=root, train=False, transform=None)
    # Transform later to avoid memory explosion.
    train_data = DataLoader(mnist_train, batch_size, shuffle=True, transform=transform_mnist)
    test_data = DataLoader(mnist_test, batch_size, shuffle=False, transform=transform_mnist)
    return (train_data, test_data)


def try_gpu():
    """If GPU is available, return mx.gpu(0); else return mx.cpu()"""
    try:
        ctx = mx.gpu()
        _ = nd.array([0], ctx=ctx)
    except:
        ctx = mx.cpu()
    return ctx


def try_all_gpus():
    """Return all available GPUs, or [mx.gpu()] if there is no GPU"""
    ctxes = []
    try:
        for i in range(16):
            ctx = mx.gpu(i)
            _ = nd.array([0], ctx=ctx)
            ctxes.append(ctx)
    except:
        pass
    if not ctxes:
        ctxes = [mx.cpu()]
    return ctxes


def sgd(params, lr, batch_size):
    """Mini-batch stochastic gradient descent."""
    for param in params:
        param[:] = param - lr * param.grad / batch_size


def accuracy(y_hat, y):
    """Get accuracy."""
    return (y_hat.argmax(axis=1) == y).mean().asscalar()


def _get_batch(batch, ctx):
    """return features and labels on ctx"""
    if isinstance(batch, mx.io.DataBatch):
        features = batch.data[0]
        labels = batch.label[0]
    else:
        features, labels = batch
    if labels.dtype != features.dtype:
        labels = labels.astype(features.dtype)
    return (gutils.split_and_load(features, ctx),
            gutils.split_and_load(labels, ctx),
            features.shape[0])


def evaluate_accuracy(data_iter, net, ctx=[mx.cpu()]):
    """Evaluate accuracy of a model on the given data set."""
    if isinstance(ctx, mx.Context):
        ctx = [ctx]
    acc = nd.array([0])
    n = 0
    if isinstance(data_iter, mx.io.MXDataIter):
        data_iter.reset()
    for batch in data_iter:
        features, labels, batch_size = _get_batch(batch, ctx)
        for X, y in zip(features, labels):
            y = y.astype('float32')
            acc += (net(X).argmax(axis=1)==y).sum().copyto(mx.cpu())
            n += y.size
        acc.wait_to_read()
    return acc.asscalar() / n


def train_cpu(net, train_iter, test_iter, loss, num_epochs, batch_size,
              params=None, lr=None, trainer=None):
    """Train and evaluate a model on CPU."""
    for epoch in range(1, num_epochs + 1):
        train_l_sum = 0
        train_acc_sum = 0
        for X, y in train_iter:
            with autograd.record():
                y_hat = net(X)
                l = loss(y_hat, y)
            l.backward()
            if trainer is None:
                sgd(params, lr, batch_size)
            else:
                trainer.step(batch_size)
            train_l_sum += l.mean().asscalar()
            train_acc_sum += accuracy(y_hat, y)
        test_acc = evaluate_accuracy(test_iter, net)
        print("epoch %d, loss %.4f, train acc %.3f, test acc %.3f"
              % (epoch, train_l_sum / len(train_iter),
                 train_acc_sum / len(train_iter), test_acc))


def train(train_iter, test_iter, net, loss, trainer, ctx, num_epochs, print_batches=None):
    """Train and evaluate a model."""
    print("training on", ctx)
    if isinstance(ctx, mx.Context):
        ctx = [ctx]
    for epoch in range(1, num_epochs + 1):
        train_l_sum, train_acc_sum, n, m = 0.0, 0.0, 0.0, 0.0
        if isinstance(train_iter, mx.io.MXDataIter):
            train_iter.reset()
        start = time()
        for i, batch in enumerate(train_iter):
            Xs, ys, batch_size = _get_batch(batch, ctx)
            ls = []
            with autograd.record():
                y_hats = [net(X) for X in Xs]
                ls = [loss(y_hat, y) for y_hat, y in zip(y_hats, ys)]
            for l in ls:
                l.backward()
            train_acc_sum += sum([(y_hat.argmax(axis=1) == y).sum().asscalar()
                                 for y_hat, y in zip(y_hats, ys)])
            train_l_sum += sum([l.sum().asscalar() for l in ls])
            trainer.step(batch_size)
            n += batch_size
            m += sum([y.size for y in ys])
            if print_batches and (i+1) % print_batches == 0:
                print("batch %d, loss %f, train acc %f" % (
                    n, train_l_sum / n, train_acc_sum / m
                ))
        test_acc = evaluate_accuracy(test_iter, net, ctx)
        print("epoch %d, loss %.4f, train acc %.3f, test acc %.3f, time %.1f sec" % (
            epoch, train_l_sum / n, train_acc_sum / m, test_acc, time() - start
        ))


class Residual(nn.HybridBlock):
    def __init__(self, channels, same_shape=True, **kwargs):
        super(Residual, self).__init__(**kwargs)
        self.same_shape = same_shape
        with self.name_scope():
            strides = 1 if same_shape else 2
            self.conv1 = nn.Conv2D(channels, kernel_size=3, padding=1,
                                  strides=strides)
            self.bn1 = nn.BatchNorm()
            self.conv2 = nn.Conv2D(channels, kernel_size=3, padding=1)
            self.bn2 = nn.BatchNorm()
            if not same_shape:
                self.conv3 = nn.Conv2D(channels, kernel_size=1,
                                      strides=strides)

    def hybrid_forward(self, F, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if not self.same_shape:
            x = self.conv3(x)
        return F.relu(out + x)

def resnet18(num_classes):
    net = nn.HybridSequential()
    with net.name_scope():
        net.add(
            nn.BatchNorm(),
            nn.Conv2D(64, kernel_size=3, strides=1),
            nn.MaxPool2D(pool_size=3, strides=2),
            Residual(64),
            Residual(64),
            Residual(128, same_shape=False),
            Residual(128),
            Residual(256, same_shape=False),
            Residual(256),
            nn.GlobalAvgPool2D(),
            nn.Dense(num_classes)
        )
    return net

def show_images(imgs, num_rows, num_cols, scale=2):
    """plot a list of images"""
    figsize = (num_cols*scale, num_rows*scale)
    _, figs = plt.subplots(num_rows, num_cols, figsize=figsize)
    for i in range(num_rows):
        for j in range(num_cols):
            figs[i][j].imshow(imgs[i*num_cols+j].asnumpy())
            figs[i][j].axes.get_xaxis().set_visible(False)
            figs[i][j].axes.get_yaxis().set_visible(False)
    plt.show()


def to_onehot(X, size):
    """Represent inputs with one-hot encoding."""
    return [nd.one_hot(x, size) for x in X.T]


def data_iter_random(corpus_indices, batch_size, num_steps, ctx=None):
    """Sample mini-batches in a random order from sequential data."""
    num_examples = (len(corpus_indices) - 1) // num_steps
    epoch_size = num_examples // batch_size
    example_indices = list(range(num_examples))
    random.shuffle(example_indices)
    def _data(pos):
        return corpus_indices[pos : pos+num_steps]
    for i in range(epoch_size):
        i = i * batch_size
        batch_indices = example_indices[i : i+batch_size]
        X = nd.array(
            [_data(j * num_steps) for j in batch_indices], ctx=ctx)
        Y = nd.array(
            [_data(j * num_steps + 1) for j in batch_indices], ctx=ctx)
        yield X, Y


def data_iter_consecutive(corpus_indices, batch_size, num_steps, ctx=None):
    """Sample mini-batches in a consecutive order from sequential data."""
    corpus_indices = nd.array(corpus_indices, ctx=ctx)
    data_len = len(corpus_indices)
    batch_len = data_len // batch_size
    indices = corpus_indices[0 : batch_size*batch_len].reshape((
        batch_size, batch_len))
    epoch_size = (batch_len - 1) // num_steps
    for i in range(epoch_size):
        i = i * num_steps
        X = indices[:, i : i+num_steps]
        Y = indices[:, i+1 : i+num_steps+1]
        yield X, Y


def grad_clipping(params, theta, ctx):
    """Clip the gradient."""
    if theta is not None:
        norm = nd.array([0.0], ctx)
        for param in params:
            norm += (param.grad ** 2).sum()
        norm = norm.sqrt().asscalar()
        if norm > theta:
            for param in params:
                param.grad[:] *= theta / norm 


def predict_rnn(rnn, prefix, num_chars, params, num_hiddens, vocab_size, ctx,
                idx_to_char, char_to_idx, get_inputs, is_lstm=False):
    """Predict the next chars given the prefix."""
    prefix = prefix.lower()
    state_h = nd.zeros(shape=(1, num_hiddens), ctx=ctx)
    if is_lstm:
        state_c = nd.zeros(shape=(1, num_hiddens), ctx=ctx)
    output = [char_to_idx[prefix[0]]]
    for i in range(num_chars + len(prefix)):
        X = nd.array([output[-1]], ctx=ctx)
        if is_lstm:
            Y, state_h, state_c = rnn(get_inputs(X, vocab_size), state_h,
                                      state_c, *params)
        else:
            Y, state_h = rnn(get_inputs(X, vocab_size), state_h, *params)
        if i < len(prefix) - 1:
            next_input = char_to_idx[prefix[i + 1]]
        else:
            next_input = int(Y[0].argmax(axis=1).asscalar())
        output.append(next_input)
    return ''.join([idx_to_char[i] for i in output])


def train_and_predict_rnn(rnn, is_random_iter, num_epochs, num_steps,
                          num_hiddens, lr, clipping_theta, batch_size,
                          vocab_size, pred_period, pred_len, prefixes,
                          get_params, get_inputs, ctx, corpus_indices,
                          idx_to_char, char_to_idx, is_lstm=False):
    """Train an RNN model and predict the next item in the sequence."""
    if is_random_iter:
        data_iter = data_iter_random
    else:
        data_iter = data_iter_consecutive
    params = get_params()
    loss = gloss.SoftmaxCrossEntropyLoss()

    for epoch in range(1, num_epochs + 1):
        if not is_random_iter:
            state_h = nd.zeros(shape=(batch_size, num_hiddens), ctx=ctx)
            if is_lstm:
                state_c = nd.zeros(shape=(batch_size, num_hiddens), ctx=ctx)
        train_l_sum = nd.array([0], ctx=ctx)
        train_l_cnt = 0
        for X, Y in data_iter(corpus_indices, batch_size, num_steps, ctx):
            if is_random_iter:
                state_h = nd.zeros(shape=(batch_size, num_hiddens), ctx=ctx)
                if is_lstm:
                    state_c = nd.zeros(shape=(batch_size, num_hiddens),
                                       ctx=ctx)
            else:
                state_h = state_h.detach()
                if is_lstm:
                    state_c = state_c.detach()       
            with autograd.record():
                if is_lstm:
                    outputs, state_h, state_c = rnn(
                        get_inputs(X, vocab_size), state_h, state_c, *params) 
                else:
                    outputs, state_h = rnn(
                        get_inputs(X, vocab_size), state_h, *params)
                y = Y.T.reshape((-1,))
                outputs = nd.concat(*outputs, dim=0)
                l = loss(outputs, y)
            l.backward()
            grad_clipping(params, clipping_theta, ctx)
            sgd(params, lr, 1)
            train_l_sum = train_l_sum + l.sum()
            train_l_cnt += l.size
        if epoch % pred_period == 0:
            print("\nepoch %d, perplexity %f"
                  % (epoch, (train_l_sum / train_l_cnt).exp().asscalar()))
            for prefix in prefixes:
                print(' - ', predict_rnn(
                    rnn, prefix, pred_len, params, num_hiddens, vocab_size,
                    ctx, idx_to_char, char_to_idx, get_inputs, is_lstm))


def data_iter(batch_size, num_examples, features, labels):
    """Iterate through a data set."""
    indices = list(range(num_examples))
    random.shuffle(indices)
    for i in range(0, num_examples, batch_size):
        j = nd.array(indices[i: min(i + batch_size, num_examples)])
        yield features.take(j), labels.take(j)


def linreg(X, w, b):
    """Linear regression."""
    return nd.dot(X, w) + b


def squared_loss(y_hat, y):
    """Squared loss."""
    return (y_hat - y.reshape(y_hat.shape)) ** 2 / 2


def optimize(batch_size, trainer, num_epochs, decay_epoch, log_interval,
             features, labels, net):
    """Optimize an objective function."""
    dataset = gdata.ArrayDataset(features, labels)
    data_iter = gdata.DataLoader(dataset, batch_size, shuffle=True)
    loss = gloss.L2Loss()
    ls = [loss(net(features), labels).mean().asnumpy()]
    for epoch in range(1, num_epochs + 1):
        # Decay the learning rate.
        if decay_epoch and epoch > decay_epoch:
            trainer.set_learning_rate(trainer.learning_rate * 0.1)
        for batch_i, (X, y) in enumerate(data_iter):
            with autograd.record():
                l = loss(net(X), y)
            l.backward()
            trainer.step(batch_size)
            if batch_i * batch_size % log_interval == 0:
                ls.append(loss(net(features), labels).mean().asnumpy())
    # To print more conveniently, use numpy.
    print('w:', net[0].weight.data(), '\nb:', net[0].bias.data(), '\n')
    es = np.linspace(0, num_epochs, len(ls), endpoint=True)
    semilogy(es, ls, 'epoch', 'loss')


def semilogy(x_vals, y_vals, x_label, y_label, x2_vals=None, y2_vals=None,
             legend=None, figsize=(3.5, 2.5)):
    """Plot x and log(y)."""
    plt.rcParams['figure.figsize'] = figsize
    set_matplotlib_formats('retina')
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.semilogy(x_vals, y_vals)
    if x2_vals and y2_vals:
        plt.semilogy(x2_vals, y2_vals)
        plt.legend(legend)
    plt.show()
