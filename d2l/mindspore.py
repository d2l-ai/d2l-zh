import os
import sys
import re
import math
import time
import random
import requests
import hashlib
import collections
import zipfile
import mindspore
import numpy as np
import mindspore.numpy as mnp
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore.ops import constexpr
from mindspore import Tensor
from matplotlib import pyplot as plt

d2l = sys.modules[__name__]

from IPython import display
from PIL import Image
import mindspore.dataset.vision as vision
import mindspore.dataset.transforms as transforms
import mindspore.dataset as ds
from mindspore.common.initializer import initializer, HeUniform, Uniform, Normal, _calculate_fan_in_and_fan_out

DATA_HUB = dict()
DATA_URL = 'http://d2l-data.s3-accelerate.amazonaws.com/'
DATA_HUB['time_machine'] = (DATA_URL + 'timemachine.txt',
                                '090b5e7e70c295757f55df93cb0a180b9691891a')
DATA_HUB['fra-eng'] = (DATA_URL + 'fra-eng.zip',
                           '94646ad1522d915e7b0f9296181140edcf86a4f5')

class Timer:
    """记录多次运行时间。"""
    def __init__(self):
        """Defined in :numref:`subsec_linear_model`"""
        self.times = []
        self.start()

    def start(self):
        """启动计时器。"""
        self.tik = time.time()

    def stop(self):
        """停止计时器并将时间记录在列表中。"""
        self.times.append(time.time() - self.tik)
        return self.times[-1]

    def avg(self):
        """返回平均时间。"""
        return sum(self.times) / len(self.times)

    def sum(self):
        """返回时间总和。"""
        return sum(self.times)

    def cumsum(self):
        """返回累计时间。"""
        return np.array(self.times).cumsum().tolist()

class Accumulator:
    """在`n`个变量上累加。"""
    def __init__(self, n):
        self.data = [0.0] * n

    def add(self, *args):
        self.data = [a + float(b) for a, b in zip(self.data, args)]

    def reset(self):
        self.data = [0.0] * len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

class Animator:
    """在动画中绘制数据。"""
    def __init__(self, xlabel=None, ylabel=None, legend=None, xlim=None,
                 ylim=None, xscale='linear', yscale='linear',
                 fmts=('-', 'm--', 'g-.', 'r:'), nrows=1, ncols=1,
                 figsize=(3.5, 2.5)):
        if legend is None:
            legend = []
        use_svg_display()
        self.fig, self.axes = plt.subplots(nrows, ncols, figsize=figsize)
        if nrows * ncols == 1:
            self.axes = [self.axes, ]
        self.config_axes = lambda: set_axes(
            self.axes[0], xlabel, ylabel, xlim, ylim, xscale, yscale, legend)
        self.X, self.Y, self.fmts = None, None, fmts

    def add(self, x, y):
        if not hasattr(y, "__len__"):
            y = [y]
        n = len(y)
        if not hasattr(x, "__len__"):
            x = [x] * n
        if not self.X:
            self.X = [[] for _ in range(n)]
        if not self.Y:
            self.Y = [[] for _ in range(n)]
        for i, (a, b) in enumerate(zip(x, y)):
            if a is not None and b is not None:
                self.X[i].append(a)
                self.Y[i].append(b)
        self.axes[0].cla()
        for x, y, fmt in zip(self.X, self.Y, self.fmts):
            self.axes[0].plot(x, y, fmt)
        self.config_axes()
        display.display(self.fig)
        display.clear_output(wait=True)

class FashionMnist():
    def __init__(self, path, kind):
        self.data, self.label = load_mnist(path, kind)

    def __getitem__(self, index):
        return self.data[index], self.label[index]

    def __len__(self):
        return len(self.data)

class ArrayData():
    def __init__(self, data):
        assert len(data) > 1
        self.data = data

    def __getitem__(self, index):
        return (i[index] for i in self.data)

    def __len__(self):
        return len(self.data[0])

class SGD(nn.Cell):
    def __init__(self, lr, batch_size, parameters):
        super().__init__()
        self.lr = lr
        self.batch_size = batch_size
        self.parameters = parameters

    def construct(self, grads):
        for idx in range(len(self.parameters)):
            ops.assign(self.parameters[idx], self.parameters[idx] - self.lr * grads[idx] / self.batch_size)
        return True

class Train(nn.Cell):
    def __init__(self, network, optimizer):
        super().__init__()
        self.network = network
        self.optimizer = optimizer
        self.grad = ops.GradOperation(get_by_list=True)

    def construct(self, *inputs):
        loss = self.network(*inputs)
        grads = self.grad(self.network, self.optimizer.parameters)(*inputs)
        loss = ops.depend(loss, self.optimizer(grads))
        return loss

class NetWithLoss(nn.Cell):
    def __init__(self, network, loss):
        super().__init__()
        self.network = network
        self.loss = loss

    def construct(self, *inputs):
        y_hat = self.network(*inputs[:-1])
        loss = self.loss(y_hat, inputs[-1])
        return loss

class TrainCh8(nn.Cell):
    def __init__(self, network, optimizer, grad_op):
        super().__init__()
        self.network = network
        self.optimizer = optimizer
        self.grad = ops.GradOperation(get_by_list=True)
        self.grad_op = grad_op

    def construct(self, *inputs):
        loss = self.network(*inputs)
        grads = self.grad(self.network, self.optimizer.parameters)(*inputs)
        grads = self.grad_op(grads)
        loss = ops.depend(loss, self.optimizer(grads))
        return loss

class NetWithLossCh8(nn.Cell):
    def __init__(self, network, loss):
        super().__init__()
        self.network = network
        self.loss = loss

    def construct(self, *inputs):
        y_hat, state = self.network(*inputs[:-1])
        loss = self.loss(y_hat, inputs[-1])
        return loss


@constexpr
def compute_kernel_size(inp_shape, output_size):
    kernel_width, kernel_height = inp_shape[2], inp_shape[3]
    if isinstance(output_size, int):
        kernel_width = math.ceil(kernel_width / output_size)
        kernel_height = math.ceil(kernel_height / output_size)
    elif isinstance(output_size, list) or isinstance(output_size, tuple):
        kernel_width = math.ceil(kernel_width / output_size[0])
        kernel_height = math.ceil(kernel_height / output_size[1])
    return (kernel_width, kernel_height)

class AdaptiveAvgPool2d(nn.Cell):
    def __init__(self, output_size=None):
        super().__init__()
        self.output_size = output_size

    def construct(self, x):
        inp_shape = x.shape
        kernel_size = compute_kernel_size(inp_shape, self.output_size)
        return ops.AvgPool(kernel_size, kernel_size)(x)

class AdaptiveMaxPool2d(nn.Cell):
    def __init__(self, output_size=None):
        super().__init__()
        self.output_size = output_size

    def construct(self, x):
        inp_shape = x.shape
        kernel_size = compute_kernel_size(inp_shape, self.output_size)
        return ops.MaxPool(kernel_size, kernel_size)(x)

class MaxPool2d(nn.Cell):
    def __init__(self, kernel_size, stride=None, padding=0):
        super().__init__()
        if stride is None:
            stride = kernel_size
        self.max_pool = ops.MaxPool(kernel_size, stride)
        self.use_pad = padding != 0
        if isinstance(padding, tuple):
            assert len(padding) == 2
            paddings = ((0, 0), (0, 0), (padding[0], padding[0]), (padding[1], padding[1]))
        elif isinstance(padding, int):
            paddings = ((0, 0),) * 2 + ((padding, padding),) * 2
        else:
            raise ValueError('padding should be a tuple include 2 numbers or a int number')
        self.pad = ops.Pad(paddings)

    def construct(self, x):
        if self.use_pad:
            x = self.pad(x)
        return self.max_pool(x)

def linreg(x, w, b):
    return ops.matmul(x, w) + b

def squared_loss(y_hat, y):
    """均方损失。"""
    return (y_hat - y.reshape(y_hat.shape)) ** 2 / 2

def load_mnist(path, kind='train'):
    import os
    import gzip
    import numpy as np

    """Load MNIST data from `path`"""
    labels_path = os.path.join(path,
                               '%s-labels-idx1-ubyte.gz'
                               %kind)
    images_path = os.path.join(path,
                               '%s-images-idx3-ubyte.gz'
                               %kind)

    with gzip.open(labels_path, 'rb') as lbpath:
        labels = np.frombuffer(lbpath.read(), dtype=np.uint8,
                               offset=8)

    with gzip.open(images_path, 'rb') as imgpath:
        images = np.frombuffer(imgpath.read(), dtype=np.uint8,
                               offset=16).reshape(len(labels), 28, 28, 1)

    return images, labels

def load_data_fashion_mnist(batch_size, resize=None, works=1):
    """将Fashion-MNIST数据集加载到内存中。"""
    data_path = "../data"
    mnist_train = FashionMnist(data_path, kind='train')
    mnist_test = FashionMnist(data_path, kind='t10k')

    mnist_train = ds.GeneratorDataset(source=mnist_train, column_names=['image', 'label'], shuffle=False)
    mnist_test = ds.GeneratorDataset(source=mnist_test, column_names=['image', 'label'], shuffle=False)
    trans = [vision.Rescale(1.0 / 255.0, 0), vision.HWC2CHW()]
    type_cast_op = transforms.TypeCast(mindspore.int32)
    if resize:
        trans.insert(0, vision.Resize(resize))
    mnist_train = mnist_train.map(trans, input_columns=["image"])
    mnist_test = mnist_test.map(trans, input_columns=["image"])
    mnist_train = mnist_train.map(type_cast_op, input_columns=['label'])
    mnist_test = mnist_test.map(type_cast_op, input_columns=['label'])
    mnist_train = mnist_train.batch(batch_size, num_parallel_workers=works)
    mnist_test = mnist_test.batch(batch_size, num_parallel_workers=works)
    return mnist_train, mnist_test

def get_fashion_mnist_labels(labels):
    """Return text labels for the Fashion-MNIST dataset.
    Defined in :numref:`sec_utils`"""
    text_labels = ['t-shirt', 'trouser', 'pullover', 'dress', 'coat',
                   'sandal', 'shirt', 'sneaker', 'bag', 'ankle boot']
    return [text_labels[int(i)] for i in labels]

def load_array(data_arrays, batch_size, is_train=True):
    """Construct a PyTorch data iterator.
    Defined in :numref:`sec_utils`"""
    dataset = ArrayData(data_arrays)
    data_column_size = len(data_arrays)
    dataset = ds.GeneratorDataset(source=dataset, column_names=[str(i) for i in range(data_column_size)], shuffle=is_train)
    dataset = dataset.batch(batch_size)
    return dataset

def show_images(imgs, num_rows, num_cols, titles=None, scale=1.5):
    """Plot a list of images.
    Defined in :numref:`sec_utils`"""
    figsize = (num_cols * scale, num_rows * scale)
    _, axes = plt.subplots(num_rows, num_cols, figsize=figsize)
    axes = axes.flatten()
    for i, (ax, img) in enumerate(zip(axes, imgs)):
        try:
            img = img.asnumpy()
        except:
            pass
        ax.imshow(img)
        ax.axes.get_xaxis().set_visible(False)
        ax.axes.get_yaxis().set_visible(False)
        if titles:
            ax.set_title(titles[i])
    return axes

def use_svg_display():
    """Use the svg format to display a plot in Jupyter.
    Defined in :numref:`sec_calculus`"""
    display.set_matplotlib_formats('svg')

def set_figsize(figsize=(3.5, 2.5)):
    """Set the figure size for matplotlib.
    Defined in :numref:`sec_calculus`"""
    use_svg_display()
    plt.rcParams['figure.figsize'] = figsize

def set_axes(axes, xlabel, ylabel, xlim, ylim, xscale, yscale, legend):
    """Set the axes for matplotlib.
    Defined in :numref:`sec_calculus`"""
    axes.set_xlabel(xlabel), axes.set_ylabel(ylabel)
    axes.set_xscale(xscale), axes.set_yscale(yscale)
    axes.set_xlim(xlim),     axes.set_ylim(ylim)
    if legend:
        axes.legend(legend)
    axes.grid()

def plot(X, Y=None, xlabel=None, ylabel=None, legend=[], xlim=None,
         ylim=None, xscale='linear', yscale='linear',
         fmts=('-', 'm--', 'g-.', 'r:'), figsize=(3.5, 2.5), axes=None):
    """Plot data points.
    Defined in :numref:`sec_calculus`"""

    def has_one_axis(X):  # True if `X` (tensor or list) has 1 axis
        return (hasattr(X, "ndim") and X.ndim == 1 or isinstance(X, list)
                and not hasattr(X[0], "__len__"))

    if has_one_axis(X): X = [X]
    if Y is None:
        X, Y = [[]] * len(X), X
    elif has_one_axis(Y):
        Y = [Y]
    if len(X) != len(Y):
        X = X * len(Y)

    set_figsize(figsize)
    if axes is None: axes = plt.gca()
    axes.cla()
    for x, y, fmt in zip(X, Y, fmts):
        axes.plot(x,y,fmt) if len(x) else axes.plot(y,fmt)
    set_axes(axes, xlabel, ylabel, xlim, ylim, xscale, yscale, legend)

def synthetic_data(w, b, num_examples):
    """生成 y = Xw + b + 噪声。"""
    X = np.random.normal(0, 1, (num_examples, len(w)))
    y = np.matmul(X, w) + b
    y += np.random.normal(0, 0.01, y.shape)
    return X.astype(np.float32), y.reshape((-1, 1)).astype(np.float32)

def accuracy(y_hat, y):
    """计算预测正确的数量。"""
    if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:
        y_hat = y_hat.argmax(axis=1)
    cmp = y_hat.asnumpy() == y.asnumpy()
    return float(cmp.sum())

def evaluate_accuracy(net, dataset):
    """计算在指定数据集上模型的精度。"""
    metric = Accumulator(2)
    for X, y in dataset.create_tuple_iterator():
        metric.add(accuracy(net(X), y), y.size)
    return metric[0] / metric[1]

def train_epoch_ch3(net, dataset, loss, optim):
    """训练模型一个迭代周期（定义见第3章）。"""
    # 定义前向网络
    def forward_fn(x, y):
        y_hat = net(x)
        l = loss(y_hat, y)
        return l
    batch_size = dataset.get_batch_size()
    metric = Accumulator(3)
    for X, y in dataset:
        grad_fn = mindspore.value_and_grad(forward_fn, grad_position=None, weights=optim.parameters)
        l, grads = grad_fn(X, y)
        y_hat = net(X)
        optim(grads)
        metric.add(float(l.asnumpy()), accuracy(y_hat, y), y.size)
    return metric[0] / metric[2] * batch_size, metric[1] / metric[2]

def train_ch3(net, train_dataset, test_dataset, loss, num_epochs, optim):  
    """训练模型（定义见第3章）。"""
    animator = Animator(xlabel='epoch', xlim=[1, num_epochs], ylim=[0.3, 0.9],
                        legend=['train loss', 'train acc', 'test acc'])
    for epoch in range(num_epochs):
        train_metrics = train_epoch_ch3(net, train_dataset, loss, optim)
        test_acc = evaluate_accuracy(net, test_dataset)
        # print(train_metrics)
        animator.add(epoch + 1, train_metrics + (test_acc,))
    train_loss, train_acc = train_metrics

def predict_ch3(net, dataset, n=6):
    """预测标签（定义见第3章）。"""
    for X, y in dataset.create_tuple_iterator():
        break
    trues = get_fashion_mnist_labels(y.asnumpy())
    preds = get_fashion_mnist_labels(net(X).argmax(axis=1).asnumpy())
    titles = [true +'\n' + pred for true, pred in zip(trues, preds)]
    show_images(
        X[0:n].reshape((n, 28, 28)), 1, n, titles=titles[0:n])

def evaluate_loss(net, loss, dataset):
    """Evaluate the loss of a model on the given dataset.
    Defined in :numref:`sec_utils`"""
    metric = Accumulator(2)  # Sum of losses, no. of examples
    for X, y in dataset.create_tuple_iterator():
        z = net(X)
        l = loss(z, y)
        metric.add(l.sum().asnumpy(), l.size)
    return metric[0] / metric[1]

def corr2d(X, K):
    """计算二维互相关运算。"""
    h, w = K.shape
    Y = mnp.zeros((X.shape[0] - h + 1, X.shape[1] - w + 1))
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            Y[i, j] = (X[i:i + h, j:j + w] * K).sum()
    return Y

def evaluate_accuracy_gpu(net, dataset, device=None):
    """使用GPU计算模型在数据集上的精度。"""
    net.set_train(False)
    metric = Accumulator(2)
    for X, y in dataset.create_tuple_iterator():
        metric.add(accuracy(net(X), y), y.size)
    return metric[0] / metric[1]

def train_ch6(net, train_dataset, test_dataset, num_epochs, lr):
    """用GPU训练模型(在第六章定义)。"""
    optim = nn.SGD(net.trainable_params(), learning_rate=lr)
    loss = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction='mean')
    net_with_loss = nn.WithLossCell(net, loss)
    train = nn.TrainOneStepCell(net_with_loss, optim)
    animator = Animator(xlabel='epoch', xlim=[1, num_epochs],
                            legend=['train loss', 'train acc', 'test acc'])
    timer, num_batches = Timer(), train_dataset.get_dataset_size()
    for epoch in range(num_epochs):
        metric = Accumulator(3)
        net.set_train()
        for i, (X, y) in enumerate(train_dataset.create_tuple_iterator()):
            timer.start()
            l = train(X, y)
            y_hat = net(X)
            metric.add(l.asnumpy() * X.shape[0], accuracy(y_hat, y), X.shape[0])
            timer.stop()
            train_l = metric[0] / metric[2]
            train_acc = metric[1] / metric[2]
            if (i + 1) % (num_batches // 5) == 0 or i == num_batches - 1:
                animator.add(epoch + (i + 1) / num_batches,
                             (train_l, train_acc, None))
        test_acc = evaluate_accuracy_gpu(net, test_dataset)
        animator.add(epoch + 1, (None, None, test_acc))
    print(f'loss {train_l:.3f}, train acc {train_acc:.3f}, '
          f'test acc {test_acc:.3f}')
    print(f'{metric[2] * num_epochs / timer.sum():.1f} examples/sec')

def download(name, cache_dir=os.path.join('..', 'data')):
    """下载一个DATA_HUB中的文件，返回本地文件名。
    Defined in :numref:`sec_kaggle_house`"""
    assert name in DATA_HUB, f"{name} 不存在于 {DATA_HUB}."
    url, sha1_hash = DATA_HUB[name]
    os.makedirs(cache_dir, exist_ok=True)
    fname = os.path.join(cache_dir, url.split('/')[-1])
    if os.path.exists(fname):
        sha1 = hashlib.sha1()
        with open(fname, 'rb') as f:
            while True:
                data = f.read(1048576)
                if not data:
                    break
                sha1.update(data)
        if sha1.hexdigest() == sha1_hash:
            return fname  # Hit cache
    print(f'正在从{url}下载{fname}...')
    r = requests.get(url, stream=True, verify=True)
    with open(fname, 'wb') as f:
        f.write(r.content)
    return fname

def download_extract(name, folder=None):
    """下载并解压zip/tar文件。
    Defined in :numref:`sec_kaggle_house`"""
    fname = download(name)
    base_dir = os.path.dirname(fname)
    data_dir, ext = os.path.splitext(fname)
    if ext == '.zip':
        fp = zipfile.ZipFile(fname, 'r')
    elif ext in ('.tar', '.gz'):
        fp = tarfile.open(fname, 'r')
    else:
        assert False, '只有zip/tar文件可以被解压缩。'
    fp.extractall(base_dir)
    return os.path.join(base_dir, folder) if folder else data_dir

def download_all():
    """下载DATA_HUB中的所有文件。
    Defined in :numref:`sec_kaggle_house`"""
    for name in DATA_HUB:
        download(name)


def read_time_machine():
    """Load the time machine dataset into a list of text lines."""
    with open(download('time_machine'), 'r') as f:
        lines = f.readlines()
    return [re.sub('[^A-Za-z]+', ' ', line).strip().lower() for line in lines]

def tokenize(lines, token='word'):
    """将文本行拆分为单词或字符词元。"""
    if token == 'word':
        return [line.split() for line in lines]
    elif token == 'char':
        return [list(line) for line in lines]
    else:
        print('错误：未知词元类型：' + token)

class Vocab:
    """文本词表"""
    def __init__(self, tokens=None, min_freq=0, reserved_tokens=None):
        """Defined in :numref:`sec_text_preprocessing`"""
        if tokens is None:
            tokens = []
        if reserved_tokens is None:
            reserved_tokens = []
        # 按出现频率排序
        counter = count_corpus(tokens)
        self.token_freqs = sorted(counter.items(), key=lambda x: x[1],
                                  reverse=True)
        # 未知词元的索引为0
        self.unk, uniq_tokens = 0, ['<unk>'] + reserved_tokens
        uniq_tokens += [token for token, freq in self.token_freqs
                        if freq >= min_freq and token not in uniq_tokens]
        self.idx_to_token, self.token_to_idx = [], dict()
        for token in uniq_tokens:
            self.idx_to_token.append(token)
            self.token_to_idx[token] = len(self.idx_to_token) - 1

    def __len__(self):
        return len(self.idx_to_token)

    def __getitem__(self, tokens):
        if not isinstance(tokens, (list, tuple)):
            return self.token_to_idx.get(tokens, self.unk)
        return [self.__getitem__(token) for token in tokens]

    def to_tokens(self, indices):
        if not isinstance(indices, (list, tuple)):
            return self.idx_to_token[indices]
        return [self.idx_to_token[index] for index in indices]

def count_corpus(tokens):
    """统计词元的频率。"""
    if len(tokens) == 0 or isinstance(tokens[0], list):
        tokens = [token for line in tokens for token in line]
    return collections.Counter(tokens)

def load_corpus_time_machine(max_tokens=-1):
    """返回时光机器数据集的词元索引列表和词表。"""
    lines = read_time_machine()
    tokens = tokenize(lines, 'char')
    vocab = Vocab(tokens)
    corpus = [vocab[token] for line in tokens for token in line]
    if max_tokens > 0:
        corpus = corpus[:max_tokens]
    return corpus, vocab

def seq_data_iter_random(corpus, batch_size, num_steps):
    """使用随机抽样生成一个小批量子序列。"""
    corpus = corpus[random.randint(0, num_steps - 1):]
    num_subseqs = (len(corpus) - 1) // num_steps
    initial_indices = list(range(0, num_subseqs * num_steps, num_steps))
    random.shuffle(initial_indices)

    def data(pos):
        return corpus[pos: pos + num_steps]

    num_batches = num_subseqs // batch_size
    for i in range(0, batch_size * num_batches, batch_size):
        initial_indices_per_batch = initial_indices[i: i + batch_size]
        X = [data(j) for j in initial_indices_per_batch]
        Y = [data(j + 1) for j in initial_indices_per_batch]
        yield mindspore.Tensor(X, mindspore.int32), mindspore.Tensor(Y, mindspore.int32)

def seq_data_iter_sequential(corpus, batch_size, num_steps):
    """使用顺序分区生成一个小批量子序列。"""
    offset = random.randint(0, num_steps)
    num_tokens = ((len(corpus) - offset - 1) // batch_size) * batch_size
    Xs = mindspore.Tensor(corpus[offset: offset + num_tokens], mindspore.int32)
    Ys = mindspore.Tensor(corpus[offset + 1: offset + 1 + num_tokens], mindspore.int32)
    Xs, Ys = Xs.reshape(batch_size, -1), Ys.reshape(batch_size, -1)
    num_batches = Xs.shape[1] // num_steps
    for i in range(0, num_steps * num_batches, num_steps):
        X = Xs[:, i: i + num_steps]
        Y = Ys[:, i: i + num_steps]
        yield X, Y

class SeqDataLoader:
    """加载序列数据的迭代器。"""
    def __init__(self, batch_size, num_steps, use_random_iter, max_tokens):
        if use_random_iter:
            self.data_iter_fn = seq_data_iter_random
        else:
            self.data_iter_fn = seq_data_iter_sequential
        self.corpus, self.vocab = load_corpus_time_machine(max_tokens)
        self.batch_size, self.num_steps = batch_size, num_steps

    def __iter__(self):
        return self.data_iter_fn(self.corpus, self.batch_size, self.num_steps)

def load_data_time_machine(batch_size, num_steps,
                           use_random_iter=False, max_tokens=10000):
    """返回时光机器数据集的迭代器和词表。"""
    data_iter = SeqDataLoader(
        batch_size, num_steps, use_random_iter, max_tokens)
    return data_iter, data_iter.vocab

def predict_ch8(prefix, num_preds, net, vocab):
    """在`prefix`后面生成新字符。"""
    net.set_train(False)
    state = net.begin_state(batch_size=1)
    outputs = [vocab[prefix[0]]]
    get_input = lambda: mindspore.Tensor([outputs[-1]], mindspore.int32).reshape((1, 1))
    for y in prefix[1:]:
        _, state = net(get_input(), state)
        outputs.append(vocab[y])
    for _ in range(num_preds):
        y, state = net(get_input(), state)
        outputs.append(int(y.argmax(axis=1).reshape(1).asnumpy()))
    return ''.join([vocab.idx_to_token[i] for i in outputs])

class TrainCh8(nn.Cell):
    def __init__(self, network, optimizer, theta):
        super().__init__()
        self.network = network
        self.optimizer = optimizer
        self.grad = ops.GradOperation(get_by_list=True)
        self.theta = theta

    def construct(self, *inputs):
        loss = self.network(*inputs)
        grads = self.grad(self.network, self.optimizer.parameters)(*inputs)
        grads = ops.clip_by_global_norm(grads, self.theta)
        loss = ops.depend(loss, self.optimizer(grads))
        return loss

class NetWithLossCh8(nn.Cell):
    def __init__(self, network, loss):
        super().__init__()
        self.network = network
        self.loss = loss

    def construct(self, *inputs):
        y_hat, state = self.network(*inputs[:-1])
        loss = self.loss(y_hat, inputs[-1])
        return loss

def train_epoch_ch8(net, state, train_iter):
    """训练网络一个迭代周期（定义见第8章）。"""
    timer = Timer()
    metric = Accumulator(2)
    for X, Y in train_iter:
        y = Y.T.reshape(-1)
        l = net(X, state, y)
        metric.add(l.asnumpy() * y.size, y.size)
    return math.exp(metric[0] / metric[1]), metric[1] / timer.stop()

def train_ch8(net, train_iter, vocab, lr, num_epochs):
    """训练模型（定义见第8章）。"""
    animator = Animator(xlabel='epoch', ylabel='perplexity',
                            legend=['train'], xlim=[10, num_epochs])

    loss = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction='mean')
    optim = nn.SGD(net.trainable_params(), lr)
    net_with_loss = NetWithLossCh8(net, loss)
    train = TrainCh8(net_with_loss, optim, 1)

    predict = lambda prefix: predict_ch8(prefix, 50, net, vocab)

    state = net.begin_state(train_iter.batch_size)
    for epoch in range(num_epochs):
        ppl, speed = train_epoch_ch8(
            train, state, train_iter)
        if (epoch + 1) % 10 == 0:
            print(predict('time traveller'))
            animator.add(epoch + 1, [ppl])

    print(f'困惑度 {ppl:.1f}, {speed:.1f} 词元/秒')
    print(predict('time traveller'))
    print(predict('traveller'))

class RNNModel(nn.Cell):
    """循环神经网络模型。"""
    def __init__(self, rnn_layer, vocab_size, num_hiddens, **kwargs):
        super(RNNModel, self).__init__(**kwargs)
        self.rnn = rnn_layer
        self.vocab_size = vocab_size
        self.num_hiddens = num_hiddens
        if not self.rnn.bidirectional:
            self.num_directions = 1
            self.linear = nn.Dense(self.num_hiddens, self.vocab_size)
        else:
            self.num_directions = 2
            self.linear = nn.Dense(self.num_hiddens * 2, self.vocab_size)
        self.on_value = Tensor(1.0, mindspore.float32)
        self.off_value = Tensor(0.0, mindspore.float32)

    def construct(self, inputs, state):
        X = ops.OneHot()(inputs.T, self.vocab_size, self.on_value, self.off_value)
        Y, state = self.rnn(X, state)
        output = self.linear(Y.reshape((-1, Y.shape[-1])))
        return output, state

    def begin_state(self, batch_size=1):
        if not isinstance(self.rnn, nn.LSTM):
            return  mnp.zeros((self.num_directions * self.rnn.num_layers,
                                 batch_size, self.num_hiddens))
        else:
            return (mnp.zeros((
                        self.num_directions * self.rnn.num_layers,
                        batch_size, self.num_hiddens)),
                    mnp.zeros((
                        self.num_directions * self.rnn.num_layers,
                        batch_size, self.num_hiddens)))

class RNNModelScratch(nn.Cell):
    """从零开始实现的循环神经网络模型"""
    def __init__(self, vocab_size, num_hiddens,
                 get_params, init_state, forward_fn):
        super().__init__()
        self.vocab_size, self.num_hiddens = vocab_size, num_hiddens
        self.params = get_params(vocab_size, num_hiddens)
        self.init_state, self.forward_fn = init_state, forward_fn
        self.on_value = Tensor(1.0, mindspore.float32)
        self.off_value = Tensor(0.0, mindspore.float32)

    def construct(self, X, state):
        X = ops.OneHot()(X.T, self.vocab_size, self.on_value, self.off_value)
        return self.forward_fn(X, state, self.params)

    def begin_state(self, batch_size):
        return self.init_state(batch_size, self.num_hiddens)

class Encoder(nn.Cell):
    """编码器-解码器架构的基本编码器接口"""
    def __init__(self, **kwargs):
        super(Encoder, self).__init__(**kwargs)

    def construct(self, X):
        raise NotImplementedError

class Decoder(nn.Cell):
    """编码器-解码器架构的基本解码器接口"""
    def __init__(self, **kwargs):
        super(Decoder, self).__init__(**kwargs)

    def init_state(self, enc_outputs):
        raise NotImplementedError

    def construct(self, X, state):
        raise NotImplementedError

class EncoderDecoder(nn.Cell):
    """编码器-解码器架构的基类"""
    def __init__(self, encoder, decoder, **kwargs):
        super(EncoderDecoder, self).__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder

    def construct(self, enc_X, dec_X, X_valid_len):
        enc_outputs = self.encoder(enc_X, X_valid_len)
        dec_state = self.decoder.init_state(enc_outputs, X_valid_len)
        dec_outputs, state, _ = self.decoder(dec_X, dec_state)
        return dec_outputs, state

def read_data_nmt():
    """载入“英语－法语”数据集"""
    data_dir = download_extract('fra-eng')
    with open(os.path.join(data_dir, 'fra.txt'), 'r',
             encoding='utf-8') as f:
        return f.read()

def tokenize_nmt(text, num_examples=None):
    """词元化“英语－法语”数据数据集"""
    source, target = [], []
    for i, line in enumerate(text.split('\n')):
        if num_examples and i > num_examples:
            break
        parts = line.split('\t')
        if len(parts) == 2:
            source.append(parts[0].split(' '))
            target.append(parts[1].split(' '))
    return source, target

def preprocess_nmt(text):
    """预处理“英语－法语”数据集。"""
    def no_space(char, prev_char):
        return char in set(',.!?') and prev_char != ' '

    text = text.replace('\u202f', ' ').replace('\xa0', ' ').lower()
    out = [' ' + char if i > 0 and no_space(char, text[i - 1]) else char
           for i, char in enumerate(text)]
    return ''.join(out)

def truncate_pad(line, num_steps, padding_token):
    """截断或填充文本序列"""
    if len(line) > num_steps:
        return line[:num_steps]
    return line + [padding_token] * (num_steps - len(line))

def build_array_nmt(lines, vocab, num_steps):
    """将机器翻译的文本序列转换成小批量"""
    lines = [vocab[l] for l in lines]
    lines = [l + [vocab['<eos>']] for l in lines]
    array = np.array([truncate_pad(l, num_steps, vocab['<pad>']) for l in lines])
    valid_len = (array != vocab['<pad>']).astype(np.int32).sum(1)
    return array, valid_len

def load_data_nmt(batch_size, num_steps, num_examples=600):
    """返回翻译数据集的迭代器和词表"""
    text = preprocess_nmt(read_data_nmt())
    source, target = tokenize_nmt(text, num_examples)
    src_vocab = Vocab(source, min_freq=2,
                          reserved_tokens=['<pad>', '<bos>', '<eos>'])
    tgt_vocab = Vocab(target, min_freq=2,
                          reserved_tokens=['<pad>', '<bos>', '<eos>'])
    src_array, src_valid_len = build_array_nmt(source, src_vocab, num_steps)
    tgt_array, tgt_valid_len = build_array_nmt(target, tgt_vocab, num_steps)
    data_arrays = (src_array, src_valid_len, tgt_array, tgt_valid_len)
    data_iter = load_array(data_arrays, batch_size)
    return data_iter, src_vocab, tgt_vocab

def show_heatmaps(matrices, xlabel, ylabel, titles=None, figsize=(2.5, 2.5),
                  cmap='Reds'):
    """Show heatmaps of matrices.
    Defined in :numref:`sec_attention-cues`"""
    use_svg_display()
    num_rows, num_cols = len(matrices), len(matrices[0])
    fig, axes = plt.subplots(num_rows, num_cols, figsize=figsize,
                                 sharex=True, sharey=True, squeeze=False)
    for i, (row_axes, row_matrices) in enumerate(zip(axes, matrices)):
        for j, (ax, matrix) in enumerate(zip(row_axes, row_matrices)):
            pcm = ax.imshow(matrix.asnumpy(), cmap=cmap)
            if i == num_rows - 1:
                ax.set_xlabel(xlabel)
            if j == 0:
                ax.set_ylabel(ylabel)
            if titles:
                ax.set_title(titles[j])
    fig.colorbar(pcm, ax=axes, shrink=0.6)

class Seq2SeqEncoder(Encoder):
    """用于序列到序列学习的循环神经网络编码器"""
    def __init__(self, vocab_size, embed_size, num_hiddens, num_layers,
                 dropout=0., **kwargs):
        super(Seq2SeqEncoder, self).__init__(**kwargs)
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.rnn = nn.GRU(embed_size, num_hiddens, num_layers,
                          dropout=dropout)
        self.num_hiddens = num_hiddens
        self.num_layers = num_layers

    def construct(self, X, X_len=None):
        hx = mnp.zeros((self.num_layers, X.shape[0], self.num_hiddens), mindspore.float32)
        X = self.embedding(X)
        X = X.transpose(1, 0, 2)
        output, state = self.rnn(X, hx)
        return output, state

class MaskedSoftmaxCELoss(nn.Cell):
    """带遮蔽的softmax交叉熵损失函数"""
    def __init__(self):
        super().__init__()
        self.softmax_ce_loss = nn.SoftmaxCrossEntropyWithLogits(sparse=True)

    def construct(self, pred, label, valid_len):
        weights = mnp.ones_like(label)
        weights = sequence_mask(weights, valid_len)
        unweighted_loss = self.softmax_ce_loss(pred.view(-1, pred.shape[-1]), label.view(-1))
        weighted_loss = (unweighted_loss.view(label.shape) * weights).mean(axis=1)
        return weighted_loss.sum()

class NetWithLossCh8_Seq2seq(nn.Cell):
    def __init__(self, network, loss):
        super().__init__()
        self.network = network
        self.loss = loss

    def construct(self, *inputs):
        y_hat, state = self.network(*inputs[:-2])
        loss = self.loss(y_hat, inputs[-2], inputs[-1])
        return loss

def train_seq2seq(net, data_iter, lr, num_epochs, tgt_vocab):
    """训练序列到序列模型"""

    optimizer = nn.Adam(net.trainable_params(), lr)
    loss = MaskedSoftmaxCELoss()
    net_with_loss = NetWithLossCh8_Seq2seq(net, loss)
    train = TrainCh8(net_with_loss, optimizer, 1)
    animator = Animator(xlabel='epoch', ylabel='loss',
                     xlim=[10, num_epochs])
    for epoch in range(num_epochs):
        timer = Timer()
        metric = Accumulator(2)
        for batch in data_iter:
            X, X_valid_len, Y, Y_valid_len = [x.astype(mindspore.int32) for x in batch]
            bos = mindspore.Tensor([tgt_vocab['<bos>']] * Y.shape[0], mindspore.int32).reshape(-1, 1)
            dec_input = mnp.concatenate([bos, Y[:, :-1]], 1)
            l = train(X, dec_input, X_valid_len, Y, Y_valid_len)
            num_tokens = Y_valid_len.sum()
            metric.add(l.sum().asnumpy(), num_tokens.asnumpy())
        if (epoch + 1) % 10 == 0:
            animator.add(epoch + 1, (metric[0] / metric[1],))
    print(f'loss {metric[0] / metric[1]:.3f}, {metric[1] / timer.stop():.1f} '
        f'tokens/sec')

def bleu(pred_seq, label_seq, k):
    """计算BLEU"""
    pred_tokens, label_tokens = pred_seq.split(' '), label_seq.split(' ')
    len_pred, len_label = len(pred_tokens), len(label_tokens)
    score = math.exp(min(0, 1 - len_label / len_pred))
    for n in range(1, k + 1):
        num_matches, label_subs = 0, collections.defaultdict(int)
        for i in range(len_label - n + 1):
            label_subs[''.join(label_tokens[i: i + n])] += 1
        for i in range(len_pred - n + 1):
            if label_subs[''.join(pred_tokens[i: i + n])] > 0:
                num_matches += 1
                label_subs[''.join(pred_tokens[i: i + n])] -= 1
        score *= math.pow(num_matches / (len_pred - n + 1), math.pow(0.5, n))
    return score

def predict_seq2seq(net, src_sentence, src_vocab, tgt_vocab, num_steps, save_attention_weights=False):
    """序列到序列模型的预测"""
    net.set_train(False)
    src_tokens = src_vocab[src_sentence.lower().split(' ')] + [
        src_vocab['<eos>']]
    enc_valid_len = mindspore.Tensor([len(src_tokens)], mindspore.int32)
    src_tokens = truncate_pad(src_tokens, num_steps, src_vocab['<pad>'])
    enc_X = mnp.expand_dims(mindspore.Tensor(src_tokens, mindspore.int32), 0)
    enc_outputs = net.encoder(enc_X, enc_valid_len)
    dec_state = net.decoder.init_state(enc_outputs, enc_valid_len)
    dec_X = mnp.expand_dims(mindspore.Tensor([tgt_vocab['<bos>']], mindspore.int32), 0)
    output_seq, attention_weight_seq = [], []
    for _ in range(num_steps):
        Y, dec_state, attention_weights = net.decoder(dec_X, dec_state)
        dec_X = Y.argmax(axis=2)
        pred = int(dec_X.squeeze(0).asnumpy())
        if save_attention_weights:
            attention_weight_seq.append(attention_weights)
        if pred == tgt_vocab['<eos>']:
            break
        output_seq.append(pred)
    return ' '.join(tgt_vocab.to_tokens(output_seq)), attention_weight_seq

def sequence_mask(X, valid_len, value=0):
    """在序列中屏蔽不相关的项"""
    maxlen = X.shape[1]
    mask = mnp.arange((maxlen), dtype=mindspore.int32)[None, :] < valid_len[:, None]
    X[~mask] = value
    return X

def masked_softmax(X, valid_lens):
    """通过在最后一个轴上掩蔽元素来执行 softmax 操作"""
    if valid_lens is None:
        return ops.Softmax(-1)(X)
    else:
        shape = X.shape
        if valid_lens.ndim == 1:
            valid_lens = mnp.repeat(valid_lens, shape[1])
        else:
            valid_lens = valid_lens.reshape(-1)
        X = sequence_mask(X.reshape(-1, shape[-1]), valid_lens,
                              value=-1e6)
        return ops.Softmax(-1)(X.reshape(shape))

class AdditiveAttention(nn.Cell):
    """加性注意力"""
    def __init__(self, key_size, query_size, num_hiddens, dropout, **kwargs):
        super(AdditiveAttention, self).__init__(**kwargs)
        self.W_k = nn.Dense(key_size, num_hiddens, has_bias=False)
        self.W_q = nn.Dense(query_size, num_hiddens, has_bias=False)
        self.w_v = nn.Dense(num_hiddens, 1, has_bias=False)
        self.dropout = nn.Dropout(1 - dropout)

    def construct(self, queries, keys, values, valid_lens):
        queries, keys = self.W_q(queries), self.W_k(keys)
        features = mnp.expand_dims(queries, 2) + mnp.expand_dims(keys, 1)
        features = mnp.tanh(features)
        scores = self.w_v(features).squeeze(-1)
        attention_weights = masked_softmax(scores, valid_lens)
        outputs = ops.BatchMatMul()(self.dropout(attention_weights), values)
        return outputs, attention_weights

class DotProductAttention(nn.Cell):
    """缩放点积注意力"""
    def __init__(self, dropout, **kwargs):
        super(DotProductAttention, self).__init__(**kwargs)
        self.dropout = nn.Dropout(1 - dropout)

    def construct(self, queries, keys, values, valid_lens=None):
        d = queries.shape[-1]
        scores = ops.BatchMatMul()(queries, keys.swapaxes(1,2)) / ops.Sqrt()(ops.ScalarToTensor()(d, mindspore.float32))

        attention_weights = masked_softmax(scores, valid_lens)
        outputs = ops.BatchMatMul()(self.dropout(attention_weights), values)
        return outputs, attention_weights

def transpose_qkv(X, num_heads):
    """为了多注意力头的并行计算而变换形状。"""
    X = X.reshape(X.shape[0], X.shape[1], num_heads, -1)

    X = X.transpose(0, 2, 1, 3)

    return X.reshape(-1, X.shape[2], X.shape[3])

def transpose_output(X, num_heads):
    """逆转 `transpose_qkv` 函数的操作。"""
    X = X.reshape(-1, num_heads, X.shape[1], X.shape[2])
    X = X.transpose(0, 2, 1, 3)
    return X.reshape(X.shape[0], X.shape[1], -1)

class MultiHeadAttention(nn.Cell):
    """多头注意力"""
    def __init__(self, key_size, query_size, value_size, num_hiddens,
                 num_heads, dropout, has_bias=False, **kwargs):
        super(MultiHeadAttention, self).__init__(**kwargs)
        self.num_heads = num_heads
        self.attention = DotProductAttention(dropout)
        self.W_q = Dense(query_size, num_hiddens, has_bias=has_bias)
        self.W_k = Dense(key_size, num_hiddens, has_bias=has_bias)
        self.W_v = Dense(value_size, num_hiddens, has_bias=has_bias)
        self.W_o = Dense(num_hiddens, num_hiddens, has_bias=has_bias)

    def construct(self, queries, keys, values, valid_lens):
        queries = transpose_qkv(self.W_q(queries), self.num_heads)
        keys = transpose_qkv(self.W_k(keys), self.num_heads)
        values = transpose_qkv(self.W_v(values), self.num_heads)

        if valid_lens is not None:
            valid_lens = mnp.repeat(
                valid_lens, repeats=self.num_heads, axis=0)

        output, attention_weights = self.attention(queries, keys, values, valid_lens)

        output_concat = transpose_output(output, self.num_heads)
        return self.W_o(output_concat), attention_weights

class PositionalEncoding(nn.Cell):
    """位置编码"""
    def __init__(self, num_hiddens, dropout, max_len=1000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(1 - dropout)
        self.P = mnp.zeros((1, max_len, num_hiddens))
        X = mnp.arange(max_len, dtype=mindspore.float32).reshape(
            -1, 1) / mnp.power(mindspore.Tensor(10000, mindspore.int32), mnp.arange(
            0, num_hiddens, 2, dtype=mindspore.float32) / num_hiddens)
        self.P[:, :, 0::2] = mnp.sin(X)
        self.P[:, :, 1::2] = mnp.cos(X)

    def construct(self, X):
        X = X + self.P[:, :X.shape[1], :]
        return self.dropout(X)

class Dense(nn.Dense):
    def __init__(self, in_channels, out_channels, has_bias=True, activation=None):
        super().__init__(in_channels, out_channels, weight_init='xavier_uniform', bias_init='zeros', has_bias=has_bias, activation=activation)
        self.reset_parameters()

    def reset_parameters(self):
        if self.has_bias:
            fan_in, _ = _calculate_fan_in_and_fan_out(self.weight.shape)
            bound = 1 / math.sqrt(fan_in)
            self.bias.set_data(initializer(Uniform(bound), [self.out_channels]))

class Embedding(nn.Embedding):
    def __init__(self, vocab_size, embedding_size, use_one_hot=False, embedding_table='normal', dtype=mindspore.float32, padding_idx=None):
        if embedding_table == 'normal':
            embedding_table = Normal(1.0)
        super().__init__(vocab_size, embedding_size, use_one_hot, embedding_table, dtype, padding_idx)
    @classmethod
    def from_pretrained_embedding(cls, embeddings:Tensor, freeze=True, padding_idx=None):
        rows, cols = embeddings.shape
        embedding = cls(rows, cols, embedding_table=embeddings, padding_idx=padding_idx)
        embedding.embedding_table.requires_grad = not freeze
        return embedding

def train_transformer(net, data_iter, lr, num_epochs, tgt_vocab):
    """训练序列到序列模型"""

    optimizer = nn.Adam(net.trainable_params(), lr)
    loss = MaskedSoftmaxCELoss()
    net_with_loss = NetWithLossCh8_Seq2seq(net, loss)
    train = TrainCh8(net_with_loss, optimizer, 1)
    animator = Animator(xlabel='epoch', ylabel='loss',
                     xlim=[10, num_epochs])
    for epoch in range(num_epochs):
        timer = Timer()
        metric = Accumulator(2)
        for batch in data_iter:
            X, X_valid_len, Y, Y_valid_len = [x.astype(mindspore.int32) for x in batch]
            bos = mindspore.Tensor([tgt_vocab['<bos>']] * Y.shape[0], mindspore.int32).reshape(-1, 1)
            dec_input = mnp.concatenate([bos, Y[:, :-1]], 1)
            batch_size, num_steps = X.shape
            dec_valid_lens = mnp.tile(mnp.arange(1, num_steps + 1), (batch_size, 1))
            l = train(X, dec_input, X_valid_len, dec_valid_lens, Y, Y_valid_len)
            num_tokens = Y_valid_len.sum()
            metric.add(l.sum().asnumpy(), num_tokens.asnumpy())
        if (epoch + 1) % 10 == 0:
            animator.add(epoch + 1, (metric[0] / metric[1],))
    print(f'loss {metric[0] / metric[1]:.3f}, {metric[1] / timer.stop():.1f} '
        f'tokens/sec')

def predict_transformer(net, src_sentence, src_vocab, tgt_vocab, num_steps, save_attention_weights=False):
    """序列到序列模型的预测"""
    net.set_train(False)
    src_tokens = src_vocab[src_sentence.lower().split(' ')] + [
        src_vocab['<eos>']]
    enc_valid_len = mindspore.Tensor([len(src_tokens)], mindspore.int32)
    src_tokens = truncate_pad(src_tokens, num_steps, src_vocab['<pad>'])
    enc_X = mnp.expand_dims(mindspore.Tensor(src_tokens, mindspore.int32), 0)
    enc_outputs, encoder_attention_weight = net.encoder(enc_X, enc_valid_len)
    dec_state = net.decoder.init_state(enc_outputs, enc_valid_len)
    dec_X = mnp.expand_dims(mindspore.Tensor([tgt_vocab['<bos>']], mindspore.int32), 0)
    output_seq, attention_weight_seq = [], []
    for _ in range(num_steps):
        Y, dec_state, attention_weights = net.decoder(dec_X, dec_state, None)
        dec_X = Y.argmax(axis=2)
        pred = int(dec_X.squeeze(0).asnumpy())
        if save_attention_weights:
            attention_weight_seq.append(attention_weights)
        if pred == tgt_vocab['<eos>']:
            break
        output_seq.append(pred)
    return ' '.join(tgt_vocab.to_tokens(output_seq)), attention_weight_seq, encoder_attention_weight

def train_2d(trainer, steps=20, f_grad=None):
    """Optimize a 2D objective function with a customized trainer.

    Defined in :numref:`subsec_gd-learningrate`"""
    # `s1` and `s2` are internal state variables that will be used later
    x1, x2, s1, s2 = -5, -2, 0, 0
    results = [(x1, x2)]
    for i in range(steps):
        if f_grad:
            x1, x2, s1, s2 = trainer(x1, x2, s1, s2, f_grad)
        else:
            x1, x2, s1, s2 = trainer(x1, x2, s1, s2)
        results.append((x1, x2))
    print(f'epoch {i + 1}, x1: {float(x1):f}, x2: {float(x2):f}')
    return results

def show_trace_2d(f, results):
    """Show the trace of 2D variables during optimization.

    Defined in :numref:`subsec_gd-learningrate`"""
    d2l.set_figsize()
    d2l.plt.plot(*zip(*results), '-o', color='#ff7f0e')
    x1, x2 = d2l.meshgrid((d2l.arange(-5.5, 1.0, 0.1),
                          d2l.arange(-3.0, 1.0, 0.1)))
    d2l.plt.contour(x1, x2, f(x1, x2), colors='#1f77b4')
    d2l.plt.xlabel('x1')
    d2l.plt.ylabel('x2')

abs = ops.abs
arange = ops.arange
concat = ops.concat
float32 = mindspore.float32
ones = ops.ones
zeros = ops.zeros
matmul = ops.matmul
stack = ops.stack
sin = ops.sin
cos = ops.cos
sinh = ops.sinh
cosh = ops.cosh
tanh = ops.tanh
exp = ops.exp
meshgrid = ops.meshgrid
linspace = ops.linspace
zeros_like = ops.zeros_like
sqrt = ops.sqrt
log = ops.log
maximum = ops.maximum
relu = ops.relu
sigmoid = ops.sigmoid
norm = ops.norm
pow = lambda x, y: ops.pow(x, y)
clip_by_value = lambda x, clip_value_min, clip_value_max: ops.clip_by_value(x, clip_value_min, clip_value_max)
uniform = lambda shape, minval, maxval: ops.uniform(shape, tensor(minval), tensor(maxval), dtype=float32)
rand = lambda size, *args: ops.rand(size, dtype=float32)
randn = lambda size, *args: ops.randn(size, dtype=float32)
tensor = lambda x: mindspore.Tensor(x, dtype=mindspore.float32)
normal = lambda shape, mean, stddev, *args : ops.normal(shape, tensor(mean), tensor(stddev), *args)
reduce_sum = lambda x, *args, **kwargs: x.sum(*args, **kwargs)
reshape = lambda x, *args, **kwargs: x.reshape(*args, **kwargs)
transpose = lambda x, *args, **kwargs: x.t(*args, **kwargs)
