from mxnet import gluon
from mxnet import autograd
from mxnet import nd
from mxnet import image
import mxnet as mx

def load_data_fashion_mnist(batch_size, resize=None):
    """download the fashion mnist dataest and then load into memory"""
    def transform_mnist(data, label):
        if resize:
            # resize to resize x resize
            data = image.imresize(data, resize, resize)
        # change data from height x weight x channel to channel x height x weight
        return nd.transpose(data.astype('float32'), (2,0,1))/255, label.astype('float32')
    mnist_train = gluon.data.vision.FashionMNIST(
        train=True, transform=transform_mnist)
    mnist_test = gluon.data.vision.FashionMNIST(
        train=False, transform=transform_mnist)
    train_data = gluon.data.DataLoader(
        mnist_train, batch_size, shuffle=True)
    test_data = gluon.data.DataLoader(
        mnist_test, batch_size, shuffle=False)
    return (train_data, test_data)

def try_gpu():
    """If GPU is available, return mx.gpu(0); else return mx.cpu()"""
    try:
        ctx = mx.gpu()
        _ = nd.zeros((1,), ctx=ctx)
    except:
        ctx = mx.cpu()
    return ctx

def SGD(params, lr):
    for param in params:
        param[:] = param - lr * param.grad

def accuracy(output, label):
    return nd.mean(output.argmax(axis=1)==label).asscalar()

def _get_batch(batch, ctx):
    """return data and label on ctx"""
    if isinstance(batch, mx.io.DataBatch):
        data = batch.data[0]
        label = batch.label[0]
    else:
        data, label = batch
    return data.as_in_context(ctx), label.as_in_context(ctx)

def evaluate_accuracy(data_iterator, net, ctx=mx.cpu()):
    acc = 0.
    if isinstance(data_iterator, mx.io.MXDataIter):
        data_iterator.reset()
    for i, batch in enumerate(data_iterator):
        data, label = _get_batch(batch, ctx)
        output = net(data)
        acc += accuracy(output, label)
    return acc / (i+1)

def train(train_data, test_data, net, loss, trainer, ctx, num_epochs, print_batches=None):
    """Train a network"""
    for epoch in range(num_epochs):
        train_loss = 0.
        train_acc = 0.
        if isinstance(train_data, mx.io.MXDataIter):
            train_data.reset()
        for i, batch in enumerate(train_data):
            data, label = _get_batch(batch, ctx)
            with autograd.record():
                output = net(data)
                L = loss(output, label)
            L.backward()

            trainer.step(data.shape[0])

            train_loss += nd.mean(L).asscalar()
            train_acc += accuracy(output, label)

            n = i + 1
            if print_batches and n % print_batches == 0:
                print("Batch %d. Loss: %f, Train acc %f" % (
                    n, train_loss/n, train_acc/n
                ))

        test_acc = evaluate_accuracy(test_data, net, ctx)
        print("Epoch %d. Loss: %f, Train acc %f, Test acc %f" % (
            epoch, train_loss/n, train_acc/n, test_acc
        ))
