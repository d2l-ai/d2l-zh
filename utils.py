from mxnet import gluon
from mxnet import autograd
from mxnet import nd
import mxnet as mx

def SGD(params, lr):
    for param in params:
        param[:] = param - lr * param.grad

def accuracy(output, label):
    return nd.mean(output.argmax(axis=1)==label).asscalar()

def evaluate_accuracy(data_iterator, net, ctx=mx.cpu()):
    acc = 0.
    for data, label in data_iterator:
        output = net(data.as_in_context(ctx))
        acc += accuracy(output, label.as_in_context(ctx))
    return acc / len(data_iterator)

def transform_mnist(data, label):
    # change data from height x weight x channel to channel x height x weight
    return nd.transpose(data.astype('float32'), (2,0,1))/255, label.astype('float32')


def load_data_fashion_mnist(batch_size, transform=transform_mnist):
    """download the fashion mnist dataest and then load into memory"""
    mnist_train = gluon.data.vision.FashionMNIST(
        train=True, transform=transform)
    mnist_test = gluon.data.vision.FashionMNIST(
        train=False, transform=transform)
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


def train(train_data, test_data, net, loss, trainer, ctx, num_epochs, print_batches=None):
    """Train a network"""
    for epoch in range(num_epochs):
        train_loss = 0.
        train_acc = 0.
        batch = 0
        for data, label in train_data:
            label = label.as_in_context(ctx)
            with autograd.record():
                output = net(data.as_in_context(ctx))
                L = loss(output, label)
            L.backward()

            trainer.step(data.shape[0])

            train_loss += nd.mean(L).asscalar()
            train_acc += accuracy(output, label)

            batch += 1
            if print_batches and batch % print_batches == 0:
                print("Batch %d. Loss: %f, Train acc %f" % (
                    batch, train_loss/batch, train_acc/batch
                ))

        test_acc = evaluate_accuracy(test_data, net, ctx)
        print("Epoch %d. Loss: %f, Train acc %f, Test acc %f" % (
            epoch, train_loss/batch, train_acc/batch, test_acc
        ))
