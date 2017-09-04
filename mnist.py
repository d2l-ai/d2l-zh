from mxnet import gluon
from mxnet import nd

def transform(data, label):
    # change data from height x weight x channel to channel x height x weight
    return nd.transpose(data.astype('float32'), (2,0,1))/255, label.astype('float32')

mnist_train = gluon.data.vision.FashionMNIST(
    train=True, transform=transform)
mnist_test = gluon.data.vision.FashionMNIST(
    train=False, transform=transform)

def load_data(batch_size):
    train_data = gluon.data.DataLoader(
        mnist_train, batch_size, shuffle=True)
    test_data = gluon.data.DataLoader(
        mnist_test, batch_size, shuffle=False)
    return (train_data, test_data)
