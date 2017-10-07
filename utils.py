from mxnet import gluon
from mxnet import autograd
from mxnet import nd
from mxnet import image
import mxnet as mx
import itchat
import threading

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

#define threading lock
lock = threading.Lock()
running = False
def lock_start():
    global lock, running
    print('wait for lock')
    with lock:
        running = True
        run_state = running
    print('start')
    return run_state

def chat_inf(wechat_name,epoch,train_loss,train_acc,train_data,test_acc):
    itchat.send("Epoch %d.\nLoss: %f\nTrain acc %f\nTest acc %f" % (epoch, train_loss/len(train_data),train_acc/len(train_data), test_acc), wechat_name)
    with lock:
        run_state = running
    return run_state

def lock_end(wechat_name):
    global lock, running
    print('op is finished!')
    itchat.send('op is finished!',wechat_name)   
    with lock:
        running = False
	
#define chat_supervise handle
batch_size = 256
learning_rate = 0.5
training_iters = 2
def chat_supervisor(target_trainer):
    @itchat.msg_register([itchat.content.TEXT])
    def chat_trigger(msg):
        global lock, running, learning_rate, training_iters, batch_size
        if msg['Text'] == u'开始':
            print('Starting')
            with lock:
                run_state = running
            if not run_state:
                try:
                    threading.Thread(target=target_trainer, args=(msg['FromUserName'], (learning_rate, training_iters, batch_size))).start()
                except:
                    msg.reply('Running')
        elif msg['Text'] == u'停止':
            print('Stopping')
    
            with lock:
                running = False
    
        elif msg['Text'] == u'参数':
            itchat.send('lr=%f, ti=%d, bs=%d'%(learning_rate, training_iters, batch_size),msg['FromUserName'])
        else:
            try:
                param = msg['Text'].split()
                key, value = param
                print(key, value)
                if key == 'lr':
                    learning_rate = float(value)
                elif key == 'ti':
                    training_iters = int(value)
                elif key == 'bs':
                    batch_size = int(value)
            except:
                pass
        return learning_rate,training_iters,batch_size
