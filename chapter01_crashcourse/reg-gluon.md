# 使用Gluon的正则化

【以下代码需改进且模块化，文字解释待加】

## 过拟合


```python
from mxnet import ndarray as nd
from mxnet import autograd
from mxnet import gluon
```


```python
num_train = 6
num_test = 1000

true_order = 5
true_w = [0.001, -0.002, 0.003, -0.004, 0.005]
true_b = 0.005

x = nd.random_normal(shape=(num_train + num_test, 1))
X = x
y = true_w[0] * X[:, 0]

for i in range(1, true_order):
    X = nd.concat(X, nd.power(x, i + 1))
    y += true_w[i] * X[:, i]
y += true_b + .01 * nd.random_normal(shape=y.shape)

X_train, X_test = X[:num_train, :], X[num_train:, :]
y_train, y_test = y[:num_train], y[num_train:]

batch_size = 10
dataset_train = gluon.data.ArrayDataset(X_train, y_train)
data_iter_train = gluon.data.DataLoader(dataset_train, batch_size, shuffle=True)
```


```python
net = gluon.nn.Sequential()
dense = gluon.nn.Dense(1)
net.add(dense)
square_loss = gluon.loss.L2Loss()
net.initialize()
```


```python
trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': 0.004})
epochs = 10
batch_size = 10
for e in range(epochs):
    total_loss = 0
    for data, label in data_iter_train:
        with autograd.record():
            output = net(data)
            loss = square_loss(output, label)
        loss.backward()
        trainer.step(batch_size)
        total_loss += nd.sum(loss).asscalar()
    print("Epoch %d, train loss: %f" % (e, total_loss / num_train))

loss_test = nd.sum(square_loss(net(X_test), y_test)).asscalar() / num_test
print("Test loss: %f" % loss_test)
```


    Epoch 0, train loss: 0.337313
    Epoch 1, train loss: 0.061348
    Epoch 2, train loss: 0.013433
    Epoch 3, train loss: 0.005093
    Epoch 4, train loss: 0.003621
    Epoch 5, train loss: 0.003341
    Epoch 6, train loss: 0.003268
    Epoch 7, train loss: 0.003231
    Epoch 8, train loss: 0.003201
    Epoch 9, train loss: 0.003172
    Test loss: 1.092651





```python
print(true_w, true_b)
print(dense.weight.data()[0], dense.bias.data()[0])
```


    ([0.001, -0.002, 0.003, -0.004, 0.005], 0.005)
    (
    [ 0.01938676  0.05800602 -0.00466745  0.07212976 -0.03435896]
    <NDArray 5 @cpu(0)>, 
    [-0.00080924]
    <NDArray 1 @cpu(0)>)



## 使用``Gluon``的正则化

我们通过优化算法的``wd``参数 (weight decay)实现对模型的正则化。这相当于$L_2$正则化。


```python
net.collect_params().initialize(force_reinit=True)
trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': 0.004,
                                                     'wd': 150.0})
epochs = 10
batch_size = 10
for e in range(epochs):
    total_loss = 0
    for data, label in data_iter_train:
        with autograd.record():
            output = net(data)
            loss = square_loss(output, label)
        loss.backward()
        trainer.step(batch_size)
        total_loss += nd.sum(loss).asscalar()
    print("Epoch %d, train loss: %f" % (e, total_loss / num_train))

loss_test = nd.sum(square_loss(net(X_test), y_test)).asscalar() / num_test
print("Test loss: %f" % loss_test)
```


    Epoch 0, train loss: 0.458372
    Epoch 1, train loss: 0.428181
    Epoch 2, train loss: 0.488526
    Epoch 3, train loss: 0.458000
    Epoch 4, train loss: 0.520910
    Epoch 5, train loss: 0.489906
    Epoch 6, train loss: 0.555476
    Epoch 7, train loss: 0.524010
    Epoch 8, train loss: 0.592364
    Epoch 9, train loss: 0.560459
    Test loss: 0.832598





```python
print(true_w, true_b)
print(dense.weight.data()[0], dense.bias.data()[0])
```


    ([0.001, -0.002, 0.003, -0.004, 0.005], 0.005)
    (
    [-0.00166943 -0.0036052  -0.00789909 -0.01736431 -0.03831115]
    <NDArray 5 @cpu(0)>, 
    [-0.00079609]
    <NDArray 1 @cpu(0)>)



## 结论



## 练习



**吐槽和讨论欢迎点[这里](https://discuss.gluon.ai/t/topic/743)**
