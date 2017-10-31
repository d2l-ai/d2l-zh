import pandas as pd
import numpy as np

train = pd.read_csv("../data/kaggle_house_pred_train.csv")
test = pd.read_csv("../data/kaggle_house_pred_test.csv")
all_X = pd.concat((train.loc[:, 'MSSubClass':'SaleCondition'],
                      test.loc[:, 'MSSubClass':'SaleCondition']))

all_X = pd.get_dummies(all_X, dummy_na=True)
all_X = all_X.fillna(all_X.mean())
num_train = train.shape[0]

X_train = all_X[:num_train].as_matrix()
X_test = all_X[num_train:].as_matrix()
y_train = train.SalePrice.as_matrix()

from mxnet import ndarray as nd
from mxnet import autograd
from mxnet import gluon
import time

start = time.time()


X_train = nd.array(X_train)
y_train = nd.array(y_train)
y_train.reshape((num_train, 1))

X_test = nd.array(X_test)

square_loss = gluon.loss.L2Loss()

def get_rmse_log(net, X_train, y_train):
    num_train = X_train.shape[0]
    clipped_preds = nd.clip(net(X_train), 1, float('inf'))
    return np.sqrt(2 * nd.sum(square_loss(
        nd.log(clipped_preds), nd.log(y_train))).asscalar() / num_train)

import random
k = 5
epochs = 100
verbose_epoch = 100

def get_net(hidden_nodes):
    net = gluon.nn.Sequential()
    with net.name_scope():
        net.add(gluon.nn.Dense(hidden_nodes))
        net.add(gluon.nn.Dense(1))
    net.initialize()
    return net

import matplotlib as mpl
mpl.rcParams['figure.dpi']= 120
import matplotlib.pyplot as plt

def train(net, X_train, y_train, X_test, y_test, epochs,
          verbose_epoch, learning_rate, weight_decay):
    train_loss = []
    if X_test is not None:
#         test_loss = []
        pass
    batch_size = 100
    dataset_train = gluon.data.ArrayDataset(X_train, y_train)
    data_iter_train = gluon.data.DataLoader(
        dataset_train, batch_size,shuffle=True)
    trainer = gluon.Trainer(net.collect_params(), 'adam',
                            {'learning_rate': learning_rate,
                             'wd': weight_decay})
    net.collect_params().initialize(force_reinit=True)
    for epoch in range(epochs):
        for data, label in data_iter_train:
            with autograd.record():
                output = net(data)
                loss = square_loss(output, label)
            loss.backward()
            trainer.step(batch_size)

            cur_train_loss = get_rmse_log(net, X_train, y_train)
        if epoch > verbose_epoch:
            print("Epoch %d, train loss: %f" % (epoch, cur_train_loss))
#         train_loss.append(cur_train_loss)
        if X_test is not None:
            cur_test_loss = get_rmse_log(net, X_test, y_test)
#             test_loss.append(cur_test_loss)
#     plt.plot(train_loss)
#     plt.legend(['train'])
#     if X_test is not None:
#         plt.plot(test_loss)
#         plt.legend(['train','test'])
#     plt.show()
    if X_test is not None:
        return cur_train_loss, cur_test_loss
    else:
        return cur_train_loss


def k_fold_cross_valid(k, epochs, verbose_epoch, X_train, y_train,
                       learning_rate, weight_decay, net):
    assert k > 1
    fold_size = X_train.shape[0] // k
    train_loss_sum = 0.0
    test_loss_sum = 0.0
    for test_i in range(k):
        X_val_test = X_train[test_i * fold_size: (test_i + 1) * fold_size, :]
        y_val_test = y_train[test_i * fold_size: (test_i + 1) * fold_size]

        val_train_defined = False
        for i in range(k):
            if i != test_i:
                X_cur_fold = X_train[i * fold_size: (i + 1) * fold_size, :]
                y_cur_fold = y_train[i * fold_size: (i + 1) * fold_size]
                if not val_train_defined:
                    X_val_train = X_cur_fold
                    y_val_train = y_cur_fold
                    val_train_defined = True
                else:
                    X_val_train = nd.concat(X_val_train, X_cur_fold, dim=0)
                    y_val_train = nd.concat(y_val_train, y_cur_fold, dim=0)
        train_loss, test_loss = train(
            net, X_val_train, y_val_train, X_val_test, y_val_test,
            epochs, verbose_epoch, learning_rate, weight_decay)
        print("Train loss: %f" % train_loss)
        train_loss_sum += train_loss
        print("Test loss: %f" % test_loss)
        test_loss_sum += test_loss
    return train_loss_sum / k, test_loss_sum / k

learning_rate_range = 6
weight_decay_range = 10
hidden_min_nodes = 10
hidden_max_nodes = 100

best_learning_rate = None
best_hidden_nodes = None
best_weight_decay = None
first_time = True
min_avg_test_loss = None
for times in range(1000):
    learning_rate = random.uniform(4, learning_rate_range)
    hidden_nodes = random.randint(hidden_min_nodes, hidden_max_nodes)
    weight_decay = random.uniform(0, weight_decay_range)
    net = get_net(hidden_nodes)
    print('*'*10)
    print('loop', times + 1)
    end = time.time() - start
    print("[timecosts: %s]" % end)
    print('learing_rate', learning_rate)
    print('hidden_nodes', hidden_nodes)
    print('weight_decay', weight_decay)
    average_train_loss, average_test_loss = k_fold_cross_valid(k, epochs, verbose_epoch, X_train, y_train,
                                                               learning_rate, weight_decay, net)
    print('average_train_loss', average_train_loss)
    print('average_test_loss', average_test_loss)
    if first_time:
        min_avg_test_loss = average_test_loss
        best_learning_rate = learning_rate
        best_hidden_nodes = hidden_nodes
        best_weight_decay = weight_decay
        first_time = False
    else:
        if min_avg_test_loss > average_test_loss:
            min_avg_test_loss = average_test_loss
            best_learning_rate = learning_rate
            best_hidden_nodes = hidden_nodes
            best_weight_decay = weight_decay
print('+'*100)
print('best_learing_rate', best_learning_rate)
print('best_hidden_nodes', best_hidden_nodes)
print('best_weight_decay', best_weight_decay)
print('min_avg_test_loss', min_avg_test_loss)

def learn(epochs, verbose_epoch, X_train, y_train, test, learning_rate,
          weight_decay, hidden_nodes):
    net = get_net(hidden_nodes)
    loss = train(net, X_train, y_train, None, None, epochs, verbose_epoch,
          learning_rate, weight_decay)
    preds = net(X_test).asnumpy()
    test['SalePrice'] = pd.Series(preds.reshape(1, -1)[0])
    submission = pd.concat([test['Id'], test['SalePrice']], axis=1)
    submission.to_csv('submission.csv', index=False)
    param_series = pd.Series([learning_rate, weight_decay, hidden_nodes, loss], index=['best_learning_rate', 'best_weight_decay',
                                                                          'best_hidden_nodes', 'loss'])
    print(param_series)
    param = param_series.to_frame().reset_index()
    param = param.rename(columns= {0: 'value'})
    param.index.name = 'param'
    param.to_csv('param.csv', index=False)

learn(epochs, verbose_epoch, X_train, y_train, test, best_learning_rate,
      best_weight_decay, best_hidden_nodes)
end = time.time() - start
print("[total timecosts: %s]" % end)