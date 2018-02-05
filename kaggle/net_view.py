import mxnet as mx
import numpy as np
import cv2
import matplotlib.pyplot as plt
import logging

# 启动日志
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)

# 定义一个网络
data = mx.symbol.Variable('data')
fc1 = mx.symbol.FullyConnected(data=data, name='fc1', num_hidden=128)
act1 = mx.symbol.Activation(data=fc1, name='relu1', act_type='relu')
fc2 = mx.symbol.FullyConnected(data=act1, name='fc2', num_hidden=64)
act2 = mx.symbol.Activation(data=fc2, name='relu2', act_type='relu')
fc3 = mx.symbol.FullyConnected(data=act2, name='fc3', num_hidden=10)
mlp = mx.symbol.SoftmaxOutput(data=fc3, name='softmax')

# 可视化网络
mx.viz.plot_network(mlp).view()
