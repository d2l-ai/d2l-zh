# 初始化模型参数

我们仍然用MLP这个例子来详细解释如何初始化模型参数。

```{.python .input  n=62}
from mxnet.gluon import nn
from mxnet import nd

def get_net():
    net = nn.Sequential()
    with net.name_scope():
        net.add(nn.Dense(4, activation="relu"))
        net.add(nn.Dense(2))
    return net

x = nd.random.uniform(shape=(3,5))
```

我们知道如果不`initialize()`直接跑forward，那么系统会抱怨说参数没有初始化。

```{.python .input  n=63}
import sys
try:
    net = get_net()
    print(net)
#     net.initialize()
    net(x)
except RuntimeError as err:
    sys.stderr.write(str(err))
```

```{.json .output n=63}
[
 {
  "name": "stdout",
  "output_type": "stream",
  "text": "Sequential(\n  (0): Dense(4, Activation(relu))\n  (1): Dense(2, linear)\n)\n"
 },
 {
  "name": "stderr",
  "output_type": "stream",
  "text": "Parameter sequential11_dense0_weight has not been initialized. Note that you should initialize parameters and create Trainer with Block.collect_params() instead of Block.params because the later does not include Parameters of nested child Blocks"
 }
]
```

正确的打开方式是这样

```{.python .input  n=64}
net.initialize()
net.collect_params()
```

```{.json .output n=64}
[
 {
  "data": {
   "text/plain": "sequential11_ (\n  Parameter sequential11_dense0_weight (shape=(4, 0), dtype=<class 'numpy.float32'>)\n  Parameter sequential11_dense0_bias (shape=(4,), dtype=<class 'numpy.float32'>)\n  Parameter sequential11_dense1_weight (shape=(2, 0), dtype=<class 'numpy.float32'>)\n  Parameter sequential11_dense1_bias (shape=(2,), dtype=<class 'numpy.float32'>)\n)"
  },
  "execution_count": 64,
  "metadata": {},
  "output_type": "execute_result"
 }
]
```

```{.python .input  n=65}
net(x)
```

```{.json .output n=65}
[
 {
  "data": {
   "text/plain": "\n[[ 0.00198434 -0.00195766]\n [ 0.00177498 -0.00035758]\n [ 0.00445696 -0.00439705]]\n<NDArray 3x2 @cpu(0)>"
  },
  "execution_count": 65,
  "metadata": {},
  "output_type": "execute_result"
 }
]
```

```{.python .input  n=66}
net.collect_params()
```

```{.json .output n=66}
[
 {
  "data": {
   "text/plain": "sequential11_ (\n  Parameter sequential11_dense0_weight (shape=(4, 5), dtype=<class 'numpy.float32'>)\n  Parameter sequential11_dense0_bias (shape=(4,), dtype=<class 'numpy.float32'>)\n  Parameter sequential11_dense1_weight (shape=(2, 4), dtype=<class 'numpy.float32'>)\n  Parameter sequential11_dense1_bias (shape=(2,), dtype=<class 'numpy.float32'>)\n)"
  },
  "execution_count": 66,
  "metadata": {},
  "output_type": "execute_result"
 }
]
```

## 访问模型参数

之前我们提到过可以通过`weight`和`bias`访问`Dense`的参数，他们是`Parameter`这个类：

```{.python .input  n=67}
w = net[0].weight
b = net[0].bias
print('name: ', net[0].name, '\nweight: ', w, '\nbias: ', b)
```

```{.json .output n=67}
[
 {
  "name": "stdout",
  "output_type": "stream",
  "text": "name:  sequential11_dense0 \nweight:  Parameter sequential11_dense0_weight (shape=(4, 5), dtype=<class 'numpy.float32'>) \nbias:  Parameter sequential11_dense0_bias (shape=(4,), dtype=<class 'numpy.float32'>)\n"
 }
]
```

然后我们可以通过`data`来访问参数，`grad`来访问对应的梯度

```{.python .input  n=68}
print('weight:', w.data())
print('weight gradient', w.grad())
print('bias:', b.data())
print('bias gradient', b.grad())
```

```{.json .output n=68}
[
 {
  "name": "stdout",
  "output_type": "stream",
  "text": "weight: \n[[ 0.0285444  -0.00729033 -0.05596824 -0.05606033  0.05872757]\n [-0.02067637  0.02999379 -0.00430512  0.06983858  0.04775962]\n [-0.04907724  0.05665069  0.05153765 -0.06474169 -0.04725099]\n [ 0.00116416  0.01617834 -0.04664135 -0.0526652   0.03906714]]\n<NDArray 4x5 @cpu(0)>\nweight gradient \n[[ 0.  0.  0.  0.  0.]\n [ 0.  0.  0.  0.  0.]\n [ 0.  0.  0.  0.  0.]\n [ 0.  0.  0.  0.  0.]]\n<NDArray 4x5 @cpu(0)>\nbias: \n[ 0.  0.  0.  0.]\n<NDArray 4 @cpu(0)>\nbias gradient \n[ 0.  0.  0.  0.]\n<NDArray 4 @cpu(0)>\n"
 }
]
```

我们也可以通过`collect_params`来访问Block里面所有的参数（这个会包括所有的子Block）。它会返回一个名字到对应Parameter的dict。既可以用正常`[]`来访问参数，也可以用`get()`，它不需要填写名字的前缀。

```{.python .input  n=85}
params = net.collect_params()
print(params)
print(net)
print(params['sequential11_dense0_weight'].data())
print(params.get('dense0_').data())
```

```{.json .output n=85}
[
 {
  "name": "stdout",
  "output_type": "stream",
  "text": "sequential11_ (\n  Parameter sequential11_dense0_weight (shape=(4, 5), dtype=<class 'numpy.float32'>)\n  Parameter sequential11_dense0_bias (shape=(4,), dtype=<class 'numpy.float32'>)\n  Parameter sequential11_dense1_weight (shape=(2, 4), dtype=<class 'numpy.float32'>)\n  Parameter sequential11_dense1_bias (shape=(2,), dtype=<class 'numpy.float32'>)\n)\nSequential(\n  (0): Dense(4, Activation(relu))\n  (1): Dense(2, linear)\n)\n\n[[ 0.0285444  -0.00729033 -0.05596824 -0.05606033  0.05872757]\n [-0.02067637  0.02999379 -0.00430512  0.06983858  0.04775962]\n [-0.04907724  0.05665069  0.05153765 -0.06474169 -0.04725099]\n [ 0.00116416  0.01617834 -0.04664135 -0.0526652   0.03906714]]\n<NDArray 4x5 @cpu(0)>\n\n[[ 0.0285444  -0.00729033 -0.05596824 -0.05606033  0.05872757]\n [-0.02067637  0.02999379 -0.00430512  0.06983858  0.04775962]\n [-0.04907724  0.05665069  0.05153765 -0.06474169 -0.04725099]\n [ 0.00116416  0.01617834 -0.04664135 -0.0526652   0.03906714]]\n<NDArray 4x5 @cpu(0)>\n"
 }
]
```

## 使用不同的初始函数来初始化

我们一直在使用默认的`initialize`来初始化权重（除了指定GPU `ctx`外）。它会把所有权重初始化成在`[-0.07, 0.07]`之间均匀分布的随机数。我们可以使用别的初始化方法。例如使用均值为0，方差为0.02的正态分布

```{.python .input  n=11}
from mxnet import init
params.initialize(init=init.Normal(sigma=0.02), force_reinit=True)
print(net[0].weight.data(), net[0].bias.data())
```

```{.json .output n=11}
[
 {
  "ename": "DeferredInitializationError",
  "evalue": "",
  "output_type": "error",
  "traceback": [
   "\u001b[0;31m---------------------------------------------------------------------------\u001b[0m",
   "\u001b[0;31mDeferredInitializationError\u001b[0m               Traceback (most recent call last)",
   "\u001b[0;32m<ipython-input-11-b9319b2fd7f2>\u001b[0m in \u001b[0;36m<module>\u001b[0;34m()\u001b[0m\n\u001b[1;32m      1\u001b[0m \u001b[0;32mfrom\u001b[0m \u001b[0mmxnet\u001b[0m \u001b[0;32mimport\u001b[0m \u001b[0minit\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m      2\u001b[0m \u001b[0mparams\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0minitialize\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0minit\u001b[0m\u001b[0;34m=\u001b[0m\u001b[0minit\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mNormal\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0msigma\u001b[0m\u001b[0;34m=\u001b[0m\u001b[0;36m0.02\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mforce_reinit\u001b[0m\u001b[0;34m=\u001b[0m\u001b[0;32mTrue\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0;32m----> 3\u001b[0;31m \u001b[0mprint\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mnet\u001b[0m\u001b[0;34m[\u001b[0m\u001b[0;36m0\u001b[0m\u001b[0;34m]\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mweight\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mdata\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mnet\u001b[0m\u001b[0;34m[\u001b[0m\u001b[0;36m0\u001b[0m\u001b[0;34m]\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mbias\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mdata\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m",
   "\u001b[0;32m/Users/thomas_young/miniconda3/envs/gluon/lib/python3.6/site-packages/mxnet/gluon/parameter.py\u001b[0m in \u001b[0;36mdata\u001b[0;34m(self, ctx)\u001b[0m\n\u001b[1;32m    337\u001b[0m         \u001b[0mNDArray\u001b[0m \u001b[0mon\u001b[0m \u001b[0mctx\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m    338\u001b[0m         \"\"\"\n\u001b[0;32m--> 339\u001b[0;31m         \u001b[0;32mreturn\u001b[0m \u001b[0mself\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0m_check_and_get\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mself\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0m_data\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mctx\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m\u001b[1;32m    340\u001b[0m \u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m    341\u001b[0m     \u001b[0;32mdef\u001b[0m \u001b[0mlist_data\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mself\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m:\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n",
   "\u001b[0;32m/Users/thomas_young/miniconda3/envs/gluon/lib/python3.6/site-packages/mxnet/gluon/parameter.py\u001b[0m in \u001b[0;36m_check_and_get\u001b[0;34m(self, arr_dict, ctx)\u001b[0m\n\u001b[1;32m    147\u001b[0m                     self.name, str(ctx), str(self._ctx_list)))\n\u001b[1;32m    148\u001b[0m         \u001b[0;32mif\u001b[0m \u001b[0mself\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0m_deferred_init\u001b[0m\u001b[0;34m:\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0;32m--> 149\u001b[0;31m             \u001b[0;32mraise\u001b[0m \u001b[0mDeferredInitializationError\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m\u001b[1;32m    150\u001b[0m         raise RuntimeError(\n\u001b[1;32m    151\u001b[0m             \u001b[0;34m\"Parameter %s has not been initialized. Note that \"\u001b[0m\u001b[0;31m \u001b[0m\u001b[0;31m\\\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n",
   "\u001b[0;31mDeferredInitializationError\u001b[0m: "
  ]
 }
]
```

看得更加清楚点：

```{.python .input}
params.initialize(init=init.One(), force_reinit=True)
print(net[0].weight.data(), net[0].bias.data())
```

更多的方法参见[init的API](https://mxnet.incubator.apache.org/api/python/optimization.html#the-mxnet-initializer-package). 

```{.json .output n=86}
[
 {
  "name": "stdout",
  "output_type": "stream",
  "text": "init weight (4, 5)\ninit weight (2, 4)\n\n[[ 5.34583473  9.91287422  8.48714352  6.86645365  7.26771355]\n [ 7.10037708  8.61027813  5.25294065  9.33191109  6.82748032]\n [ 9.87760735  5.0831399   9.27901649  6.1537118   5.05857038]\n [ 8.82455826  6.79989052  9.72061729  8.64995289  8.74999619]]\n<NDArray 4x5 @cpu(0)> \n[ 0.  0.  0.  0.]\n<NDArray 4 @cpu(0)>\n"
 }
]
```

## 延后的初始化

我们之前提到过Gluon的一个便利的地方是模型定义的时候不需要指定输入的大小，在之后做forward的时候会自动推测参数的大小。我们具体来看这是怎么工作的。

新创建一个网络，然后打印参数。你会发现两个全连接层的权重的形状里都有0。 这是因为在不知道输入数据的情况下，我们无法判断它们的形状。

```{.python .input  n=15}
net = get_net()
net.collect_params()
```

```{.json .output n=15}
[
 {
  "name": "stdout",
  "output_type": "stream",
  "text": "sequential1_ (\n  Parameter sequential1_dense0_weight (shape=(4, 0), dtype=<class 'numpy.float32'>)\n  Parameter sequential1_dense0_bias (shape=(4,), dtype=<class 'numpy.float32'>)\n  Parameter sequential1_dense1_weight (shape=(2, 0), dtype=<class 'numpy.float32'>)\n  Parameter sequential1_dense1_bias (shape=(2,), dtype=<class 'numpy.float32'>)\n)\n"
 }
]
```

然后我们初始化

```{.python .input}
net.initialize()
net.collect_params()
```

你会看到我们形状并没有发生变化，这是因为我们仍然不能确定权重形状。真正的初始化发生在我们看到数据时。

```{.python .input  n=17}
net(x)
net.collect_params()
```

```{.json .output n=17}
[
 {
  "name": "stdout",
  "output_type": "stream",
  "text": "init weight (4, 5)\ninit weight (2, 4)\n"
 },
 {
  "data": {
   "text/plain": "\n[[ 822.10296631  871.62133789]\n [ 815.94232178  867.7588501 ]\n [ 667.56713867  707.49414062]]\n<NDArray 3x2 @cpu(0)>"
  },
  "execution_count": 17,
  "metadata": {},
  "output_type": "execute_result"
 }
]
```

这时候我们看到shape里面的0被填上正确的值了。

## 共享模型参数

有时候我们想在层之间共享同一份参数，我们可以通过Block的`params`输出参数来手动指定参数，而不是让系统自动生成。

```{.python .input  n=22}
net = nn.Sequential()
with net.name_scope():
    net.add(nn.Dense(4, activation="relu"))
    net.add(nn.Dense(4, activation="relu"))
    net.add(nn.Dense(4, activation="relu", params=net[-1].params))
    net.add(nn.Dense(2))
```

初始化然后打印

```{.python .input}
net.initialize()
net(x)
print(net[1].weight.data())
print(net[2].weight.data())
```

## 自定义初始化方法

下面我们自定义一个初始化方法。它通过重载`_init_weight`来实现不同的初始化方法。（注意到Gluon里面`bias`都是默认初始化成0）

```{.python .input}
class MyInit(init.Initializer):
    def __init__(self):
        super(MyInit, self).__init__()
        self._verbose = True
    def _init_weight(self, _, arr):
        # 初始化权重，使用out=arr后我们不需指定形状
        print('init weight', arr.shape)
        nd.random.uniform(low=5, high=10, out=arr)

net = get_net()
net.initialize(MyInit())
net(x)
net[0].weight.data()
```

当然我们也可以通过`Parameter.set_data`来直接改写权重。注意到由于有延后初始化，所以我们通常可以通过调用一次`net(x)`来确定权重的形状先。

```{.python .input}
net = get_net()
net.initialize()
net(x)

print('default weight:', net[1].weight.data())

w = net[1].weight
w.set_data(nd.ones(w.shape))

print('init to all 1s:', net[1].weight.data())
```

```{.json .output n=54}
[
 {
  "name": "stdout",
  "output_type": "stream",
  "text": "init weight (4, 4)\ninit weight (2, 4)\n\n[[ 7.37433767  5.09596586  7.35066128  6.50787401]\n [ 8.58037281  8.30086708  6.43995523  6.45038795]\n [ 6.91731119  8.09007645  8.74584961  7.14384365]\n [ 9.39226151  5.67737055  5.51431656  6.49141169]]\n<NDArray 4x4 @cpu(0)>\n\n[[ 7.37433767  5.09596586  7.35066128  6.50787401]\n [ 8.58037281  8.30086708  6.43995523  6.45038795]\n [ 6.91731119  8.09007645  8.74584961  7.14384365]\n [ 9.39226151  5.67737055  5.51431656  6.49141169]]\n<NDArray 4x4 @cpu(0)>\n"
 }
]
```

```{.python .input  n=95}
print(net[0].params['sequential11_dense0_weight'].data())
print(net[1].params)
print(net.collect_params().get('dense0_weight').data())
print(net[0].bias.data())
```

```{.json .output n=95}
[
 {
  "name": "stdout",
  "output_type": "stream",
  "text": "\n[[ 5.34583473  9.91287422  8.48714352  6.86645365  7.26771355]\n [ 7.10037708  8.61027813  5.25294065  9.33191109  6.82748032]\n [ 9.87760735  5.0831399   9.27901649  6.1537118   5.05857038]\n [ 8.82455826  6.79989052  9.72061729  8.64995289  8.74999619]]\n<NDArray 4x5 @cpu(0)>\nsequential11_dense1_ (\n  Parameter sequential11_dense1_weight (shape=(2, 4), dtype=<class 'numpy.float32'>)\n  Parameter sequential11_dense1_bias (shape=(2,), dtype=<class 'numpy.float32'>)\n)\n\n[[ 5.34583473  9.91287422  8.48714352  6.86645365  7.26771355]\n [ 7.10037708  8.61027813  5.25294065  9.33191109  6.82748032]\n [ 9.87760735  5.0831399   9.27901649  6.1537118   5.05857038]\n [ 8.82455826  6.79989052  9.72061729  8.64995289  8.74999619]]\n<NDArray 4x5 @cpu(0)>\n\n[ 0.  0.  0.  0.]\n<NDArray 4 @cpu(0)>\n"
 }
]
```

## 总结

我们可以很灵活地访问和修改模型参数。

## 练习

1. 研究下`net.collect_params()`返回的是什么？`net.params`呢？
1. 如何对每个层使用不同的初始化函数
1. 如果两个层共用一个参数，那么求梯度的时候会发生什么？

**吐槽和讨论欢迎点**[这里](https://discuss.gluon.ai/t/topic/987)
