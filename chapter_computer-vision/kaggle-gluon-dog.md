# 实战Kaggle比赛：狗的品种识别 (ImageNet Dogs)


我们将在本节动手实战Kaggle比赛中的狗的品种识别问题。该比赛的网页地址是

> https://www.kaggle.com/c/dog-breed-identification

在这个比赛中，我们将识别120类不同品种的狗。这个比赛的数据集实际上是著名的ImageNet的子集数据集。和上一节CIFAR-10数据集中的图像不同，ImageNet数据集中的图像的高和宽更大，且大小不一。

图9.17展示了该比赛的网页信息。为了便于提交结果，请先在Kaggle网站上注册账号。

![狗的品种识别比赛的网页信息。比赛数据集可通过点击“Data”标签获取](../img/kaggle-dog.png)


首先，导入实验所需的包或模块。

```{.python .input}
import sys
sys.path.insert(0, '..')

import collections
import datetime
import gluonbook as gb
import math
from mxnet import autograd, gluon, init, nd
from mxnet.gluon import data as gdata, loss as gloss, model_zoo, nn
import os
import shutil
import zipfile
```

## 获取数据集

比赛数据分为训练数据集和测试数据集。训练集包含10,222张图片。测试集包含10,357张图片。两个数据集中的图片格式都是jpg。这些图片都含有RGB三个通道（彩色），高和宽的大小不一。训练集中狗的类别共有120种，例如拉布拉多、贵宾、腊肠、萨摩耶、哈士奇、吉娃娃和约克夏。

### 下载数据集

登录Kaggle后，我们可以点击图9.17所示的狗的品种识别比赛网页上的“Data”标签，并分别下载训练数据集“train.zip”、测试数据集“test.zip”和训练数据集标签“label.csv.zip”。下载完成后，将它们分别存放在以下路径：

* ../data/kaggle_dog/train.zip
* ../data/kaggle_dog/test.zip
* ../data/kaggle_dog/labels.csv.zip


为方便快速上手，我们提供了上述数据集的小规模采样“train_valid_test_tiny.zip”。如果你将使用上述Kaggle比赛的完整数据集，还需要把下面`demo`变量改为`False`。

```{.python .input  n=1}
# 如果使用下载的 Kaggle 比赛的完整数据集，把下面改为 False。
demo = True
data_dir = '../data/kaggle_dog'

if demo:
    zipfiles= ['train_valid_test_tiny.zip']
else:
    zipfiles= ['train.zip', 'test.zip', 'labels.csv.zip']

for f in zipfiles:
    with zipfile.ZipFile(data_dir + '/' + f, 'r') as z:
        z.extractall(data_dir)
```

### 整理数据集

我们定义下面的`reorg_dog_data`函数来整理Kaggle比赛的完整数据集。整理后，同一类狗的图片将被放在同一个文件夹下，便于我们稍后读取。
该函数中的参数`valid_ratio`是验证集中每类狗的样本数与原始训练集中数量最少一类的狗的样本数（66）之比。

```{.python .input  n=2}
def reorg_dog_data(data_dir, label_file, train_dir, test_dir, input_dir, 
                   valid_ratio):
    # 读取训练数据标签。
    with open(os.path.join(data_dir, label_file), 'r') as f:
        # 跳过文件头行（栏名称）。
        lines = f.readlines()[1:]
        tokens = [l.rstrip().split(',') for l in lines]
        idx_label = dict(((idx, label) for idx, label in tokens))

    # 训练集中数量最少一类的狗的样本数。
    min_n_train_per_label = (
        collections.Counter(idx_label.values()).most_common()[:-2:-1][0][1])
    # 验证集中每类狗的样本数。
    n_valid_per_label = math.floor(min_n_train_per_label * valid_ratio)
    label_count = {}

    def mkdir_if_not_exist(path):
        if not os.path.exists(os.path.join(*path)):
            os.makedirs(os.path.join(*path))

    # 整理训练和验证集。
    for train_file in os.listdir(os.path.join(data_dir, train_dir)):
        idx = train_file.split('.')[0]
        label = idx_label[idx]
        mkdir_if_not_exist([data_dir, input_dir, 'train_valid', label])
        shutil.copy(os.path.join(data_dir, train_dir, train_file),
                    os.path.join(data_dir, input_dir, 'train_valid', label))
        if label not in label_count or label_count[label] < n_valid_per_label:
            mkdir_if_not_exist([data_dir, input_dir, 'valid', label])
            shutil.copy(os.path.join(data_dir, train_dir, train_file),
                        os.path.join(data_dir, input_dir, 'valid', label))
            label_count[label] = label_count.get(label, 0) + 1
        else:
            mkdir_if_not_exist([data_dir, input_dir, 'train', label])
            shutil.copy(os.path.join(data_dir, train_dir, train_file),
                        os.path.join(data_dir, input_dir, 'train', label))

    # 整理测试集。
    mkdir_if_not_exist([data_dir, input_dir, 'test', 'unknown'])
    for test_file in os.listdir(os.path.join(data_dir, test_dir)):
        shutil.copy(os.path.join(data_dir, test_dir, test_file),
                    os.path.join(data_dir, input_dir, 'test', 'unknown'))
```

由于我们在这里仅仅使用小数据集，于是将批量大小设为1。在实际训练和测试时，我们应使用Kaggle比赛的完整数据集并调用`reorg_dog_data`函数整理数据集。相应地，我们也需要将批量大小`batch_size`设为一个较大的整数，例如128。

```{.python .input  n=3}
if demo:
    # 注意：此处使用小数据集。
    input_dir = 'train_valid_test_tiny'
    # 注意：此处将批量大小相应设小。使用 Kaggle 比赛的完整数据集时可设较大整数。
    batch_size = 1
else:
    label_file = 'labels.csv'
    train_dir = 'train'
    test_dir = 'test'
    input_dir = 'train_valid_test'
    batch_size = 128
    valid_ratio = 0.1
    reorg_dog_data(data_dir, label_file, train_dir, test_dir, input_dir, 
                   valid_ratio)
```

## 图片增广

为应对过拟合，我们在这里使用`transforms`来增广数据集。例如，加入`transforms.RandomFlipLeftRight()`即可随机对图片做镜面反转。我们也通过`transforms.Normalize()`对彩色图像RGB三个通道分别做标准化。以下列举了部分操作。这些操作可以根据需求来决定是否使用或修改。

```{.python .input  n=4}
transform_train = gdata.vision.transforms.Compose([
    # 随机对图片裁剪出面积为原图片面积 0.08 到 1 倍之间、且高和宽之比在 3/4 和 4/3 之间
    # 的图片，再放缩为高和宽均为 224 像素的新图片。
    gdata.vision.transforms.RandomResizedCrop(224, scale=(0.08, 1.0),
                                              ratio=(3.0/4.0, 4.0/3.0)),
    # 随机左右翻转图片。
    gdata.vision.transforms.RandomFlipLeftRight(),
    # 随机抖动亮度、对比度和饱和度。
    gdata.vision.transforms.RandomColorJitter(brightness=0.4, contrast=0.4,
                                              saturation=0.4),
    # 随机加噪音。
    gdata.vision.transforms.RandomLighting(0.1),
    
    # 将图片像素值按比例缩小到 0 和 1 之间，并将数据格式从“高 * 宽 * 通道”改为
    # “通道 * 高 * 宽”。
    gdata.vision.transforms.ToTensor(),
    # 对图片的每个通道做标准化。
    gdata.vision.transforms.Normalize([0.485, 0.456, 0.406],
                                      [0.229, 0.224, 0.225])
])

# 测试时，只使用确定性的图像预处理操作。
transform_test = gdata.vision.transforms.Compose([
    gdata.vision.transforms.Resize(256),
    # 将图片中央的高和宽均为 224 的正方形区域裁剪出来。
    gdata.vision.transforms.CenterCrop(224),
    gdata.vision.transforms.ToTensor(),
    gdata.vision.transforms.Normalize([0.485, 0.456, 0.406],
                                      [0.229, 0.224, 0.225])
])
```

接下来，我们可以使用`ImageFolderDataset`类来读取整理后的数据集，其中每个数据样本包括图像和标签。需要注意的是，我们要在`DataLoader`中调用刚刚定义好的图片增广函数。其中`transform_first`函数指明对每个数据样本中的图像做数据增广。

```{.python .input  n=5}
# 读取原始图像文件。flag=1 说明输入图像有三个通道（彩色）。
train_ds = gdata.vision.ImageFolderDataset(
    os.path.join(data_dir, input_dir, 'train'), flag=1)
valid_ds = gdata.vision.ImageFolderDataset(
    os.path.join(data_dir, input_dir, 'valid'), flag=1)
train_valid_ds = gdata.vision.ImageFolderDataset(
    os.path.join(data_dir, input_dir, 'train_valid'), flag=1)
test_ds = gdata.vision.ImageFolderDataset(
    os.path.join(data_dir, input_dir, 'test'), flag=1)

train_data = gdata.DataLoader(train_ds.transform_first(transform_train),
                              batch_size, shuffle=True, last_batch='keep')
valid_data = gdata.DataLoader(valid_ds.transform_first(transform_test),
                              batch_size, shuffle=True, last_batch='keep')
train_valid_data = gdata.DataLoader(train_valid_ds.transform_first(
    transform_train), batch_size, shuffle=True, last_batch='keep')
test_data = gdata.DataLoader(test_ds.transform_first(transform_test),
                             batch_size, shuffle=False, last_batch='keep')
```

## 定义模型并使用微调

这个比赛的数据属于ImageNet数据集的子集，因此我们可以应用[“微调”](fine-tuning.md)一节中介绍的思路，选用在ImageNet完整数据集上预训练过的模型，并通过微调在比赛数据集上进行训练。Gluon提供了丰富的预训练模型，我们在这里以预训练过的ResNet-34模型为例。由于比赛数据集属于预训练数据集的子集，我们可以重用预训练模型在输出层的输入（即特征），并将原输出层替换成新的可以训练的小规模输出网络，例如两个串联的全连接层。由于预训练模型的参数在训练中是固定的，我们既节约了它们的训练时间，又节省了存储它们梯度所需的空间。

需要注意的是，我们在图片增广中使用了ImageNet数据集上RGB三个通道的均值和标准差做标准化，这和预训练模型所做的标准化是一致的。

```{.python .input  n=6}
def get_net(ctx):
    # 设 pretrained=True 就能获取预训练模型的参数。第一次使用时需要联网下载。
    finetune_net = model_zoo.vision.resnet34_v2(pretrained=True)
    # 定义新的输出网络。
    finetune_net.output_new = nn.HybridSequential(prefix='')
    finetune_net.output_new.add(nn.Dense(256, activation='relu'))
    # 120是输出的类别数。
    finetune_net.output_new.add(nn.Dense(120))
    # 初始化输出网络。
    finetune_net.output_new.initialize(init.Xavier(), ctx=ctx)
    # 把模型参数分配到即将用于计算的 CPU 或 GPU 上。
    finetune_net.collect_params().reset_ctx(ctx)
    return finetune_net
```

## 定义训练函数

我们将依赖模型在验证集上的表现来选择模型并调节超参数。模型的训练函数`train`只会训练我们定义的输出网络。我们记录了每个迭代周期的训练时间。这有助于比较不同模型的时间开销。

```{.python .input  n=7}
loss = gloss.SoftmaxCrossEntropyLoss()

def get_loss(data, net, ctx):
    l = 0.0
    for X, y in data:
        y = y.as_in_context(ctx)
        # 计算预训练模型输出层的输入，即特征。
        output_features = net.features(X.as_in_context(ctx))
        # 将特征作为我们定义的输出网络的输入，计算输出。
        outputs = net.output_new(output_features)
        l += loss(outputs, y).mean().asscalar()
    return l / len(data)

def train(net, train_data, valid_data, num_epochs, lr, wd, ctx, lr_period,
          lr_decay):
    # 只训练我们定义的输出网络。
    trainer = gluon.Trainer(net.output_new.collect_params(), 'sgd',
                            {'learning_rate': lr, 'momentum': 0.9, 'wd': wd})
    prev_time = datetime.datetime.now()
    for epoch in range(num_epochs):
        train_l = 0.0
        if epoch > 0 and epoch % lr_period == 0:
            trainer.set_learning_rate(trainer.learning_rate * lr_decay)
        for X, y in train_data:
            y = y.astype('float32').as_in_context(ctx)
            # 计算预训练模型输出层的输入，即特征。
            output_features = net.features(X.as_in_context(ctx))
            with autograd.record():
                # 将特征作为我们定义的输出网络的输入，计算输出。
                outputs = net.output_new(output_features)
                l = loss(outputs, y)
            # 反向传播只发生在我们定义的输出网络上。
            l.backward()
            trainer.step(batch_size)
            train_l += l.mean().asscalar()
        cur_time = datetime.datetime.now()
        h, remainder = divmod((cur_time - prev_time).seconds, 3600)
        m, s = divmod(remainder, 60)
        time_s = "time %02d:%02d:%02d" % (h, m, s)
        if valid_data is not None:
            valid_loss = get_loss(valid_data, net, ctx)
            epoch_s = ("epoch %d, train loss %f, valid loss %f, "
                       % (epoch, train_l / len(train_data), valid_loss))
        else:
            epoch_s = ("epoch %d, train loss %f, "
                       % (epoch, train_l / len(train_data)))
        prev_time = cur_time
        print(epoch_s + time_s + ', lr ' + str(trainer.learning_rate))
```

## 训练并验证模型

现在，我们可以训练并验证模型了。以下的超参数都是可以调节的，例如增加迭代周期。

```{.python .input  n=9}
ctx = gb.try_gpu()
num_epochs = 1
# 学习率。
lr = 0.01
# 权重衰减参数。
wd = 1e-4
# 优化算法的学习率将在每 10 个迭代周期时自乘 0.1。
lr_period = 10
lr_decay = 0.1

net = get_net(ctx)
net.hybridize()
train(net, train_data, valid_data, num_epochs, lr, wd, ctx, lr_period,
      lr_decay)
```

## 对测试集分类并在Kaggle提交结果

当得到一组满意的模型设计和超参数后，我们使用全部训练数据集（含验证集）重新训练模型，并对测试集分类。注意，我们要用刚训练好的输出网络做预测。

```{.python .input  n=8}
net = get_net(ctx)
net.hybridize()
train(net, train_valid_data, None, num_epochs, lr, wd, ctx, lr_period,
      lr_decay)

preds = []
for data, label in test_data:
    # 计算预训练模型输出层的输入，即特征。
    output_features = net.features(data.as_in_context(ctx))
    # 将特征作为我们定义的输出网络的输入，计算输出。
    output = nd.softmax(net.output_new(output_features))
    preds.extend(output.asnumpy())
ids = sorted(os.listdir(os.path.join(data_dir, input_dir, 'test/unknown')))
with open('submission.csv', 'w') as f:
    f.write('id,' + ','.join(train_valid_ds.synsets) + '\n')
    for i, output in zip(ids, preds):
        f.write(i.split('.')[0] + ',' + ','.join(
            [str(num) for num in output]) + '\n')
```

执行完上述代码后，会生成一个“submission.csv”文件。这个文件符合Kaggle比赛要求的提交格式。这时我们可以在Kaggle上把对测试集分类的结果提交并查看分类准确率。你需要登录Kaggle网站，访问ImageNet Dogs比赛网页，并点击右侧“Submit Predictions”或“Late Submission”按钮。然后，点击页面下方“Upload Submission File”选择需要提交的分类结果文件。最后，点击页面最下方的“Make Submission”按钮就可以查看结果了。


## 小结

* 我们可以使用在ImageNet数据集上预训练的模型并微调，从而以较小的计算开销对ImageNet的子集数据集做分类。


## 练习

* 使用Kaggle完整数据集，把批量大小`batch_size`和迭代周期数`num_epochs`分别调大些，可以在Kaggle上拿到什么样的结果？
* 使用更深的预训练模型并微调，你能获得更好的结果吗？
* 扫码直达讨论区，在社区交流方法和结果。你能发掘出其他更好的技巧吗？

## 扫码直达[讨论区](https://discuss.gluon.ai/t/topic/2399)

![](../img/qr_kaggle-gluon-dog.svg)

## 参考文献

[1] Kaggle ImageNet Dogs比赛网址。https://www.kaggle.com/c/dog-breed-identification
