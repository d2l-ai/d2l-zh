# 目标检测数据集
:label:`sec_object-detection-dataset`

目标检测领域没有像MNIST和Fashion-MNIST那样的小数据集。
为了快速测试目标检测模型，[**我们收集并标记了一个小型数据集**]。
首先，我们拍摄了一组香蕉的照片，并生成了1000张不同角度和大小的香蕉图像。
然后，我们在一些背景图片的随机位置上放一张香蕉的图像。
最后，我们在图片上为这些香蕉标记了边界框。

## [**下载数据集**]

包含所有图像和CSV标签文件的香蕉检测数据集可以直接从互联网下载。

```{.python .input}
%matplotlib inline
from d2l import mxnet as d2l
from mxnet import gluon, image, np, npx
import os
import pandas as pd

npx.set_np()
```

```{.python .input}
#@tab pytorch
%matplotlib inline
from d2l import torch as d2l
import torch
import torchvision
import os
import pandas as pd
```

```{.python .input  n=1}
#@tab paddle
%matplotlib inline
from d2l import paddle as d2l
import paddle
import paddle.vision as paddlevision
import os
import pandas as pd
```

```{.json .output n=1}
[
 {
  "name": "stderr",
  "output_type": "stream",
  "text": "grep: warning: GREP_OPTIONS is deprecated; please use an alias or script\n"
 }
]
```

```{.python .input  n=2}
#@tab all
#@save
d2l.DATA_HUB['banana-detection'] = (
    d2l.DATA_URL + 'banana-detection.zip',
    '5de26c8fce5ccdea9f91267273464dc968d20d72')
```

## 读取数据集

通过`read_data_bananas`函数，我们[**读取香蕉检测数据集**]。
该数据集包括一个的CSV文件，内含目标类别标签和位于左上角和右下角的真实边界框坐标。

```{.python .input}
#@save
def read_data_bananas(is_train=True):
    """读取香蕉检测数据集中的图像和标签"""
    data_dir = d2l.download_extract('banana-detection')
    csv_fname = os.path.join(data_dir, 'bananas_train' if is_train
                             else 'bananas_val', 'label.csv')
    csv_data = pd.read_csv(csv_fname)
    csv_data = csv_data.set_index('img_name')
    images, targets = [], []
    for img_name, target in csv_data.iterrows():
        images.append(image.imread(
            os.path.join(data_dir, 'bananas_train' if is_train else
                         'bananas_val', 'images', f'{img_name}')))
        # 这里的target包含（类别，左上角x，左上角y，右下角x，右下角y），
        # 其中所有图像都具有相同的香蕉类（索引为0）
        targets.append(list(target))
    return images, np.expand_dims(np.array(targets), 1) / 256
```

```{.python .input}
#@tab pytorch
#@save
def read_data_bananas(is_train=True):
    """读取香蕉检测数据集中的图像和标签"""
    data_dir = d2l.download_extract('banana-detection')
    csv_fname = os.path.join(data_dir, 'bananas_train' if is_train
                             else 'bananas_val', 'label.csv')
    csv_data = pd.read_csv(csv_fname)
    csv_data = csv_data.set_index('img_name')
    images, targets = [], []
    for img_name, target in csv_data.iterrows():
        images.append(torchvision.io.read_image(
            os.path.join(data_dir, 'bananas_train' if is_train else
                         'bananas_val', 'images', f'{img_name}')))
        # 这里的target包含（类别，左上角x，左上角y，右下角x，右下角y），
        # 其中所有图像都具有相同的香蕉类（索引为0）
        targets.append(list(target))
    return images, torch.tensor(targets).unsqueeze(1) / 256
```

```{.python .input  n=3}
#@tab paddle
#@save
def read_data_bananas(is_train=True):
    """读取香蕉检测数据集中的图像和标签"""
    data_dir = d2l.download_extract('banana-detection')
    csv_fname = os.path.join(data_dir, 'bananas_train' if is_train
                             else 'bananas_val', 'label.csv')
    csv_data = pd.read_csv(csv_fname)
    csv_data = csv_data.set_index('img_name')
    images, targets = [], []
    for img_name, target in csv_data.iterrows():
        paddle.vision.set_image_backend('cv2')
        images.append(paddlevision.image_load(os.path.join(data_dir, 'bananas_train' if is_train else
        'bananas_val', 'images', f'{img_name}'))[..., ::-1])
        # 这里的target包含（类别，左上角x，左上角y，右下角x，右下角y）
        # 其中所有图像都具有相同的香蕉类（索引为0）
        targets.append(list(target))
    return images, paddle.to_tensor(targets).unsqueeze(1) / 256
```

通过使用`read_data_bananas`函数读取图像和标签，以下`BananasDataset`类别将允许我们[**创建一个自定义`Dataset`实例**]来加载香蕉检测数据集。

```{.python .input}
#@save
class BananasDataset(gluon.data.Dataset):
    """一个用于加载香蕉检测数据集的自定义数据集"""
    def __init__(self, is_train):
        self.features, self.labels = read_data_bananas(is_train)
        print('read ' + str(len(self.features)) + (f' training examples' if
              is_train else f' validation examples'))

    def __getitem__(self, idx):
        return (self.features[idx].astype('float32').transpose(2, 0, 1),
                self.labels[idx])

    def __len__(self):
        return len(self.features)
```

```{.python .input}
#@tab pytorch
#@save
class BananasDataset(torch.utils.data.Dataset):
    """一个用于加载香蕉检测数据集的自定义数据集"""
    def __init__(self, is_train):
        self.features, self.labels = read_data_bananas(is_train)
        print('read ' + str(len(self.features)) + (f' training examples' if
              is_train else f' validation examples'))

    def __getitem__(self, idx):
        return (self.features[idx].float(), self.labels[idx])

    def __len__(self):
        return len(self.features)
```

```{.python .input  n=4}
#@tab paddle
#@save
class BananasDataset(paddle.io.Dataset):
    """一个用于加载香蕉检测数据集的自定义数据集"""
    def __init__(self, is_train):
        self.features, self.labels = read_data_bananas(is_train)
        print('read ' + str(len(self.features)) + (f' training examples' if
              is_train else f' validation examples'))

    def __getitem__(self, idx):
        return (paddle.to_tensor(self.features[idx], dtype='float32').transpose([2, 0, 1]), self.labels[idx])

    def __len__(self):
        return len(self.features)
```

最后，我们定义`load_data_bananas`函数，来[**为训练集和测试集返回两个数据加载器实例**]。对于测试集，无须按随机顺序读取它。

```{.python .input}
#@save
def load_data_bananas(batch_size):
    """加载香蕉检测数据集"""
    train_iter = gluon.data.DataLoader(BananasDataset(is_train=True),
                                       batch_size, shuffle=True)
    val_iter = gluon.data.DataLoader(BananasDataset(is_train=False),
                                     batch_size)
    return train_iter, val_iter
```

```{.python .input}
#@tab pytorch
#@save
def load_data_bananas(batch_size):
    """加载香蕉检测数据集"""
    train_iter = torch.utils.data.DataLoader(BananasDataset(is_train=True),
                                             batch_size, shuffle=True)
    val_iter = torch.utils.data.DataLoader(BananasDataset(is_train=False),
                                           batch_size)
    return train_iter, val_iter
```

```{.python .input  n=5}
#@tab paddle
#@save
def load_data_bananas(batch_size):
    """加载香蕉检测数据集"""
    train_iter = paddle.io.DataLoader(BananasDataset(is_train=True),
                                             batch_size=batch_size, shuffle=True)
    val_iter = paddle.io.DataLoader(BananasDataset(is_train=False),
                                           batch_size=batch_size)
    return train_iter, val_iter
```

让我们[**读取一个小批量，并打印其中的图像和标签的形状**]。
图像的小批量的形状为（批量大小、通道数、高度、宽度），看起来很眼熟：它与我们之前图像分类任务中的相同。
标签的小批量的形状为（批量大小，$m$，5），其中$m$是数据集的任何图像中边界框可能出现的最大数量。

小批量计算虽然高效，但它要求每张图像含有相同数量的边界框，以便放在同一个批量中。
通常来说，图像可能拥有不同数量个边界框；因此，在达到$m$之前，边界框少于$m$的图像将被非法边界框填充。
这样，每个边界框的标签将被长度为5的数组表示。
数组中的第一个元素是边界框中对象的类别，其中-1表示用于填充的非法边界框。
数组的其余四个元素是边界框左上角和右下角的（$x$，$y$）坐标值（值域在0到1之间）。
对于香蕉数据集而言，由于每张图像上只有一个边界框，因此$m=1$。

```{.python .input  n=6}
#@tab all
batch_size, edge_size = 32, 256
train_iter, _ = load_data_bananas(batch_size)
batch = next(iter(train_iter))
batch[0].shape, batch[1].shape
```

```{.json .output n=6}
[
 {
  "name": "stderr",
  "output_type": "stream",
  "text": "/root/anaconda3/envs/d2l/lib/python3.8/site-packages/paddle/tensor/creation.py:125: DeprecationWarning: `np.object` is a deprecated alias for the builtin `object`. To silence this warning, use `object` by itself. Doing this will not modify any behavior and is safe. \nDeprecated in NumPy 1.20; for more details and guidance: https://numpy.org/devdocs/release/1.20.0-notes.html#deprecations\n  if data.dtype == np.object:\n"
 },
 {
  "ename": "RuntimeError",
  "evalue": "ResourceExhaustedError: \n\nOut of memory error on GPU 7. Cannot allocate 39.250000kB memory on GPU 7, 85.412131TB memory has been allocated and available memory is only 0.000000B.\n\nPlease check whether there is any other process using GPU 7.\n1. If yes, please stop them, or start PaddlePaddle on another GPU.\n2. If no, please decrease the batch size of your model. \nIf the above ways do not solve the out of memory problem, you can try to use CUDA managed memory. The command is `export FLAGS_use_cuda_managed_memory=false`.\n (at /paddle/paddle/fluid/memory/allocation/cuda_allocator.cc:87)\n",
  "output_type": "error",
  "traceback": [
   "\u001b[0;31m---------------------------------------------------------------------------\u001b[0m",
   "\u001b[0;31mRuntimeError\u001b[0m                              Traceback (most recent call last)",
   "Input \u001b[0;32mIn [6]\u001b[0m, in \u001b[0;36m<cell line: 3>\u001b[0;34m()\u001b[0m\n\u001b[1;32m      1\u001b[0m \u001b[38;5;66;03m#@tab all\u001b[39;00m\n\u001b[1;32m      2\u001b[0m batch_size, edge_size \u001b[38;5;241m=\u001b[39m \u001b[38;5;241m32\u001b[39m, \u001b[38;5;241m256\u001b[39m\n\u001b[0;32m----> 3\u001b[0m train_iter, _ \u001b[38;5;241m=\u001b[39m \u001b[43mload_data_bananas\u001b[49m\u001b[43m(\u001b[49m\u001b[43mbatch_size\u001b[49m\u001b[43m)\u001b[49m\n\u001b[1;32m      4\u001b[0m batch \u001b[38;5;241m=\u001b[39m \u001b[38;5;28mnext\u001b[39m(\u001b[38;5;28miter\u001b[39m(train_iter))\n\u001b[1;32m      5\u001b[0m batch[\u001b[38;5;241m0\u001b[39m]\u001b[38;5;241m.\u001b[39mshape, batch[\u001b[38;5;241m1\u001b[39m]\u001b[38;5;241m.\u001b[39mshape\n",
   "Input \u001b[0;32mIn [5]\u001b[0m, in \u001b[0;36mload_data_bananas\u001b[0;34m(batch_size)\u001b[0m\n\u001b[1;32m      3\u001b[0m \u001b[38;5;28;01mdef\u001b[39;00m \u001b[38;5;21mload_data_bananas\u001b[39m(batch_size):\n\u001b[1;32m      4\u001b[0m     \u001b[38;5;124;03m\"\"\"\u52a0\u8f7d\u9999\u8549\u68c0\u6d4b\u6570\u636e\u96c6\"\"\"\u001b[39;00m\n\u001b[0;32m----> 5\u001b[0m     train_iter \u001b[38;5;241m=\u001b[39m paddle\u001b[38;5;241m.\u001b[39mio\u001b[38;5;241m.\u001b[39mDataLoader(\u001b[43mBananasDataset\u001b[49m\u001b[43m(\u001b[49m\u001b[43mis_train\u001b[49m\u001b[38;5;241;43m=\u001b[39;49m\u001b[38;5;28;43;01mTrue\u001b[39;49;00m\u001b[43m)\u001b[49m,\n\u001b[1;32m      6\u001b[0m                                              batch_size\u001b[38;5;241m=\u001b[39mbatch_size, shuffle\u001b[38;5;241m=\u001b[39m\u001b[38;5;28;01mTrue\u001b[39;00m)\n\u001b[1;32m      7\u001b[0m     val_iter \u001b[38;5;241m=\u001b[39m paddle\u001b[38;5;241m.\u001b[39mio\u001b[38;5;241m.\u001b[39mDataLoader(BananasDataset(is_train\u001b[38;5;241m=\u001b[39m\u001b[38;5;28;01mFalse\u001b[39;00m),\n\u001b[1;32m      8\u001b[0m                                            batch_size\u001b[38;5;241m=\u001b[39mbatch_size)\n\u001b[1;32m      9\u001b[0m     \u001b[38;5;28;01mreturn\u001b[39;00m train_iter, val_iter\n",
   "Input \u001b[0;32mIn [4]\u001b[0m, in \u001b[0;36mBananasDataset.__init__\u001b[0;34m(self, is_train)\u001b[0m\n\u001b[1;32m      5\u001b[0m \u001b[38;5;28;01mdef\u001b[39;00m \u001b[38;5;21m__init__\u001b[39m(\u001b[38;5;28mself\u001b[39m, is_train):\n\u001b[0;32m----> 6\u001b[0m     \u001b[38;5;28mself\u001b[39m\u001b[38;5;241m.\u001b[39mfeatures, \u001b[38;5;28mself\u001b[39m\u001b[38;5;241m.\u001b[39mlabels \u001b[38;5;241m=\u001b[39m \u001b[43mread_data_bananas\u001b[49m\u001b[43m(\u001b[49m\u001b[43mis_train\u001b[49m\u001b[43m)\u001b[49m\n\u001b[1;32m      7\u001b[0m     \u001b[38;5;28mprint\u001b[39m(\u001b[38;5;124m'\u001b[39m\u001b[38;5;124mread \u001b[39m\u001b[38;5;124m'\u001b[39m \u001b[38;5;241m+\u001b[39m \u001b[38;5;28mstr\u001b[39m(\u001b[38;5;28mlen\u001b[39m(\u001b[38;5;28mself\u001b[39m\u001b[38;5;241m.\u001b[39mfeatures)) \u001b[38;5;241m+\u001b[39m (\u001b[38;5;124mf\u001b[39m\u001b[38;5;124m'\u001b[39m\u001b[38;5;124m training examples\u001b[39m\u001b[38;5;124m'\u001b[39m \u001b[38;5;28;01mif\u001b[39;00m\n\u001b[1;32m      8\u001b[0m           is_train \u001b[38;5;28;01melse\u001b[39;00m \u001b[38;5;124mf\u001b[39m\u001b[38;5;124m'\u001b[39m\u001b[38;5;124m validation examples\u001b[39m\u001b[38;5;124m'\u001b[39m))\n",
   "Input \u001b[0;32mIn [3]\u001b[0m, in \u001b[0;36mread_data_bananas\u001b[0;34m(is_train)\u001b[0m\n\u001b[1;32m     15\u001b[0m     \u001b[38;5;66;03m# \u8fd9\u91cc\u7684target\u5305\u542b\uff08\u7c7b\u522b\uff0c\u5de6\u4e0a\u89d2x\uff0c\u5de6\u4e0a\u89d2y\uff0c\u53f3\u4e0b\u89d2x\uff0c\u53f3\u4e0b\u89d2y\uff09\u001b[39;00m\n\u001b[1;32m     16\u001b[0m     \u001b[38;5;66;03m# \u5176\u4e2d\u6240\u6709\u56fe\u50cf\u90fd\u5177\u6709\u76f8\u540c\u7684\u9999\u8549\u7c7b\uff08\u7d22\u5f15\u4e3a0\uff09\u001b[39;00m\n\u001b[1;32m     17\u001b[0m     targets\u001b[38;5;241m.\u001b[39mappend(\u001b[38;5;28mlist\u001b[39m(target))\n\u001b[0;32m---> 18\u001b[0m \u001b[38;5;28;01mreturn\u001b[39;00m images, \u001b[43mpaddle\u001b[49m\u001b[38;5;241;43m.\u001b[39;49m\u001b[43mto_tensor\u001b[49m\u001b[43m(\u001b[49m\u001b[43mtargets\u001b[49m\u001b[43m)\u001b[49m\u001b[38;5;241m.\u001b[39munsqueeze(\u001b[38;5;241m1\u001b[39m) \u001b[38;5;241m/\u001b[39m \u001b[38;5;241m256\u001b[39m\n",
   "File \u001b[0;32m~/anaconda3/envs/d2l/lib/python3.8/site-packages/decorator.py:232\u001b[0m, in \u001b[0;36mdecorate.<locals>.fun\u001b[0;34m(*args, **kw)\u001b[0m\n\u001b[1;32m    230\u001b[0m \u001b[38;5;28;01mif\u001b[39;00m \u001b[38;5;129;01mnot\u001b[39;00m kwsyntax:\n\u001b[1;32m    231\u001b[0m     args, kw \u001b[38;5;241m=\u001b[39m fix(args, kw, sig)\n\u001b[0;32m--> 232\u001b[0m \u001b[38;5;28;01mreturn\u001b[39;00m \u001b[43mcaller\u001b[49m\u001b[43m(\u001b[49m\u001b[43mfunc\u001b[49m\u001b[43m,\u001b[49m\u001b[43m \u001b[49m\u001b[38;5;241;43m*\u001b[39;49m\u001b[43m(\u001b[49m\u001b[43mextras\u001b[49m\u001b[43m \u001b[49m\u001b[38;5;241;43m+\u001b[39;49m\u001b[43m \u001b[49m\u001b[43margs\u001b[49m\u001b[43m)\u001b[49m\u001b[43m,\u001b[49m\u001b[43m \u001b[49m\u001b[38;5;241;43m*\u001b[39;49m\u001b[38;5;241;43m*\u001b[39;49m\u001b[43mkw\u001b[49m\u001b[43m)\u001b[49m\n",
   "File \u001b[0;32m~/anaconda3/envs/d2l/lib/python3.8/site-packages/paddle/fluid/wrapped_decorator.py:25\u001b[0m, in \u001b[0;36mwrap_decorator.<locals>.__impl__\u001b[0;34m(func, *args, **kwargs)\u001b[0m\n\u001b[1;32m     22\u001b[0m \u001b[38;5;129m@decorator\u001b[39m\u001b[38;5;241m.\u001b[39mdecorator\n\u001b[1;32m     23\u001b[0m \u001b[38;5;28;01mdef\u001b[39;00m \u001b[38;5;21m__impl__\u001b[39m(func, \u001b[38;5;241m*\u001b[39margs, \u001b[38;5;241m*\u001b[39m\u001b[38;5;241m*\u001b[39mkwargs):\n\u001b[1;32m     24\u001b[0m     wrapped_func \u001b[38;5;241m=\u001b[39m decorator_func(func)\n\u001b[0;32m---> 25\u001b[0m     \u001b[38;5;28;01mreturn\u001b[39;00m \u001b[43mwrapped_func\u001b[49m\u001b[43m(\u001b[49m\u001b[38;5;241;43m*\u001b[39;49m\u001b[43margs\u001b[49m\u001b[43m,\u001b[49m\u001b[43m \u001b[49m\u001b[38;5;241;43m*\u001b[39;49m\u001b[38;5;241;43m*\u001b[39;49m\u001b[43mkwargs\u001b[49m\u001b[43m)\u001b[49m\n",
   "File \u001b[0;32m~/anaconda3/envs/d2l/lib/python3.8/site-packages/paddle/fluid/framework.py:434\u001b[0m, in \u001b[0;36m_dygraph_only_.<locals>.__impl__\u001b[0;34m(*args, **kwargs)\u001b[0m\n\u001b[1;32m    431\u001b[0m \u001b[38;5;28;01mdef\u001b[39;00m \u001b[38;5;21m__impl__\u001b[39m(\u001b[38;5;241m*\u001b[39margs, \u001b[38;5;241m*\u001b[39m\u001b[38;5;241m*\u001b[39mkwargs):\n\u001b[1;32m    432\u001b[0m     \u001b[38;5;28;01massert\u001b[39;00m _non_static_mode(\n\u001b[1;32m    433\u001b[0m     ), \u001b[38;5;124m\"\u001b[39m\u001b[38;5;124mWe only support \u001b[39m\u001b[38;5;124m'\u001b[39m\u001b[38;5;132;01m%s\u001b[39;00m\u001b[38;5;124m()\u001b[39m\u001b[38;5;124m'\u001b[39m\u001b[38;5;124m in dynamic graph mode, please call \u001b[39m\u001b[38;5;124m'\u001b[39m\u001b[38;5;124mpaddle.disable_static()\u001b[39m\u001b[38;5;124m'\u001b[39m\u001b[38;5;124m to enter dynamic graph mode.\u001b[39m\u001b[38;5;124m\"\u001b[39m \u001b[38;5;241m%\u001b[39m func\u001b[38;5;241m.\u001b[39m\u001b[38;5;18m__name__\u001b[39m\n\u001b[0;32m--> 434\u001b[0m     \u001b[38;5;28;01mreturn\u001b[39;00m \u001b[43mfunc\u001b[49m\u001b[43m(\u001b[49m\u001b[38;5;241;43m*\u001b[39;49m\u001b[43margs\u001b[49m\u001b[43m,\u001b[49m\u001b[43m \u001b[49m\u001b[38;5;241;43m*\u001b[39;49m\u001b[38;5;241;43m*\u001b[39;49m\u001b[43mkwargs\u001b[49m\u001b[43m)\u001b[49m\n",
   "File \u001b[0;32m~/anaconda3/envs/d2l/lib/python3.8/site-packages/paddle/tensor/creation.py:184\u001b[0m, in \u001b[0;36mto_tensor\u001b[0;34m(data, dtype, place, stop_gradient)\u001b[0m\n\u001b[1;32m    176\u001b[0m     \u001b[38;5;28;01mreturn\u001b[39;00m core\u001b[38;5;241m.\u001b[39meager\u001b[38;5;241m.\u001b[39mTensor(\n\u001b[1;32m    177\u001b[0m         value\u001b[38;5;241m=\u001b[39mdata,\n\u001b[1;32m    178\u001b[0m         place\u001b[38;5;241m=\u001b[39mplace,\n\u001b[0;32m   (...)\u001b[0m\n\u001b[1;32m    181\u001b[0m         name\u001b[38;5;241m=\u001b[39m\u001b[38;5;28;01mNone\u001b[39;00m,\n\u001b[1;32m    182\u001b[0m         stop_gradient\u001b[38;5;241m=\u001b[39mstop_gradient)\n\u001b[1;32m    183\u001b[0m \u001b[38;5;28;01melse\u001b[39;00m:\n\u001b[0;32m--> 184\u001b[0m     \u001b[38;5;28;01mreturn\u001b[39;00m \u001b[43mpaddle\u001b[49m\u001b[38;5;241;43m.\u001b[39;49m\u001b[43mTensor\u001b[49m\u001b[43m(\u001b[49m\n\u001b[1;32m    185\u001b[0m \u001b[43m        \u001b[49m\u001b[43mvalue\u001b[49m\u001b[38;5;241;43m=\u001b[39;49m\u001b[43mdata\u001b[49m\u001b[43m,\u001b[49m\n\u001b[1;32m    186\u001b[0m \u001b[43m        \u001b[49m\u001b[43mplace\u001b[49m\u001b[38;5;241;43m=\u001b[39;49m\u001b[43mplace\u001b[49m\u001b[43m,\u001b[49m\n\u001b[1;32m    187\u001b[0m \u001b[43m        \u001b[49m\u001b[43mpersistable\u001b[49m\u001b[38;5;241;43m=\u001b[39;49m\u001b[38;5;28;43;01mFalse\u001b[39;49;00m\u001b[43m,\u001b[49m\n\u001b[1;32m    188\u001b[0m \u001b[43m        \u001b[49m\u001b[43mzero_copy\u001b[49m\u001b[38;5;241;43m=\u001b[39;49m\u001b[38;5;28;43;01mFalse\u001b[39;49;00m\u001b[43m,\u001b[49m\n\u001b[1;32m    189\u001b[0m \u001b[43m        \u001b[49m\u001b[43mstop_gradient\u001b[49m\u001b[38;5;241;43m=\u001b[39;49m\u001b[43mstop_gradient\u001b[49m\u001b[43m)\u001b[49m\n",
   "\u001b[0;31mRuntimeError\u001b[0m: ResourceExhaustedError: \n\nOut of memory error on GPU 7. Cannot allocate 39.250000kB memory on GPU 7, 85.412131TB memory has been allocated and available memory is only 0.000000B.\n\nPlease check whether there is any other process using GPU 7.\n1. If yes, please stop them, or start PaddlePaddle on another GPU.\n2. If no, please decrease the batch size of your model. \nIf the above ways do not solve the out of memory problem, you can try to use CUDA managed memory. The command is `export FLAGS_use_cuda_managed_memory=false`.\n (at /paddle/paddle/fluid/memory/allocation/cuda_allocator.cc:87)\n"
  ]
 }
]
```

## [**演示**]

让我们展示10幅带有真实边界框的图像。
我们可以看到在所有这些图像中香蕉的旋转角度、大小和位置都有所不同。
当然，这只是一个简单的人工数据集，实践中真实世界的数据集通常要复杂得多。

```{.python .input}
imgs = (batch[0][0:10].transpose(0, 2, 3, 1)) / 255
axes = d2l.show_images(imgs, 2, 5, scale=2)
for ax, label in zip(axes, batch[1][0:10]):
    d2l.show_bboxes(ax, [label[0][1:5] * edge_size], colors=['w'])
```

```{.python .input}
#@tab pytorch
imgs = (batch[0][0:10].permute(0, 2, 3, 1)) / 255
axes = d2l.show_images(imgs, 2, 5, scale=2)
for ax, label in zip(axes, batch[1][0:10]):
    d2l.show_bboxes(ax, [label[0][1:5] * edge_size], colors=['w'])
```

```{.python .input}
#@tab paddle
imgs = (batch[0][0:10].transpose([0, 2, 3, 1])) / 255
axes = d2l.show_images(imgs, 2, 5, scale=2)
for ax, label in zip(axes, batch[1][0:10]):
    d2l.show_bboxes(ax, [label[0][1:5] * edge_size], colors=['w'])
```

## 小结

* 我们收集的香蕉检测数据集可用于演示目标检测模型。
* 用于目标检测的数据加载与图像分类的数据加载类似。但是，在目标检测中，标签还包含真实边界框的信息，它不出现在图像分类中。

## 练习

1. 在香蕉检测数据集中演示其他带有真实边界框的图像。它们在边界框和目标方面有什么不同？
1. 假设我们想要将数据增强（例如随机裁剪）应用于目标检测。它与图像分类中的有什么不同？提示：如果裁剪的图像只包含物体的一小部分会怎样？

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/3203)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/3202)
:end_tab:
