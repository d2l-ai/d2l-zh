# 数据预处理
:label:`sec_pandas`

为了能用深度学习来解决现实世界的问题，我们经常从预处理原始数据开始，
而不是从那些准备好的张量格式数据开始。
在Python中常用的数据分析工具中，我们通常使用`pandas`软件包。
像庞大的Python生态系统中的许多其他扩展包一样，`pandas`可以与张量兼容。
本节我们将简要介绍使用`pandas`预处理原始数据，并将原始数据转换为张量格式的步骤。
后面的章节将介绍更多的数据预处理技术。

## 读取数据集

举一个例子，我们首先(**创建一个人工数据集，并存储在CSV（逗号分隔值）文件**)
`../data/house_tiny.csv`中。
以其他格式存储的数据也可以通过类似的方式进行处理。
下面我们将数据集按行写入CSV文件中。

```{.python .input}
#@tab all
import os

os.makedirs(os.path.join('..', 'data'), exist_ok=True)
data_file = os.path.join('..', 'data', 'house_tiny.csv')
with open(data_file, 'w') as f:
    f.write('NumRooms,Alley,Price\n')  # 列名
    f.write('NA,Pave,127500\n')  # 每行表示一个数据样本
    f.write('2,NA,106000\n')
    f.write('4,NA,178100\n')
    f.write('NA,NA,140000\n')
```

要[**从创建的CSV文件中加载原始数据集**]，我们导入`pandas`包并调用`read_csv`函数。该数据集有四行三列。其中每行描述了房间数量（“NumRooms”）、巷子类型（“Alley”）和房屋价格（“Price”）。

```{.python .input}
#@tab all
# 如果没有安装pandas，只需取消对以下行的注释来安装pandas
# !pip install pandas
import pandas as pd

data = pd.read_csv(data_file)
print(data)
```

## 处理缺失值

注意，“NaN”项代表缺失值。
[**为了处理缺失的数据，典型的方法包括*插值法*和*删除法*，**]
其中插值法用一个替代值弥补缺失值，而删除法则直接忽略缺失值。
在(**这里，我们将考虑插值法**)。

通过位置索引`iloc`，我们将`data`分成`inputs`和`outputs`，
其中前者为`data`的前两列，而后者为`data`的最后一列。
对于`inputs`中缺少的数值，我们用同一列的均值替换“NaN”项。

```{.python .input}
#@tab all
inputs, outputs = data.iloc[:, 0:2], data.iloc[:, 2]
inputs = inputs.fillna(inputs.mean())
print(inputs)
```

[**对于`inputs`中的类别值或离散值，我们将“NaN”视为一个类别。**]
由于“巷子类型”（“Alley”）列只接受两种类型的类别值“Pave”和“NaN”，
`pandas`可以自动将此列转换为两列“Alley_Pave”和“Alley_nan”。
巷子类型为“Pave”的行会将“Alley_Pave”的值设置为1，“Alley_nan”的值设置为0。
缺少巷子类型的行会将“Alley_Pave”和“Alley_nan”分别设置为0和1。

```{.python .input}
#@tab all
inputs = pd.get_dummies(inputs, dummy_na=True)
print(inputs)
```

## 转换为张量格式

[**现在`inputs`和`outputs`中的所有条目都是数值类型，它们可以转换为张量格式。**]
当数据采用张量格式后，可以通过在 :numref:`sec_ndarray`中引入的那些张量函数来进一步操作。

```{.python .input}
from mxnet import np

X, y = np.array(inputs.values), np.array(outputs.values)
X, y
```

```{.python .input}
#@tab pytorch
import torch

X, y = torch.tensor(inputs.values), torch.tensor(outputs.values)
X, y
```

```{.python .input}
#@tab tensorflow
import tensorflow as tf

X, y = tf.constant(inputs.values), tf.constant(outputs.values)
X, y
```

```{.python .input}
#@tab paddle
import warnings
warnings.filterwarnings(action='ignore')
import paddle

X, y = paddle.to_tensor(inputs.values), paddle.to_tensor(outputs.values)
X, y
```

```{.python .input}
#@tab mindspore
import mindspore

X, y = mindspore.Tensor(inputs.values), mindspore.Tensor(outputs.values)
X, y
```

## 小结

* `pandas`软件包是Python中常用的数据分析工具中，`pandas`可以与张量兼容。
* 用`pandas`处理缺失的数据时，我们可根据情况选择用插值法和删除法。

## 练习

创建包含更多行和列的原始数据集。

1. 删除缺失值最多的列。
2. 将预处理后的数据集转换为张量格式。

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/1749)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/1750)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/1748)
:end_tab:

:begin_tab:`paddle`
[Discussions](https://discuss.d2l.ai/t/11681)
:end_tab:
