# 数据预处理
:label:`sec_pandas`

到目前为止，我们已经引入了各种技术来处理已经存储在张量中的数据。为了应用深度学习来解决现实世界的问题，我们经常从预处理原始数据开始，而不是那些以张量格式准备好的数据开始。在 Python 中常用的数据分析工具中，通常使用 `pandas` 软件包。像庞大的 Python 生态系统中的许多其他扩展包一样，`pandas` 可以与张量一起工作。因此，我们将简要介绍使用 `pandas` 预处理原始数据并将其转换为张量格式的步骤。我们将在后面的章节中介绍更多的数据预处理技术。

## 读取数据集

样本，我们首先创建存储在 csv（逗号分隔值）文件 `../data/house_tiny.csv` 中的人工数据集。以其他格式存储的数据可以通过类似的方式进行处理。以下 `mkdir_if_not_exist` 函数可确保目录 `../data` 存在。请注意，注释 `# @save `是一个特殊的标记，其中以下函数、类或语句保存在 `d2l` 软件包中，以便稍后可以直接调用它们（例如 `d2l.mkdir_if_not_exist(path)`）而无需重新定义。

```{.python .input}
#@tab all
import os

def mkdir_if_not_exist(path):  #@save
    """Make a directory if it does not exist."""
    if not isinstance(path, str):
        path = os.path.join(*path)
    if not os.path.exists(path):
        os.makedirs(path)
```

下面我们将数据集一行写入 csv 文件中。

```{.python .input}
#@tab all
data_file = '../data/house_tiny.csv'
mkdir_if_not_exist('../data')
with open(data_file, 'w') as f:
    f.write('NumRooms,Alley,Price\n')  # Column names
    f.write('NA,Pave,127500\n')  # Each row represents a data point
    f.write('2,NA,106000\n')
    f.write('4,NA,178100\n')
    f.write('NA,NA,140000\n')
```

要从创建的 csv 文件加载原始数据集，我们导入 `pandas` 包并调用 `read_csv` 函数。此数据集有四行和三列，其中每行描述房屋的房间数量（“NumRooms”）、胡同类型（“胡同”）和价格（“价格”）。

```{.python .input}
#@tab all
# If pandas is not installed, just uncomment the following line:
# !pip install pandas
import pandas as pd

data = pd.read_csv(data_file)
print(data)
```

## 处理丢失的数据

请注意，“NaN” 条目缺少值。为了处理缺失的数据，典型的方法包括 * 插入 * 和 * 删除 *，其中插入用替换值替换值替换，而删除则忽略缺失值。在这里，我们将考虑归咎。

通过基于整数位置的索引 (`iloc`)，我们将 `data` 分成 `inputs` 和 `outputs`，其中前者接受前两列，而后者只保留最后一列。对于缺少的 `inputs` 中的数值，我们用同一列的平均值替换 “NaN” 条目。

```{.python .input}
#@tab all
inputs, outputs = data.iloc[:, 0:2], data.iloc[:, 2]
inputs = inputs.fillna(inputs.mean())
print(inputs)
```

对于 `inputs` 中的类别值或离散值，我们将 “NaN” 视为一个类别。由于 “巷子” 列只接受两种类型的类别值 “铺路” 和 “NaN”，`pandas` 可以自动将此列转换为两列 “巷子铺路” 和 “巷子南”。巷子类型为 “铺路” 的行会将 “胡同路面” 和 “胡同南” 的值设置为 1 和 0。缺少胡同类型的行会将其值设置为 0 和 1。

```{.python .input}
#@tab all
inputs = pd.get_dummies(inputs, dummy_na=True)
print(inputs)
```

## 转换为张量格式

现在 `inputs` 和 `outputs` 中的所有条目都是数值的，它们可以转换为张量格式。一旦数据采用这种格式，它们可以用我们在:numref:`sec_ndarray` 中引入的那些张量函数进一步操作。

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

## 摘要

* 像庞大的 Python 生态系统中的许多其他扩展包一样，`pandas` 可以与张量一起工作。
* 输入和删除可用于处理丢失的数据。

## 练习

创建包含更多行和列的原始数据集。

1. 删除缺失值最多的列。
2. 将预处理的数据集转换为张量格式。

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/28)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/29)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/195)
:end_tab:
