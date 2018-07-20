# 通过MXNet/Gluon来动手学习深度学习

[![Build Status](http://ci.mxnet.io/job/gluon-tutorials-zh/badge/icon)](http://ci.mxnet.io/job/gluon-tutorials-zh/)

主页在 [https://zh.gluon.ai/](https://zh.gluon.ai/)。

请使用 [https://discuss.gluon.ai](https://discuss.gluon.ai) 讨论或报告问题。

## 如何贡献

所有notebook是用markdown格式存储，这样方便merge改动。jupyter可以通过notedown来直接使用markdown，[参考这里安装](./chapter_preface/install.md#使用notedown插件来读写github源文件)

build服务器在 http://ci.mxnet.io 。这台服务器有两块Nvidia M60。

可以使用 http://zh.gluon.ai.s3-website-us-west-2.amazonaws.com/ 来访问没有加载过 CDN 的版本，对代码的改动刷新更快。

## 编译HTML版本

所有markdown文件需要在提交前清除output，它们会在服务器上重新执行生成结果。所以需要保证每个notebook执行不要太久，目前限制是20min。

在本地可以如下build html（需要GPU支持）

```{.python .input}
conda env update -f build/build.yml
source activate gluon_zh_docs
make html
```

生成的html会在`_build/html`。

如果没有改动notebook里面源代码，所以不想执行notebook，可以使用

```{.python .input}
make html EVAL=0
```

但这样生成的html将不含有输出结果。

## 编译PDF版本

编译pdf版本需要xelatex、librsvg2-bin（svg图片转pdf）和思源字体。在Ubuntu可以这样安装。

```{.python .input}
sudo apt-get install texlive-full
sudo apt-get install librsvg2-bin
```

```{.python .input}
wget https://github.com/adobe-fonts/source-han-sans/raw/release/OTF/SourceHanSansSC.zip
wget https://github.com/adobe-fonts/source-han-serif/raw/release/OTF/SourceHanSerifSC_SB-H.zip
wget https://github.com/adobe-fonts/source-han-serif/raw/release/OTF/SourceHanSerifSC_EL-M.zip

unzip SourceHanSansSC.zip
unzip SourceHanSerifSC_EL-M.zip
unzip SourceHanSerifSC_SB-H.zip

sudo mv SourceHanSansSC SourceHanSerifSC_EL-M SourceHanSerifSC_SB-H /usr/share/fonts/opentype/
sudo fc-cache -f -v
```

这时候可以通过 `fc-list :lang=zh` 来查看安装的中文字体。

同样的去下载和安装英文字体

```
wget -O source-serif-pro.zip https://www.fontsquirrel.com/fonts/download/source-serif-pro
unzip source-serif-pro -d source-serif-pro
sudo mv source-serif-pro /usr/share/fonts/opentype/

wget -O source-sans-pro.zip https://www.fontsquirrel.com/fonts/download/source-sans-pro
unzip source-sans-pro -d source-sans-pro
sudo mv source-sans-pro /usr/share/fonts/opentype/

wget -O source-code-pro.zip https://www.fontsquirrel.com/fonts/download/source-code-pro
unzip source-code-pro -d source-code-pro
sudo mv source-code-pro /usr/share/fonts/opentype/

sudo fc-cache -f -v
```

然后可以编译了。

```{.python .input}
make pdf
```

## 其他安装


```{.python .input}
python -m spacy download en # 需已 pip install spacy
```


## 英汉术语对照


accuracy，准确率

baseline，基准

batch，批量

bias，偏差

binary classification，二元分类

bucketing，分桶

class，类

classification，分类

collaborative filtering，协同过滤

cost，成本

cross-entropy，交叉熵

data set，数据集

decision boundary，决策边界

dense，稠密

dense layer，全连接层

dropout，丢弃法

empirical risk minimization，经验风险最小化

epoch，迭代周期（周期）

example，样本

feature, 特征

fully connected layer，全连接层

hidden layer，隐藏层

hidden variable，隐藏变量

generalization，泛化

hyperparameter，超参数

hypothesis，假设

import，导入

independent and identically distributed(i.i.d)，独立同分布

instance，实例

label，标签

logistic regression，逻辑回归

mean squared error，均方误差

metric，指标

mini-batch，小批量

normalization，归一化

operator，运算符

optimizer，优化器

parameter，参数

perplexity，困惑度

pipeline，流水线

size，大小

transformation，变换


## 样式规范

贡献请遵照本教程的[样式规范](FORMAT.md)。

