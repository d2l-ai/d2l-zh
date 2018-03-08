# 通过MXNet/Gluon来动手学习深度学习

主页在 [https://zh.gluon.ai/](https://zh.gluon.ai/)。

请使用 [https://discuss.gluon.ai](https://discuss.gluon.ai) 讨论或报告问题。

## 如何贡献

所有notebook是用markdown格式存储，这样方便merge改动。jupyter可以通过notedown来直接使用markdown，[参考这里安装](./chapter_preface/install.md#使用notedown插件来读写github源文件)

build服务器在 http://ci.mxnet.io 。这台服务器有两块Nvidia M60。

所有markdown文件需要在提交前清除output，它们会在服务器上重新执行生成结果。所以需要保证每个notebook执行不要太久，目前限制是20min。

在本地可以如下build html（需要GPU支持）

```bash
conda env update -f build/build.yml
source activate gluon_zh_docs
make html
```

生成的html会在`_build/html`。

如果没有改动notebook里面源代码，所以不想执行notebook，可以使用

```
make html EVAL=0
```

但这样生成的html将不含有输出结果。

## 编译PDF版本

编译pdf版本需要xelatex，和思源字体。在Ubuntu可以这样安装。

```bash
sudo apt-get install texlive-full
```

```bash
wget https://github.com/adobe-fonts/source-han-sans/raw/release/OTF/SourceHanSansHWSC.zip
wget https://github.com/adobe-fonts/source-han-serif/raw/release/OTF/SourceHanSerifSC_EL-M.zip
unzip SourceHanSansHWSC.zip
unzip SourceHanSerifSC_EL-M.zip
sudo mv SourceHanSansHWSC SourceHanSerifSC_EL-M /usr/share/fonts/opentype/
sudo fc-cache -f -v
```

这时候可以通过 `fc-list :lang=zh` 来查看安装的中文字体。

然后可以编译了。

```bash
make pdf
```


## Terminology 中英术语对照表

action, 动作

adversarial learning, 对抗学习

agent, 智能体

attribute space, 属性空间

attribute value, 属性值

attribute, 属性

binary classification, 二分类

classification, 分类

cluster, 簇

clustering, 聚类

confidence, 确信度

contextual bandit problem, 情境式赌博机问题

covariate shift, 协变量转移

credit assignment problem, 信用分配问题

cross-entropy, 交叉熵

data set, 数据集

dimensionality, 维数

distribution, 分布

reinforcement learning, 强化学习

example, 样例

feature vector, 特征向量

feature, 特征

generalization, 泛化

generative adversarial networks, 生成对抗网络

ground-truth, 真相、真实

hypothesis, 假设

independent and identically distributed(i.i.d), 独立同分布

instance, 示例

label space, 标注空间

label, 标注

learing algorithm, 学习算法

learned model, 学得模型

learner, 学习器

learning, 学习

machine translation, 机器翻译

Markov Decision Process, 马尔可夫决策过程

model, 模型

multi-armed bandit problem, 多臂赌博机问题

multi-class classification, 多分类

negative class, 反类

offline learning, 离线学习

positive class, 正类

prediction, 预测

principal component analysis, 主成分分析

regression, 回归

reinforcement learning, 强化学习

representation learning, 表征学习

sample space, 样本空间

sample, 样本

sepecilization, 特化

sequence learning, 序列学习

subspace estimation, 子空间估计

supervised learning, 监督学习

testing sample, 测试样本

testing, 测试

time step, 时间步长

training data, 训练数据

training sample, 训练样本

training set, 训练集

training, 训练

unsupervised learning, 无监督学习
