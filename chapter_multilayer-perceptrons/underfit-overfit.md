# 模型选择、欠拟合和过拟合
:label:`sec_model_selection`

作为机器学习科学家，我们的目标是发现 * 模式 *。但是，我们如何确保我们已经真正发现了一种 * 通用 * 模式，而不是简单地记住我们的数据？样本，想象一下，我们想寻找遗传标记中的模式，将患者与他们的痴呆状态联系起来，其中标签是从 $\{\text{dementia}, \text{mild cognitive impairment}, \text{healthy}\}$ 集中绘制的。由于每个人的基因唯一标识他们（忽略相同的兄弟姐妹），因此可以记住整个数据集。

我们不希望我们的模型说
*“那是鲍勃！我记得他！他患有痴呆症！”*
原因很简单。当我们在未来部署模型时，我们会遇到模型从未见过的患者。我们的预测只有当我们的模型真正发现了一个 * 通用 * 模式时才会有用。

为了更正式地回顾一下，我们的目标是发现能够捕捉我们训练集的基础人群中的规律性的模式。如果我们在这一努力中取得成功, 那么我们甚至可以成功地评估我们以前从未遇到过的个人的风险.这个问题-如何发现 * 泛化 * 的模式-是机器学习的根本问题。

危险在于，当我们训练模型时，我们只访问一小部分数据样本。最大的公共图像数据集包含大约 100 万个影像。更多的时候，我们必须只从成千上万的数据点学习。在一个大型医院系统中，我们可以访问成千上万的医疗记录。在处理有限样本时，我们会冒着风险，我们可能会发现明显的关联，当我们收集更多数据时，这些关联结果不会阻挡。

比我们拟合底层分布更接近训练数据的现象称为 * 覆盖 *，用于对付过拟合的技术称为 * 规则化 *。在前面的部分中，您可能已经在尝试时尚 MNist 数据集时观察到了这种效果。如果在实验过程中改变了模型结构或超参数，您可能会注意到，如果有足够的神经元、层和训练时代，模型最终可以在训练集上达到完美的准确率，即使测试数据的准确度下降。

## 训练错误和泛化错误

为了更正式地讨论这一现象，我们需要区分训练误差和泛化误差。* 训练误差 * 是在训练数据集上计算的模型误差，而 * 泛化误差 * 则是预期模型的误差，如果我们将其应用于从与我们的原始样本相同的底层数据分布中绘制的无限数据点流。

有问题的是，我们永远无法准确计算泛化误差。这是因为无限数据流是一个虚构的对象。在实践中，我们必须 * 估计 * 泛化误差，方法是将我们的模型应用于一个独立的测试集，这些测试集由我们的训练集中预留的数据点随机选择组成。

以下三个思维实验将有助于更好地说明这种情况。考虑一个大学生试图为她的期末考试做准备。勤奋的学生将努力使用前几年的考试来测试自己的能力。尽管如此，在过去的考试中表现出色并不能保证她在重要时会出色。实例，学生可能会尝试通过 Rote 学习考试问题的答案来准备。这需要学生记住很多东西。她甚至可能记得过去考试的答案。另一位学生可能会通过试图了解给出某些答案的原因做好准备。在大多数情况下，后者会做得更好。

同样，考虑一个简单地使用查找表来回答问题的模型。如果允许的输入集是离散的，并且相当小，那么可能在查看 * 多 * 训练示例后，这种方法可以很好地执行。然而，这个模型没有能力做比随机猜测更好，当面对它从来没有见过的例子。实际上，输入空间太大，无法记住对应于每个可想象的输入的答案。样本，考虑黑色和白色 $28\times28$ 图像。如果每个像素可以采取 $256$ 灰度值中的一个，则有 $256^{784}$ 可能的图像。这意味着，低分辨率灰度缩略图大小的图像比宇宙中的原子要多得多。即使我们能够遇到这样的数据，我们也无法承受存储查找表的费用。

最后，考虑试图根据一些可能可用的上下文功能对抛硬币的结果进行分类的问题（类 0：头，类 1：尾）。假设硬币是公平的。无论我们提出什么算法，泛化误差总是为 $\frac{1}{2}$。然而，对于大多数算法，即使我们没有任何功能，我们也应该预计我们的训练误差会大大降低，这取决于平局的运气！考虑数据集 {0、1、1、1、1、0、1}。我们的无功能算法必须回退总是预测 * 多数类 *，从我们的有限样本中看起来是 *1*。在这种情况下，始终预测类 1 的模型将产生 $\frac{1}{3}$ 的误差，远远好于我们的泛化误差。随着数据量的增加，头部分数与 $\frac{1}{2}$ 显著偏差的概率降低，我们的训练误差将与泛化误差相匹配。

### 统计学习理论

由于泛化是机器学习的根本问题，你可能不会感到惊讶的是，许多数学家和理论家都致力于发展形式的理论来描述这种现象。Glivenko 和 Cantelli 在其 [同名定理](https://en.wikipedia.org/wiki/Glivenko%E2%80%93Cantelli_theorem) 中得出了训练误差收敛到泛化误差的速率。在一系列开创性论文中，[瓦普尼克和切尔沃嫩基斯](https://en.wikipedia.org/wiki/Vapnik%E2%80%93Chervonenkis_theory) 将这一理论扩展到了更一般的函数类别。这项工作为统计学习理论奠定了基础。

在标准的监督学习设置中，我们到现在为止已经解决并将在本书的大部分内容中坚持使用，我们假设训练数据和测试数据都是从 * 相同 * 发行版中 * 独立 * 绘制的。这通常称为 *i.d. 假设 *，这意味着采样我们数据的过程没有内存。换句话说，绘制的第二个示例和绘制的第三个示例与绘制的第二个样本和第二个样本没有更多的相关性。

作为一名优秀的机器学习科学家需要批判性思考，而且你已经应该在这个假设中挖空洞，想出假设失败的常见情况。如果我们根据 UCSF 医疗中心的患者收集的数据对死亡风险预测器进行培训，并将其应用于马萨诸塞州总医院的患者，该怎么办？这些分布根本不完全相同。此外，抽奖可能会在时间上相互关联。如果我们对推文的主题进行分类，该怎么办？新闻周期将在所讨论的专题中产生时间依赖性，违反任何独立性假设。

有时候，我们可以摆脱轻微违反身份假设的情况，我们的模型将继续非常好地工作。毕竟，几乎每个现实世界的应用程序都至少涉及一些轻微违反 i.d. 假设，然而，我们有许多有用的工具可用于各种应用程序，如人脸识别，语音识别和语言翻译。

其他违规行为肯定会造成麻烦。想象一下，样本如，如果我们试图通过专门对大学生进行人脸识别系统培训，然后想将其部署为监测养老院人口中老年病学的工具。由于大学生看起来与老年人有很大差异，这种情况不太可能很好。

在随后的章节中，我们将讨论违反 i.d. 假设引起的问题。目前，即使认为 I.d. 假设是理所当然的，理解一泛化也是一个巨大的问题。此外，阐明精确的理论基础，这些基础可能会解释为什么深度神经网络泛化以及它们确实会继续激发学习理论中最伟大的头脑。

当我们训练模型时，我们会尝试搜索一个尽可能适合训练数据的函数。如果函数非常灵活，以至于它可以像真正的关联一样容易捕捉到虚假模式，那么它可能会执行 * 太好 *，而不会生成一个能很好地概括看不到的数据的模型。这正是我们想要避免或至少控制的东西。深度学习中的许多技术都是启发式和技巧，旨在防止过拟合。

### 模型复杂性

当我们有简单的模型和丰富的数据时，我们预计泛化误差与训练误差相似。当我们使用更复杂的模型和更少的示例时，我们预计训练误差会下降，但泛化差距会扩大。究竟构成模型复杂度是一个复杂的问题。许多因素决定一个模型是否能够很好地推广。样本，具有更多参数的模型可能会被认为更复杂。其参数可以采用更广泛的值范围的模型可能会更加复杂。通常使用神经网络，我们认为一个模型需要更多的训练迭代更复杂，一个模型受 * 早期停止 *（较少的训练迭代）的影响不太复杂。

很难比较实质性不同的模型类（例如，决策树与神经网络）的成员之间的复杂性。现在，一个简单的经验法则非常有用：一个能够轻松解释任意事实的模型是统计人员认为复杂的模型，而一个只有有限的表现力，但仍然能够很好地解释数据的模型可能更接近真相。在哲学上，这与 Popper 的科学理论的伪造性标准密切相关：如果理论适合数据，并且有特定的测试可以用来反驳这一理论是好的。这一点很重要，因为所有统计估计都是
*后备 *，
也就是说，我们在观察事实之后估计，因此容易受到相关的谬误之害。现在，我们将把这一理念放在一边，坚持更具体的问题。

在本节中，为了给你一些直觉，我们将重点介绍一些倾向于影响模型类的概括性的因素：

1. 可调谐参数的数量。当可调谐参数的数量（有时称为 * 自由度 *）较大时，模型往往更容易过拟合。
1. 参数获取的值。当权重可以采取更大范围的值时，模型更容易过拟合。
1. 培训示例的数量。即使模型很简单，只包含一个或两个示例的数据集也很容易过度拟合。但是，使用数百万个示例过拟合数据集需要一个非常灵活的模型。

## 型号选择

在机器学习中，我们通常在评估几个候选模型后选择我们的最终模型。此过程称为 * 型号选择 *。有时候，需要比较的模型在本质上有根本不同（例如，决策树与线性模型）。在其他时候，我们正在比较已经使用不同超参数设置训练过的同一类模型的成员。

样本，对于 MLP，我们可能希望将模型与不同数量的隐藏层、不同数量的隐藏单位以及应用于每个隐藏层的激活函数的各种选择进行比较。为了确定候选模型中最好的，我们通常会使用验证数据集。

### 验证数据集

原则上，在我们选择了所有超参数之后，我们不应该触摸我们的测试集。如果我们在模型选择过程中使用测试数据，则存在测试数据过度拟合的风险。那么我们就会遇到严重的麻烦了如果我们超出了我们的训练数据，总是会对测试数据进行评估，以保证我们的诚实。但是，如果我们超出测试数据，我们怎么会知道呢？

因此，我们绝不应依赖测试数据进行模型选择。然而，我们不能仅仅依赖训练数据进行模型选择，因为我们无法估计用于训练模型的数据的泛化误差。

在实际应用中，图片变得更加泥泞。虽然理想情况下，我们只接触一次测试数据，但是为了评估最佳模型或者相互比较少量模型，但实际测试数据很少在一次使用后被丢弃。我们很少能为每一轮实验提供一套新的测试。

解决这个问题的常见做法是将我们的数据分成三种方式，除了培训和测试数据集外，还包含一个 * 验证数据集 *（或 * 验证集 *）。结果是一种模糊的做法，验证和测试数据之间的界限令人担忧的模糊性。除非另有明确说明，否则在本书中的实验中，我们真的正在使用应该被称为训练数据和验证数据，没有真正的测试集。因此，在本书的每个实验中报告的准确率实际上是验证准确率，而不是真正的测试集准确率。

### 折叠交叉验证

当训练数据稀少时，我们甚至可能无法保存足够的数据来构成正确的验证集。这个问题的一个流行的解决方案是使用 $K$* 倍交叉验证 *。在这里，原始训练数据被拆分为 $K$ 非重叠子集。然后，模型训练和验证执行 $K$ 次，每次在 $K-1$ 子集上进行训练，并在不同的子集上进行验证（该回合中未用于训练的子集）。最后，通过对 $K$ 实验结果的平均值来估计训练和验证误差。

## 不合身还是过度拟合？

当我们比较训练和验证错误时，我们希望注意两种常见情况。首先，我们要注意我们的训练误差和验证误差都很大，但它们之间有一点差距的情况。如果模型无法减少训练误差，这可能意均值我们的模型太简单（即表现力不足），无法捕获我们试图建模的模式。此外，由于我们的训练和验证错误之间的 * 泛化差距 * 很小，我们有理由相信我们可以用更复杂的模型摆脱。这种现象被称为 * 不适应 *。

另一方面，正如我们上面讨论的那样，我们希望注意训练误差明显低于验证误差的情况，这表明严重的 * 覆盖 *。请注意，过拟合并不总是一件坏事。特别是对于深度学习，众所周知，最佳预测模型在训练数据上的表现通常比在保持数据上的表现要好得多。最终，我们通常更关心验证误差，而不是培训和验证错误之间的差距。

我们是否超适或不适合可能取决于模型的复杂性和可用训练数据集的大小，我们将在下面讨论两个主题。

### 模型复杂性

为了说明一些关于过拟合和模型复杂度的经典直觉，我们给出了一个使用多项式的样本。鉴于训练数据由单个特征 $x$ 和相应的实值标签 $y$ 组成, 我们试图找出 $d$ 度的多项式

$$\hat{y}= \sum_{i=0}^d x^i w_i$$

来估计标签上的标签。这只是一个线性回归问题，其中我们的特征由 $x$ 的幂给出，模型的权重由 $w_i$ 给出，并由 $w_0$ 自 $x^0 = 1$ 以来的偏差给出。由于这只是一个线性回归问题，我们可以使用平方误差作为我们的损失函数。

高阶多项式函数比低阶多项式函数复杂得多，因为高阶多项式具有更多的参数，而且模型函数的选择范围也更宽。修复训练数据集时，较高阶多项式函数应始终实现相对于较低度多项式的训练误差（在最坏情况下，相等）。事实上，每当每个数据点的不同值为 $x$ 时，度等于数据点数的多项式函数就可以完美地拟合训练集。我们在 :numref:`fig_capacity_vs_error` 中对多项式度与欠拟合与过拟合之间的关系进行了可视化。

![Influence of model complexity on underfitting and overfitting](../img/capacity-vs-error.svg)
:label:`fig_capacity_vs_error`

### 数据集大小

要记住的另一个重要考虑因素是数据集的大小。修复我们的模型，训练数据集中的样本越少，我们遇到过拟合的可能性就越大（而且更严重）。随着训练数据量的增加，泛化误差通常会减少。此外，在一般情况下，更多的数据永远不会受到伤害。对于固定任务和数据分布，模型复杂度与数据集大小之间通常存在关系。考虑到更多的数据，我们可能会尝试拟合一个更复杂的模型。没有足够的数据，更简单的模型可能会更难击败。对于许多任务，当数千个训练示例可用时，深度学习的性能仅优于线性模型。当前深度学习的成功在某种程度上归功于互联网公司、廉价存储、连接设备以及经济的广泛数字化等大量数据集。

## 多项式回归

我们现在可以通过将多项式拟合到数据来交互地探索这些概念。

```{.python .input}
from d2l import mxnet as d2l
from mxnet import gluon, np, npx
from mxnet.gluon import nn
import math
npx.set_np()
```

```{.python .input}
#@tab pytorch
from d2l import torch as d2l
import torch
from torch import nn
import numpy as np
import math
```

```{.python .input}
#@tab tensorflow
from d2l import tensorflow as d2l
import tensorflow as tf
import numpy as np
import math
```

### 生成数据集

首先我们需要数据。给定 $x$，我们将使用以下立方多项式生成训练和测试数据的标签：

$$y = 5 + 1.2x - 3.4\frac{x^2}{2!} + 5.6 \frac{x^3}{3!} + \epsilon \text{ where }
\epsilon \sim \mathcal{N}(0, 0.1^2).$$

噪声项 $\epsilon$ 服从正态分布，均值为 0，标准差为 0.1。为了优化，我们通常希望避免非常大的梯度或损耗值。这就是为什么 * 功能 * 从 $x^i$ 重新调整到 $\ 帧 {x^i} {i!}$。它允许我们避免大指数 $i$ 的非常大的值。我们将为训练集和测试集合合成 100 个样本。

```{.python .input}
#@tab all
max_degree = 20  # Maximum degree of the polynomial
n_train, n_test = 100, 100  # Training and test dataset sizes
true_w = np.zeros(max_degree)  # Allocate lots of empty space
true_w[0:4] = np.array([5, 1.2, -3.4, 5.6])

features = np.random.normal(size=(n_train + n_test, 1))
np.random.shuffle(features)
poly_features = np.power(features, np.arange(max_degree).reshape(1, -1))
for i in range(max_degree):
    poly_features[:, i] /= math.gamma(i + 1)  # `gamma(n)` = (n-1)!
# Shape of `labels`: (`n_train` + `n_test`,)
labels = np.dot(poly_features, true_w)
labels += np.random.normal(scale=0.1, size=labels.shape)
```

同样，存储在 `poly_features` 中的单体按伽玛函数重新缩放，其中 $\ 伽玛 (n) = (n-1)!$。查看生成的数据集中的前两个样本。从技术上讲，值 1 是一个特征，即与偏差对应的常量特征。

```{.python .input}
#@tab pytorch, tensorflow
# Convert from NumPy ndarrays to tensors
true_w, features, poly_features, labels = [d2l.tensor(x, dtype=
    d2l.float32) for x in [true_w, features, poly_features, labels]]
```

```{.python .input}
#@tab all
features[:2], poly_features[:2, :], labels[:2]
```

### 培训和测试模型

让我们首先实现一个函数来评估给定数据集上的损失。

```{.python .input}
#@tab all
def evaluate_loss(net, data_iter, loss):  #@save
    """Evaluate the loss of a model on the given dataset."""
    metric = d2l.Accumulator(2)  # Sum of losses, no. of examples
    for X, y in data_iter:
        l = loss(net(X), y)
        metric.add(d2l.reduce_sum(l), d2l.size(l))
    return metric[0] / metric[1]
```

现在定义训练功能。

```{.python .input}
def train(train_features, test_features, train_labels, test_labels,
          num_epochs=400):
    loss = gluon.loss.L2Loss()
    net = nn.Sequential()
    # Switch off the bias since we already catered for it in the polynomial
    # features
    net.add(nn.Dense(1, use_bias=False))
    net.initialize()
    batch_size = min(10, train_labels.shape[0])
    train_iter = d2l.load_array((train_features, train_labels), batch_size)
    test_iter = d2l.load_array((test_features, test_labels), batch_size,
                               is_train=False)
    trainer = gluon.Trainer(net.collect_params(), 'sgd',
                            {'learning_rate': 0.01})
    animator = d2l.Animator(xlabel='epoch', ylabel='loss', yscale='log',
                            xlim=[1, num_epochs], ylim=[1e-3, 1e2],
                            legend=['train', 'test'])
    for epoch in range(num_epochs):
        d2l.train_epoch_ch3(net, train_iter, loss, trainer)
        if epoch == 0 or (epoch + 1) % 20 == 0:
            animator.add(epoch + 1, (evaluate_loss(net, train_iter, loss),
                                     evaluate_loss(net, test_iter, loss)))
    print('weight:', net[0].weight.data().asnumpy())
```

```{.python .input}
#@tab pytorch
def train(train_features, test_features, train_labels, test_labels,
          num_epochs=400):
    loss = nn.MSELoss()
    input_shape = train_features.shape[-1]
    # Switch off the bias since we already catered for it in the polynomial
    # features
    net = nn.Sequential(nn.Linear(input_shape, 1, bias=False))
    batch_size = min(10, train_labels.shape[0])
    train_iter = d2l.load_array((train_features, train_labels.reshape(-1,1)),
                                batch_size)
    test_iter = d2l.load_array((test_features, test_labels.reshape(-1,1)),
                               batch_size, is_train=False)
    trainer = torch.optim.SGD(net.parameters(), lr=0.01)
    animator = d2l.Animator(xlabel='epoch', ylabel='loss', yscale='log',
                            xlim=[1, num_epochs], ylim=[1e-3, 1e2],
                            legend=['train', 'test'])
    for epoch in range(num_epochs):
        d2l.train_epoch_ch3(net, train_iter, loss, trainer)
        if epoch == 0 or (epoch + 1) % 20 == 0:
            animator.add(epoch + 1, (evaluate_loss(net, train_iter, loss),
                                     evaluate_loss(net, test_iter, loss)))
    print('weight:', net[0].weight.data.numpy())
```

```{.python .input}
#@tab tensorflow
def train(train_features, test_features, train_labels, test_labels,
          num_epochs=400):
    loss = tf.losses.MeanSquaredError()
    input_shape = train_features.shape[-1]
    # Switch off the bias since we already catered for it in the polynomial
    # features
    net = tf.keras.Sequential()
    net.add(tf.keras.layers.Dense(1, use_bias=False))
    batch_size = min(10, train_labels.shape[0])
    train_iter = d2l.load_array((train_features, train_labels), batch_size)
    test_iter = d2l.load_array((test_features, test_labels), batch_size,
                               is_train=False)
    trainer = tf.keras.optimizers.SGD(learning_rate=.01)
    animator = d2l.Animator(xlabel='epoch', ylabel='loss', yscale='log',
                            xlim=[1, num_epochs], ylim=[1e-3, 1e2],
                            legend=['train', 'test'])
    for epoch in range(num_epochs):
        d2l.train_epoch_ch3(net, train_iter, loss, trainer)
        if epoch == 0 or (epoch + 1) % 20 == 0:
            animator.add(epoch + 1, (evaluate_loss(net, train_iter, loss),
                                     evaluate_loss(net, test_iter, loss)))
    print('weight:', net.get_weights()[0].T)
```

### 三阶多项式函数拟合（正态）

我们首先使用三阶多项式函数，它与数据生成函数的顺序相同。结果表明，该模型的训练损耗和测试损耗都能有效降低。学习的模型参数也接近真实值 $w = [5, 1.2, -3.4, 5.6]$。

```{.python .input}
#@tab all
# Pick the first four dimensions, i.e., 1, x, x^2/2!, x^3/3! from the
# polynomial features
train(poly_features[:n_train, :4], poly_features[n_train:, :4],
      labels[:n_train], labels[n_train:])
```

### 线性函数拟合（欠拟合）

让我们再来看一下线性函数拟合。在早期时代下降之后，很难进一步减少该模型的训练损失。最后一个迭代周期（周期) 迭代完成后，训练损失仍然很高。当用于拟合非线性模式（如此处的三阶多项式函数）时，线性模型可能会不适合。

```{.python .input}
#@tab all
# Pick the first two dimensions, i.e., 1, x, from the polynomial features
train(poly_features[:n_train, :2], poly_features[n_train:, :2],
      labels[:n_train], labels[n_train:])
```

### 高阶多项式函数拟合（过拟合）

现在让我们尝试使用程度过高的多项式训练模型。在这里，没有足够的数据来了解较高度系数应该具有接近零的值。因此，我们过于复杂的模型非常容易受到训练数据中噪声的影响。虽然训练损失能够有效降低，但测试损失仍然要高得多。它表明复杂模型会覆盖数据。

```{.python .input}
#@tab all
# Pick all the dimensions from the polynomial features
train(poly_features[:n_train, :], poly_features[n_train:, :],
      labels[:n_train], labels[n_train:], num_epochs=1500)
```

在接下来的章节中，我们将继续讨论过于拟合的问题以及处理这些问题的方法，例如体重衰减和丢弃法。

## 摘要

* 由于无法根据训练误差估计泛化误差，简单地将训练误差降至最低并不一定均值泛化误差的减少。机器学习模型需要小心防止过拟合，以便最大限度地减少泛化误差。
* 验证集可用于模型选择，前提是它的使用不太宽松。
* 不适合意味着模型无法减少训练误差。当训练误差远低于验证误差时，会出现过拟合。
* 我们应该选择适当复杂的模型，避免使用不足的训练样本。

## 练习

1. 你能解决多项式回归问题吗？提示：使用线性代数。
1. 多项式的礼宾模型选择：
    * 绘制训练损失与模型复杂度（多项式的程度）。你观察到什么？您需要多种程度的多项式才能将训练损失降至 0？
    * 在这种情况下绘制测试损失。
    * 生成与数据量函数相同的图。
1. 如果你放弃归一化会发生什么（$1/i！$) of the polynomial features $ x 一个美元？你能以其他方式解决这个问题吗？
1. 你能期望看到零泛化误差吗？

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/96)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/97)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/234)
:end_tab:
