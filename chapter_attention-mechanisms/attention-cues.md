# 注意力提示
:label:`sec_attention-cues`

谢谢你关注这本书。注意力是一种稀缺的资源：目前你正在阅读这本书那么就无法阅读其它的书。因此，与金钱类似，你付出注意力是占用了机会成本的。为了确保您现在投入的注意力是值得的，我们也非常积极地将我们的注意力投入在制作一本好书上。注意力是生命拱门的基石，也是成就非凡作品的关键。

自从经济学开始研究稀缺资源的分配问题，我们就处在注意力经济的时代，人类的注意力已被视为可交易的有限、宝贵和稀缺的商品。为了利用它，已经开发了许多商业模式。在音乐或视频流媒体服务上，我们要么将注意力耗费在他们的广告里，要么付钱来将其隐藏。在线上游戏的世界里，为了角色的成长，我们要么将注意力耗费在可以吸引新玩家的战斗中，要么付钱使角色立即变得强大。没什么是免费的。

总而言之，我们并不缺少环境中的信息，我们缺少的是注意力。在研究视觉场景时，我们的视神经收到的信息大约为每秒 $10^8$ 位，远远超过了我们的大脑能够完全处理的水平。幸运的是，我们的祖先从经验（也称为数据）中学到，*并非所有的感官输入都是一样的*。遍观人类历史，只将注意力引向感兴趣的一小部分信息的能力使我们的大脑能够更明智地分配资源来生存、成长和社交，例如发现捕食者、猎物和配偶。

## 生物学中的注意力提示

为了解释我们的注意力是如何被应用在视觉世界，一个双组（two-component）的框架已经出现并广泛流传。这个想法可以追溯到 19 世纪 90 年代的威廉·詹姆斯，他被认为是 “美国心理学之父” :cite:`James.2007`。在这个框架中，受试者使用 *非自主提示*（nonvolitional cue）和 *自主提示*（volitional cue）有选择地引导注意力的焦点。

非自主性提示是基于环境中物体的显著性和显眼性。想象一下，你面前有五个物品：一份报纸、一篇研究论文、一杯咖啡、一本笔记本和一本书，如 :numref:`fig_eye-coffee` 。所有纸制品都是黑白印刷的，只有咖啡杯是红色的。换句话说，这种咖啡杯在这种视觉环境中本身就是突出和显眼的，会自动且不自主地引起人们的注意。所以你把视网膜的中央凹（fovea，视力最高的黄斑中心）对准咖啡杯，如 :numref:`fig_eye-coffee` 所示。

![Using the nonvolitional cue based on saliency (red cup, non-paper), attention is involuntarily directed to the coffee.](../img/eye-coffee.svg)
:width:`400px`
:label:`fig_eye-coffee`

喝咖啡后，你变得有点兴奋并想读一本书。所以你转过头，重新聚焦你的视线，然后看向 :numref:`fig_eye-book` 中描述的书。与刚才在 :numref:`fig_eye-coffee` 中，根据显著程度而选择了咖啡杯的情况不同，在这种任务驱使的情况下，你看向了那一本书，这是在认知和自主控制下进行的选择。利用了自主提示这种基于变量选择模式的注意力显得更为深思熟虑，而且因为受试者的主观尝试也更加有效。

![Using the volitional cue (want to read a book) that is task-dependent, attention is directed to the book under volitional control.](../img/eye-book.svg)
:width:`400px`
:label:`fig_eye-book`

## 查询、键和值

受到非自主提示和自主提示的启发，我们将在下文中通过纳入这两种注意力提示，来解释注意力机制的设计的框架。

首先，我们考虑只有非自主提示的相对简单的情况。想要对感官输入偏倚地进行选择，我们可以简单地使用参数化的全连接层，甚至是非参数化的最大池化或平均池化。

因此，将注意力机制与那些全连接层或池化层区别开来的，是这种设计是否包含了自主提示。在注意力机制的背景下，我们将自主提示看做是 *查询*（Queries）。给定任何查询，注意力机制通过 *注意力池化*（attention pooling）实现对感官输入（sensory inputs）（例如中间特征表示（intermediate feature representations））的偏倚选择。在注意力机制的背景下，这类输入被称为 *值*（Values）。通常来说，每个 *值* 都与一个 *键*（Keys） 配对，键可以看做是对感官输入的非自主提示。如 :numref:`fig_qkv` 所示，我们可以设计注意力池化，以便对给定的查询（自主提示）可以与键（非自主提示）相互作用，从而完成对于值（感官输入）的偏倚选择。

![Attention mechanisms bias selection over values (sensory inputs) via attention pooling, which incorporates queries (volitional cues) and keys (nonvolitional cues).](../img/qkv.svg)
:label:`fig_qkv`

请注意，注意力机制的设计有许多不同方案。例如，我们还可以设计一中使用强化学习方法 :cite:`Mnih.Heess.Graves.ea.2014` 进行训练的，不可微（non-differentiable）的注意力模型。鉴于 :numref:`fig_qkv` 中描述的框架是最常见的，该框架下的模型将成为本章我们关注的重点。

## 注意力的可视化

平均池化可以被视为输入的加权平均值，其权重是均匀分布的。实际上，注意力池化也使用加权平均来聚合 *值*，其权重是由给定的 *查询* 和不同 *键* 之间计算得出的。

```{.python .input}
from d2l import mxnet as d2l
from mxnet import np, npx
npx.set_np()
```

```{.python .input}
#@tab pytorch
from d2l import torch as d2l
import torch
```

```{.python .input}
#@tab tensorflow
from d2l import tensorflow as d2l
import tensorflow as tf
```

为了对注意力权重进行可视化，我们定义了 `show_heatmaps` 函数。它的输入 `matrices` 具有如下形状 (要显示的行数, 要显示的列数, 查询数, 键数)。

```{.python .input}
#@tab all
#@save
def show_heatmaps(matrices, xlabel, ylabel, titles=None, figsize=(2.5, 2.5),
                  cmap='Reds'):
    d2l.use_svg_display()
    num_rows, num_cols = matrices.shape[0], matrices.shape[1]
    fig, axes = d2l.plt.subplots(num_rows, num_cols, figsize=figsize,
                                 sharex=True, sharey=True, squeeze=False)
    for i, (row_axes, row_matrices) in enumerate(zip(axes, matrices)):
        for j, (ax, matrix) in enumerate(zip(row_axes, row_matrices)):
            pcm = ax.imshow(d2l.numpy(matrix), cmap=cmap)
            if i == num_rows - 1:
                ax.set_xlabel(xlabel)
            if j == 0:
                ax.set_ylabel(ylabel)
            if titles:
                ax.set_title(titles[j])
    fig.colorbar(pcm, ax=axes, shrink=0.6);
```

为了进行演示，我们考虑一个简单的情况，即仅当查询和键相同时，注意力权重为 1；否则为 0。

```{.python .input}
#@tab all
attention_weights = d2l.reshape(d2l.eye(10), (1, 1, 10, 10))
show_heatmaps(attention_weights, xlabel='Keys', ylabel='Queries')
```

在接下来的章节中，我们会经常调用此函数来对注意力权重进行可视化。

## 小结

* 人类的注意力是有限、宝贵和稀缺的资源。
* 受试者通过非自主提示和自主提示有选择地引导自己的注意力。前者基于显著程度，后者取决于任务驱动。
* 由于包含了自主提示，注意力机制有别于全连接层或池化层。
* 注意机制通过注意力池化来对值（感官输入）进行偏倚性的选择，注意力池化中融入了查询（自主提示）和键（非自主提示）。键和值是一一对应的。
* 我们可以直观地对查询和键之间的注意力权重进行可视化。

## 练习

1. 在机器翻译的解码器生成一个一个词组成的序列的过程中，自主提示，非自主提示和感官输入分别可能是什么？
1. 随机生成 $10 \times 10$ 矩阵并使用 `softmax` 运算来确保每行都是有效的概率分布。可视化输出注意力权重。

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/1596)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/1592)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/1710)
:end_tab:
