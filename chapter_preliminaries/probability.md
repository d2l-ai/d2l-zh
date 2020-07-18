# 概率
:label:`sec_prob`

在某种形式上，机器学习就是做出预测。鉴于患者的临床病史，我们可能希望预测下一年心脏病发作的 * 概率 *。在异常检测中，我们可能需要评估一组来自飞机的喷气发动机的读数如何正常运行。在强化学习中，我们希望代理人能够在环境中智能地行动。这意味着我们需要考虑在每个可用操作下获得高奖励的概率。当我们建立推荐系统时，我们也需要考虑概率。样本，假设 * 我们为一家大型在线书商工作。我们可能需要估计特定用户购买特定图书的概率。为此，我们需要使用概率的语言。整个课程，专业，论文，职业，甚至部门，都致力于概率。所以很自然，我们在这部分的目标不是教授整个科目。相反，我们希望能让你离开地面，教你足够，你可以开始构建你的第一个深度学习模型，并给你足够的风味，你可以开始在你愿意的情况下自己探索它。

我们已经在前面的章节中援引了概率，但没有说明它们确切的是什么，也没有举出具体的样本。让我们现在通过考虑第一种情况变得更加严重：根据照片区分猫和狗。这听起来可能很简单，但它实际上是一个巨大的挑战。首先，问题的难度可能取决于图像的分辨率。

![Images of varying resolutions ($10 \times 10$, $20 \times 20$, $40 \times 40$, $80 \times 80$, and $160 \times 160$ pixels).](../img/cat_dog_pixels.png)
:width:`300px`
:label:`fig_cat_dog`

如 :numref:`fig_cat_dog` 所示，虽然人类很容易以 $160 \times 160$ 像素的分辨率识别猫和狗，但它在 $40 \times 40$ 像素上变得具有挑战性，而且在 $10 \times 10$ 像素下几乎是不可能的。换句话说，我们能够在很远的距离（从而降低分辨率）区分猫和狗的能力可能会接近不知情的猜测。概率为我们提供了一种正式的推理方式，我们的确定性水平。如果我们完全肯定的图像描绘了一只猫，我们说，* 概率 * 相应的标签 $y$ 是 "猫"，表示 $P(y=$ "猫" $)$ 等于 $1$。如果我们没有证据表明 $y =$ “猫” 或 $y =$ “狗”，那么我们可以说这两种可能性是同样的
*有可能表示这一点为 "猫" $) = P(y=$ "狗"。如果我们是合理的
自信，但不知道图像描绘了一只猫，我们可能会分配一个概率 $0.5  < P(y=$ “猫” $) < 1$。

现在考虑第二个案例：给出一些天气监测数据，我们想预测明天台北下雨的概率。如果是夏季，雨可能会带有概率 0.5。

在这两种情况下，我们都有一定的兴趣价值。在这两种情况下，我们都不确定结果。但这两种情况之间有一个关键区别。在这第一种情况下，图像实际上是狗或猫，我们只是不知道哪个。在第二种情况下，结果实际上可能是一个随机的事件，如果你相信这样的事情 (和大多数物理学家做的)。因此，概率是一种灵活的语言来推理我们的确定性程度，并且可以在广泛的情况下有效应用。

## 基本概率论

假设我们投了一个模具，并想知道看到 1 而不是另一个数字的机会是多少。如果模具是公平的，所有六个结果 $\{1, \ldots, 6\}$ 都同样可能发生，因此我们将在六个案例中看到 $1$。形式上，我们声明，$1$ 发生的概率为 $\frac{1}{6}$。

对于我们从工厂收到的真实模具，我们可能不知道那些比例，我们需要检查它是否有污染。调查模具的唯一方法是多次投射并记录结果。对于每个模具，我们将观察到 $\{1, \ldots, 6\}$ 中的值。鉴于这些结果，我们希望调查观察每个结果的概率。

每个值的一种自然方法是对该值进行单独的计数，并将其除以总投注数。这给出了给定 * 事件 * 概率的 * 估计 *。* 大数量定律 * 告诉我们，随着投球数量的增加，这个估计将越来越接近真正的基础概率。在详细介绍这里发生的事情之前，让我们试试一下。

首先，让我们导入必要的软件包。

```{.python .input}
%matplotlib inline
from d2l import mxnet as d2l
from mxnet import np, npx
import random
npx.set_np()
```

```{.python .input}
#@tab pytorch
%matplotlib inline
from d2l import torch as d2l
import torch
from torch.distributions import multinomial
```

```{.python .input}
#@tab tensorflow
%matplotlib inline
from d2l import tensorflow as d2l
import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
```

接下来，我们将希望能够铸造模具。在统计数据中，我们称之为从概率分布 * 采样 * 中绘制示例的过程。将概率分配给多个离散选择的分布称为
*多项分配 *。我们将给出一个更正式的定义
*分发 * 后来，但在较高的水平上，将它看作是一个
事件的概率。

要绘制单个样本，我们只需传入概率向量。输出是另一个长度相同的向量：它在索引 $i$ 处的值是采样结果对应于 $i$ 的次数。

```{.python .input}
fair_probs = [1.0 / 6] * 6
np.random.multinomial(1, fair_probs)
```

```{.python .input}
#@tab pytorch
fair_probs = torch.ones([6]) / 6
multinomial.Multinomial(1, fair_probs).sample()
```

```{.python .input}
#@tab tensorflow
fair_probs = tf.ones(6) / 6
tfp.distributions.Multinomial(1, fair_probs).sample()
```

如果你运行一堆采样器，你会发现你每次都会得到随机值。与估计模具的公平性一样，我们通常希望从同一分布中生成许多样本。使用 Python `for` 循环执行此操作将是无法忍受的缓慢，所以我们使用的函数支持一次绘制多个样本，返回我们可能希望的任何形状的独立样本数组。

```{.python .input}
np.random.multinomial(10, fair_probs)
```

```{.python .input}
#@tab pytorch
multinomial.Multinomial(10, fair_probs).sample()
```

```{.python .input}
#@tab tensorflow
tfp.distributions.Multinomial(10, fair_probs).sample()
```

现在我们知道如何对模具进行样本，我们可以模拟 1000 个卷。然后，我们可以通过和计数, 每个 1000 卷之后, 多少次每个数字被卷.具体来说，我们计算相对频率作为真实概率的估计值。

```{.python .input}
counts = np.random.multinomial(1000, fair_probs).astype(np.float32)
counts / 1000
```

```{.python .input}
#@tab pytorch
# Store the results as 32-bit floats for division
counts = multinomial.Multinomial(1000, fair_probs).sample()
counts / 1000  # Relative frequency as the estimate
```

```{.python .input}
#@tab tensorflow
counts = tfp.distributions.Multinomial(1000, fair_probs).sample()
counts / 1000
```

因为我们从公平模具中生成数据，所以我们知道每个结果都有真实概率 $\frac{1}{6}$，大约为 $0.167$，所以上面的输出估计值看起来不错。

我们还可以直观地看到这些概率如何随着时间的推移而收敛到真实概率。让我们进行 500 组实验，每组抽取 10 个样本。

```{.python .input}
counts = np.random.multinomial(10, fair_probs, size=500)
cum_counts = counts.astype(np.float32).cumsum(axis=0)
estimates = cum_counts / cum_counts.sum(axis=1, keepdims=True)

d2l.set_figsize((6, 4.5))
for i in range(6):
    d2l.plt.plot(estimates[:, i].asnumpy(),
                 label=("P(die=" + str(i + 1) + ")"))
d2l.plt.axhline(y=0.167, color='black', linestyle='dashed')
d2l.plt.gca().set_xlabel('Groups of experiments')
d2l.plt.gca().set_ylabel('Estimated probability')
d2l.plt.legend();
```

```{.python .input}
#@tab pytorch
counts = multinomial.Multinomial(10, fair_probs).sample((500,))
cum_counts = counts.cumsum(dim=0)
estimates = cum_counts / cum_counts.sum(dim=1, keepdims=True)

d2l.set_figsize((6, 4.5))
for i in range(6):
    d2l.plt.plot(estimates[:, i].numpy(),
                 label=("P(die=" + str(i + 1) + ")"))
d2l.plt.axhline(y=0.167, color='black', linestyle='dashed')
d2l.plt.gca().set_xlabel('Groups of experiments')
d2l.plt.gca().set_ylabel('Estimated probability')
d2l.plt.legend();
```

```{.python .input}
#@tab tensorflow
counts = tfp.distributions.Multinomial(10, fair_probs).sample(500)
cum_counts = tf.cumsum(counts, axis=0)
estimates = cum_counts / tf.reduce_sum(cum_counts, axis=1, keepdims=True)

d2l.set_figsize((6, 4.5))
for i in range(6):
    d2l.plt.plot(estimates[:, i].numpy(),
                 label=("P(die=" + str(i + 1) + ")"))
d2l.plt.axhline(y=0.167, color='black', linestyle='dashed')
d2l.plt.gca().set_xlabel('Groups of experiments')
d2l.plt.gca().set_ylabel('Estimated probability')
d2l.plt.legend();
```

每条实体曲线对应于模具的六个值中的一个，并给出了我们估计的模具在每组实验之后估计出该值的概率。虚线给出了真实的底层概率。随着我们通过进行更多的实验获得更多的数据，$6$ 实体曲线会收敛到真实概率。

### 概率论公理

当处理模具的卷轴时，我们将集 $\mathcal{S} = \{1, 2, 3, 4, 5, 6\}$ 称为 * 样品空间 * 或 * 结果空间 *，其中每个元素都是 * 结果 *。* 事件 * 是来自给定样本空间的一组结果。实例，“看到 $5$”（$\{5\}$）和 “看到奇数”（$\{1, 3, 5\}$）都是滚动模具的有效事件。请注意，如果随机实验的结果是在事件 $\mathcal{A}$ 中，则事件 $\mathcal{A}$ 已经发生。也就是说，如果 $3$ 点滚动模具后面临，自 $3 \in \{1, 3, 5\}$ 以来，我们可以说，“看到奇数” 的事件发生了。

形式上，* 概率 * 可以被认为是将集合映射到真实值的函数。在给定的样本空间 $\mathcal{S}$ 中，事件的概率 $\mathcal{A}$，表示为 $P(\mathcal{A})$，满足以下属性：

* 对于任何事件 $\mathcal{A}$，其概率从不是负数，即 $P(\mathcal{A}) \geq 0$；
* 整个样本空间的概率为 $1$，即 $P(\mathcal{S}) = 1$；
* 对于任何事件 $\mathcal{A}_1, \mathcal{A}_2, \ldots$ 的可计数序列，这些事件相互排斥 *（全部 $i \neq j$ 为 $\mathcal{A}_i \cap \mathcal{A}_j = \emptyset$），发生任何事件的概率等于其各个概率的总和，即 $P(\bigcup_{i=1}^{\infty} \mathcal{A}_i) = \sum_{i=1}^{\infty} P(\mathcal{A}_i)$。

这些也是概率理论的公理，由科尔莫戈罗夫于 1933 年提出。由于这个公理系统，我们可以避免任何关于随机性的哲学争议；相反，我们可以用数学语言严格地推理。实例，通过让事件 $\mathcal{A}_1$ 为整个采样空间，而所有 $i > 1$ 为 $\mathcal{A}_i = \emptyset$，我们可以证明 $P(\emptyset) = 0$，即不可能发生事件的概率是 $0$。

### 随机变量

在铸模的随机实验中，我们引入了 * 随机变量 * 的概念。随机变量几乎可以是任何数量，并且不是确定性的。在随机实验中，它可能需要一组可能性中的一个值。考虑一个随机变量 $X$，其值在滚动模具的样本空间 $\mathcal{S} = \{1, 2, 3, 4, 5, 6\}$ 中。我们可以将 “看到一个 $5$” 的事件表示为 $\{X = 5\}$ 或 $X = 5$，其概率表示为 $P(\{X = 5\})$ 或 $P(X = 5)$。通过 $P(X = a)$，我们区分了随机变量 $X$ 和 $X$ 可以采取的值（例如 $a$）。然而，这种步伐会导致一个繁琐的符号。对于紧凑的表示法，一方面，我们可以将 $P(X)$ 表示为随机变量 $X$ 上的 * 分布 *：分布告诉我们 $X$ 获得任何值的概率。另一方面，我们可以简单地编写 $P(a)$ 来表示随机变量取值 $a$ 的概率。由于概率理论中的事件是来自样本空间的一组结果，因此我们可以为随机变量指定值范围。样本，$P(1 \leq X \leq 3)$ 表示事件的概率 $\{1 \leq X \leq 3\}$，这意味着 $\{X = 1, 2, \text{or}, 3\}$。等价地，$P(1 \leq X \leq 3)$ 表示随机变量 $X$ 可以从 $\{1, 2, 3\}$ 获得一个值的概率。

请注意，* 离散 * 随机变量（如模具的侧面）和 * 连续 * 变量（如人权重和身高）之间存在微妙的差异。问两个人是否具有完全相同的身高没有什么意义。如果我们进行足够精确的测量，你会发现这个星球上没有两个人具有完全相同的高度。事实上，如果我们采取足够精细的测量，你不会有相同的高度，当你醒来，当你去睡觉。因此，没有任何意义来询问一个人身高 1.80139278297192196202 米高的概率。鉴于世界人口的人类概率几乎是 0。在这种情况下，询问某人的身高是否落入给定的间隔，比如 1.79 米和 1.81 米之间更有意义。在这些情况下，我们将数值视为 * 密度 * 的可能性量化。正好 1.80 米的高度没有概率，但非零密度。在任何两个不同高度之间的间隔中，我们有非零概率。在本节的其余部分中，我们将考虑离散空间中的概率。对于连续随机变量的概率，您可以参考 :numref:`sec_random_variables`。

## 处理多个随机变量

很多时候，我们会想一次考虑多个随机变量。实例，我们可能需要模拟疾病和症状之间的关系。鉴于疾病和症状，说 “流感” 和 “咳嗽”，可能发生也可能不发生在患者有某种可能性。虽然我们希望两者的概率接近于零，但我们可能需要估计这些概率和它们之间的关系，以便我们可以运用我们的推断来实现更好的医疗服务。

作为一个更复杂的样本，图像包含数百万像素，因此数百万个随机变量。在许多情况下，图像会附带一个标签，标识图像中的对象。我们也可以将标签视为一个随机变量。我们甚至可以将所有元数据视为随机变量，例如位置、时间、光圈、焦距、ISO、对焦距离和相机类型。所有这些都是联合发生的随机变量。当我们处理多个随机变量时，有几个感兴趣的数量。

### 联合概率

第一个被称为 * 关节概率 * $P(A = a, B=b)$。给出任何值 $a$ 和 $b$, 联合概率让我们回答, 什么是概率是 $A=a$ 和 $B=b$ 同时?请注意，对于任何值，对于任何值，这种情况必须是这样，因为要发生 $A=a$ 和 $B=b$，就必须发生 * 和 * $A=a$ 也必须发生（反之亦然）。因此，单独的可能性不大于 $A=a$ 和 $B=b$ 的可能性。

### 条件概率

这给我们带来了一个有趣的比例：$0 \leq \frac{P(A=a, B=b)}{P(A=a)} \leq 1$。我们称这个比率为 * 条件概率 *，并用 $P(B=b \mid A=a)$ 表示它：它是 $B=b$ 的概率，前提是发生了 $A=a$。

### 贝耶斯定理

使用条件概率的定义，我们可以得出统计数据中最有用和最著名的方程之一：* Bayes 定理 *。它如下所示。通过建设，我们有 * 乘法规则 * 这 $P(A, B) = P(B \mid A) P(A)$.根据对称性，这也适用于 $P(A, B) = P(A \mid B) P(B)$。假设这样的情况。求解我们得到的条件变量之一

$$P(A \mid B) = \frac{P(B \mid A) P(A)}{P(B)}.$$

请注意，在这里我们使用更紧凑的表示法，其中 $P(A, B)$ 是一个 * 联合分布 *，$P(A \mid B)$ 是一个 * 条件分布 *。这种分布可以根据特定值 $A = a, B=b$ 进行评估。

### 边缘化

Bayes 定理是非常有用的，如果我们想从另一件事中推断一件事，比如因果，但我们只知道相反方向的属性，正如我们将在本节后面看到的那样。为了使这项工作，我们需要的一个重要操作是 * 边缘化 *。这是从 $P(A, B)$ 中确定 $P(B)$ 的操作。我们可以看到，$B$ 的概率相当于计算 $A$ 的所有可能选择，并将所有选择的联合概率聚合在一起：

$$P(B) = \sum_{A} P(A, B),$$

，也称为 * 总和规则 *。边缘化结果的概率或分布称为 * 边际概率 * 或 * 边际分布 *。

### 独立性

另一个要检查的有用属性是 * 依赖 * 与 * 独立 *。两个随机变量 $A$ 和 $B$ 是独立的，意味着一个事件的发生不会透露有关 $B$ 事件的发生情况的任何信息。在这种情况下，统计人员通常将这一点表达为 $A \perp  B$。从贝耶斯定理，它立即遵循这也是 $P(A \mid B) = P(A)$。在所有其他情况下，我们称 $A$ 和 $B$ 依赖。实例，一个模具的两个连续卷是独立的。相比之下，灯开关的位置和房间的亮度并不是（它们不是完全确定的，但是，因为我们总是可以有一个破碎的灯泡，电源故障，或者一个破碎的开关）。

由于 $P(A \mid B) = \frac{P(A, B)}{P(B)} = P(A)$ 等价于 $P(A, B) = P(A)P(B)$，因此当且仅当两个随机变量的联合分布是其各自分布的乘积时，两个随机变量是独立的。同样，两个随机变量 $A$ 和 $B$ 是 * 条件独立的 * 给定另一个随机变量 $C$，当且仅当 $P(A, B \mid C) = P(A \mid C)P(B \mid C)$ 时。这个数据表示为 $A \perp B \mid C$。

### 应用程序
:label:`subsec_probability_hiv_app`

让我们考验我们的技能。假设医生对患者进行艾滋病测试。这个测试是相当准确的，如果患者健康但报告他患病，则只有 1% 的概率失败。此外，如果患者真正拥有艾滋病毒，它永远不会发现艾滋病毒。我们使用 $D_1$ 来表示诊断（如果阳性，则为 $1$，如果阳性，则为 $0$）和 $H$ 来表示艾滋病病毒的状态（如果阳性，则为 $0$）。:numref:`conditional_prob_D1` 列出了这样的条件概率。

：条件概率为 $P(D_1 \mid H)$。

| Conditional probability | $H=1$ | $H=0$ |
|---|---|---|
|$P(D_1 = 1 \mid H)$|            1 |         0.01 |
|$P(D_1 = 0 \mid H)$|            0 |         0.99 |
:label:`conditional_prob_D1`

请注意，列总和都是 1（但行和不是），因为条件概率需要总和最多 1，就像概率一样。让我们计算出患者感染艾滋病的概率，如果测试回来呈阳性，即 $P(H = 1 \mid D_1 = 1)$。显然，这将取决于疾病的常见程度，因为它会影响虚假警报的数量。假设人口是相当健康的，例如，$P(H=1) = 0.0015$。为了应用贝耶斯定理，我们需要运用边缘化和乘法规则来确定

$$\begin{aligned}
&P(D_1 = 1) \\
=& P(D_1=1, H=0) + P(D_1=1, H=1)  \\
=& P(D_1=1 \mid H=0) P(H=0) + P(D_1=1 \mid H=1) P(H=1) \\
=& 0.011485.
\end{aligned}
$$

因此，我们得到

$$\begin{aligned}
&P(H = 1 \mid D_1 = 1)\\ =& \frac{P(D_1=1 \mid H=1) P(H=1)}{P(D_1=1)} \\ =& 0.1306 \end{aligned}.$$

换句话说，尽管使用了非常准确的测试，患者实际上患有艾滋病的几率只有 13.06%。正如我们所看到的，概率可能是违反直觉的。

患者在收到这样可怕的消息后应该怎么办？很可能，患者会要求医生进行另一次测试以获得清晰度。第二个测试具有不同的特性，它不如第一个测试那么好，如 :numref:`conditional_prob_D2` 所示。

：条件概率为 $P(D_2 \mid H)$。

| Conditional probability | $H=1$ | $H=0$ |
|---|---|---|
|$P(D_2 = 1 \mid H)$|            0.98 |         0.03 |
|$P(D_2 = 0 \mid H)$|            0.02 |         0.97 |
:label:`conditional_prob_D2`

不幸的是，第二次测试也回来了积极的，太。让我们通过假设条件独立性来计算出援引 Bayes 定理的必要概率：

$$\begin{aligned}
&P(D_1 = 1, D_2 = 1 \mid H = 0) \\
=& P(D_1 = 1 \mid H = 0) P(D_2 = 1 \mid H = 0)  \\
=& 0.0003,
\end{aligned}
$$

$$\begin{aligned}
&P(D_1 = 1, D_2 = 1 \mid H = 1) \\
=& P(D_1 = 1 \mid H = 1) P(D_2 = 1 \mid H = 1)  \\
=& 0.98.
\end{aligned}
$$

现在我们可以应用边缘化和乘法规则：

$$\begin{aligned}
&P(D_1 = 1, D_2 = 1) \\
=& P(D_1 = 1, D_2 = 1, H = 0) + P(D_1 = 1, D_2 = 1, H = 1)  \\
=& P(D_1 = 1, D_2 = 1 \mid H = 0)P(H=0) + P(D_1 = 1, D_2 = 1 \mid H = 1)P(H=1)\\
=& 0.00176955.
\end{aligned}
$$

最后，鉴于两个阳性检测，患者患有艾滋病的概率为

$$\begin{aligned}
&P(H = 1 \mid D_1 = 1, D_2 = 1)\\
=& \frac{P(D_1 = 1, D_2 = 1 \mid H=1) P(H=1)}{P(D_1 = 1, D_2 = 1)} \\
=& 0.8307.
\end{aligned}
$$

也就是说，第二次测试使我们能够获得更高的信心，即不是一切都很好。尽管第二次检验比第一次检验的准确性要低得多，但它仍然显著改善了我们的估计。

## 期望和差异

为了总结概率分布的关键特征，我们需要一些测量方法。随机变量 $X$ 的 * 期望 *（或平均值）表示为

$$E[X] = \sum_{x} x P(X = x).$$

当函数 $f(x)$ 的输入是从分布 $P$ 中抽取的随机变量时，将 $f(x)$ 的期望值计算为

$$E_{x \sim P}[f(x)] = \sum_x f(x) P(x).$$

在许多情况下，我们希望通过随机变量 $X$ 与其预期的偏差来衡量。这可以通过方差量化

$$\mathrm{Var}[X] = E\left[(X - E[X])^2\right] =
E[X^2] - E[X]^2.$$

它的平方根称为 * 标准差 *。随机变量函数的方差通过函数偏离函数的期望程度，因为随机变量的不同值 $x$ 从其分布中采样：

$$\mathrm{Var}[f(x)] = E\left[\left(f(x) - E[f(x)]\right)^2\right].$$

## 摘要

* 我们可以从概率分布中采样。
* 我们可以使用联合分布、条件分布、Bayes 定理、边缘化和独立性假设来分析多个随机变量。
* 预期和方差为概率分布的关键特征提供了有用的度量。

## 练习

1. 我们进行了 $m=500$ 组实验，每组抽取 $n=10$ 个样本。不同的情况下，也有差异。观察和分析实验结果。
1. 给定两个概率为 $P(\mathcal{A})$ 和 $P(\mathcal{B})$ 的事件，计算 $P(\mathcal{A} \cup \mathcal{B})$ 和 $P(\mathcal{A} \cap \mathcal{B})$ 的上限和下限。（提示：使用 [Venn Diagram](https://en.wikipedia.org/wiki/Venn_diagram) 显示情况。）
1. 假设我们有一系列随机变量，例如 $B$，$B$ 和 $C$，其中 $B$ 只依赖于 $A$，而 $C$ 只取决于 $B$，你能简化联合概率 $P(A, B, C)$ 吗？（提示：这是一个 [Markov Chain](https://en.wikipedia.org/wiki/Markov_chain)。）
1. 在 :numref:`subsec_probability_hiv_app` 中，第一个测试更准确。为什么不第二次运行第一次测试？

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/36)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/37)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/198)
:end_tab:
