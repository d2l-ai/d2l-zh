

<!--
 * @version:
 * @Author:  StevenJokes https://github.com/StevenJokes
 * @Date: 2020-07-03 20:41:38
 * @LastEditors:  StevenJokes https://github.com/StevenJokes
 * @LastEditTime: 2020-07-03 20:42:31
 * @Description:
 * @TODO::
 * @Reference:
-->

# 注意力机制

在[9.7节](http://preview.d2l.ai/d2l-en/PR-1102/chapter_recurrent-modern/seq2seq.html#sec-seq2seq)中，我们将源序列输入信息编码为周期性单元状态，然后将其传递给解码器生成目标序列。目标序列中的标记可能与源序列中的一个或多个标记密切相关，而不是整个源序列。例如，当翻译Hello world为Bonjour le monde。例如，Bonjour映射到Hello, monde映射到world。在seq2seq模型中，解码器可以隐式地从编码器传递的状态中选择相应的信息。

注意力是一种通用的合并方法，其输入上存在偏差对齐。 注意机制的核心组件是注意层，为简单起见称为注意力。 注意层的输入称为查询。 对于查询，注意力返回基于内存的输出-记忆层中编码的一组键-值对。 更具体地说，假设存储器包含n个键值对（k1，v1），…，（kn，vn），其中ki∈Rdk，vi∈Rdv。 给定查询q∈Rdq，注意层返回形状与值相同的输出o∈Rdv。

TODO:FIG

注意力机制的整个过程如[图10.1.2](http://preview.d2l.ai/d2l-en/PR-1102/chapter_attention-mechanisms/attention.html#fig-attention-output)所示。为了计算注意力的输出，我们首先使用一个度量查询和关键字之间相似性的得分函数，即数值函数。对于每个键k1, knk1, kn，我们计算分数a1, ana1, an，通过

TODO:MATH

接下来我们使用softmax来获得注意权重，即

TODO:MATH

最后，输出是这些值的加权和:

TODO:MATH

TODO:FIG

分数功能的不同选择导致了不同的注意层次。下面，我们介绍两个常用的注意层。在深入实现之前，我们首先要表达两个操作符来启动和运行:一个softmax操作符的掩码版本和一个专门的点操作符`batch dot`。

mask softmax接受三维输入，并允许我们通过指定最后一个维度的有效长度来过滤一些元素。(有关有效长度的定义，请参阅[第9.5节](http://preview.d2l.ai/d2l-en/PR-1102/chapter_recurrent-modern/machine-translation-and-dataset.html#sec-machine-translation)。)因此，任何超出有效长度的值都将被掩盖为 0 。让我们实现`mask_softmax`函数。

TODO:CODE

为了说明这个函数是如何工作的，我们构造两个 2×4 矩阵作为输入。此外，我们指定第一个示例的有效长度为2，第二个示例为3。然后，正如我们从以下输出中看到的，有效长度以外的值被屏蔽为零。

TODO:CODE

此外，第二个运算符`batch_dot`分别采用形状为（b，n，m）和（b，m，k）的两个输入X和Y，并返回形状为（b，n，k）的输出。 具体来说，它会计算i = {1，…，b}的b点乘积，即

TODO:CODE

TODO:CODE

## 点积注意力

配备了以上两个运算符：masked_softmax和batch_dot，让我们深入研究两个广泛使用的关注层的细节。 第一个是点积注意事项：它假设查询的维数与键相同，即所有i的q，ki∈Rd。 点积注意力通过查询和键之间的点积计算分数，然后将其除以d-√以最小化维度d对分数的不相关影响。 换一种说法，

TODO:MATH

除了一维查询和键之外，我们总是可以将它们泛化为多维查询和键。假设Q Rm dQ Rm d包含mm查询，K Rn dK Rn d拥有所有的nn键。我们可以计算所有的mnmn分数

TODO:MATH

使用[(10.1.6)](http://preview.d2l.ai/d2l-en/PR-1102/chapter_attention-mechanisms/attention.html#equation-eq-alpha-qk)，我们可以实现点积注意层DotProductAttention，它支持一批查询和键值对。此外，为了正则化，我们还使用了一个dropout层。

TODO:CODE

让我们在一个玩具样例中测试类`DotProductAttention`。首先，创建两个批处理，其中每个批处理有一个查询和10个键-值对。通过valid len参数，我们指定将检查第一个批处理的前22个键值对和第二个批处理的 6 个键值对。因此，即使这两个批具有相同的查询和键-值对，我们也会获得不同的输出。

TODO:CODE

正如我们在上面看到的，点积注意力只是将查询和键相乘，并希望从中得到它们的相似性。然而，查询和键可能不是同一维的。为了解决这个问题，我们可以求助于多层感知器注意。

## 多层感知器

在多层感知器注意中，我们通过可学习的权重参数将查询和键都投影到Rh中。 假设可学习的权重为Wk∈Rh×dk，Wq∈Rh×dq和v∈Rh。 然后得分函数定义为

TODO:MATH

直观上，你可以把 Wkk+Wqq 想象为将特征维中的键和值连接起来，并将它们输入到一个隐含层大小为 h 、输出层大小为 1 的单层感知器中。在这个隐层中，激活函数为 tanh ，不存在偏置。现在让我们实现多层感知器注意。

TODO:CODE

为了测试上面的`MLPAttention`类，我们使用与上一个玩具示例相同的输入。正如我们在下面看到的，尽管`MLPAttention`包含了一个额外的MLP模型，但我们得到的输出与`DotProductAttention`相同。

TODO:CODE

## 小结

- 注意层显式地选择相关信息。
- 注意层的内存由键-值对组成，因此它的输出接近键与查询相似的值。
- 两种常用的注意模型是点积注意和多层感知器注意。

## 练习

1. 点积关注和多层感知器关注分别有什么优点和缺点?
