

<!--
 * @version:
 * @Author:  StevenJokes https://github.com/StevenJokes
 * @Date: 2020-07-03 19:33:12
 * @LastEditors:  StevenJokes https://github.com/StevenJokes
 * @LastEditTime: 2020-07-03 20:38:58
 * @Description:translate
 * @TODO::
 * @Reference:http://preview.d2l.ai/d2l-en/PR-1102/chapter_attention-mechanisms/seq2seq-attention.html
-->

# 带注意力机制的序列到序列

在本节中，我们将注意力机制添加到[9.7节](http://preview.d2l.ai/d2l-en/PR-1102/chapter_recurrent-modern/seq2seq.html#sec-seq2seq)中介绍的序列到序列(seq2seq)模型中，以显式地用权重聚合状态。[图10.2.1](http://preview.d2l.ai/d2l-en/PR-1102/chapter_attention-mechanisms/seq2seq-attention.html#fig-s2s-attention)显示了在时间步长 t 进行编码和解码的模型架构。在这里，注意层的记忆由编码器在每个时间步长所看到的编码器输出的所有信息组成。在解码期间，使用前一个时间步长 t-1 的解码器输出作为查询。注意模型的输出被视为上下文信息，并与 D_t 连接。最后，我们将串联输入译码器。

TODO:FIG

为了用attention模型说明seq2seq的整体架构，其编码器和解码器的层结构如[图10.2.2](http://preview.d2l.ai/d2l-en/PR-1102/chapter_attention-mechanisms/seq2seq-attention.html#fig-s2s-attention-details)所示。

TODO:FIG

TODO:CODE

由于带有注意机制的seq2seq编码器与9.7节中的Seq2SeqEncoder相同，所以我们将只关注解码器。我们增加了一个MLP注意层(MLPAttention)，它与译码器中的LSTM层具有相同的隐藏大小。然后，我们通过传递来自编码器的三个项来初始化解码器的状态。

- 编码器输出的所有时间步长:它们作为注意层存储器，具有相同的键和值;
- 编码器最终时间步长隐藏状态:作为初始解码器的隐藏状态;
- 编码器有效长度:因此注意层将不会考虑编码器输出中的填充令牌。

在解码的每个时间步长，我们使用解码器的最后一层RNN的输出作为关注层的查询。然后将注意力模型的输出与输入嵌入向量连接到RNN层中。虽然RNN层隐藏状态也包含了解码器的历史信息，但是attention输出显式地选择了基于`enc_valid_len`的编码器输出，使得attention输出挂起了其他不相关的信息。

让我们实现`Seq2SeqAttentionDecoder`，看看它与[9.7.2节](http://preview.d2l.ai/d2l-en/PR-1102/chapter_recurrent-modern/seq2seq.html#sec-seq2seq-decoder)中的seq2seq中的解码器有什么不同。

TODO:CODE

现在我们可以用注意力模型来测试seq2seq。为了与[9.7节](http://preview.d2l.ai/d2l-en/PR-1102/chapter_recurrent-modern/seq2seq.html#sec-seq2seq)中的模型保持一致，我们对`vocab_size`, `embed_size`, `num_hiddens`, and `num_layers`. 使用相同的超参数。结果，我们得到相同的解码器输出形状，但状态结构改变。

TODO:CODE

与[第9.7.4节](http://preview.d2l.ai/d2l-en/PR-1102/chapter_recurrent-modern/seq2seq.html#sec-seq2seq-training)类似，我们使用相同的训练超参数和相同的训练损失来尝试一个玩具模型。从结果中我们可以看出，由于训练数据集中的序列相对较短，额外的注意层并没有带来显著的改善。由于编码器和解码器注意层的计算开销，该模型比不注意时的seq2seq模型要慢得多。

最后，我们预测了几个示例。

TODO:CODE

- 带有注意力的seq2seq模型向不带有注意力的模型添加了额外的注意力层。
- 带有注意力模型的seq2seq解码器从编码器中传递三项内容:编码器所有时间步长的输出、编码器最终时间步长的隐藏状态和编码器有效长度。

## 练习

1. 使用相同的参数比较Seq2SeqAttentionDecoder和Seq2seqDecoder，并检查它们的损耗。
1. 你能想到Seq2SeqAttentionDecoder比Seq2seqDecoder更好的用例吗
