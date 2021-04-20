# 编码器-解码器结构
:label:`sec_encoder-decoder`

正如我们在:numref:`sec_machine_translation`中所讨论的，机器翻译是序列转换模型的一个核心问题，其输入和输出都是可变长度序列。为了处理这种类型的输入和输出，我们可以设计一个包含两个主要组件的结构。第一个组件是一个*编码器*（encoder）：它接受一个可变长度的序列作为输入，并将其转换为具有固定形状的编码状态。第二个组件是*解码器*（decoder）：它将固定形状的编码状态映射到可变长度序列。这被称为*编码器-解码器*（encoder-decoder）结构。如:numref:`fig_encoder_decoder`所示。

![编码器-解码器结构](../img/encoder-decoder.svg)
:label:`fig_encoder_decoder`

让我们以英语到法语的机器翻译为例。给定一个英文的输入序列：“They”、“are”、“watching”、“.”，这种编码器-解码器结构首先将可变长度的输入编码成一个状态，然后对该状态进行解码，一个标记一个标记地生成翻译后的序列令牌作为输出：“Ils”、“regordent”、“.”。由于编码器-解码器结构构成了后续章节中不同序列转换模型的基础，因此本节将把该结构转换为稍后将实现的接口。

## 编码器

在编码器接口中，我们只指定编码器采用可变长度序列作为输入`X`。实现将由任何继承这个`Encoder`基类的模型提供。

```{.python .input}
from mxnet.gluon import nn

#@save
class Encoder(nn.Block):
    """编码器-解码器结构的基本编码器接口。"""
    def __init__(self, **kwargs):
        super(Encoder, self).__init__(**kwargs)

    def forward(self, X, *args):
        raise NotImplementedError
```

```{.python .input}
#@tab pytorch
from torch import nn

#@save
class Encoder(nn.Module):
    """编码器-解码器结构的基本编码器接口。"""
    def __init__(self, **kwargs):
        super(Encoder, self).__init__(**kwargs)

    def forward(self, X, *args):
        raise NotImplementedError
```

## 解码器

在下面的解码器接口中，我们添加了一个额外的`init_state`函数来将编码器输出（`enc_outputs`）转换为编码状态。请注意，此步骤可能需要额外的输入，例如输入的有效长度，这在:numref:`subsec_mt_data_loading`中进行了解释。为了逐个标记生成可变长度标记序列，每次解码器可将输入（例如，在前一时间步生成的标记）和编码状态映射到当前时间步的输出标记。

```{.python .input}
#@save
class Decoder(nn.Block):
    """编码器-解码器结构的基本解码器接口。"""
    def __init__(self, **kwargs):
        super(Decoder, self).__init__(**kwargs)

    def init_state(self, enc_outputs, *args):
        raise NotImplementedError

    def forward(self, X, state):
        raise NotImplementedError
```

```{.python .input}
#@tab pytorch
#@save
class Decoder(nn.Module):
    """编码器-解码器结构的基本解码器接口。"""
    def __init__(self, **kwargs):
        super(Decoder, self).__init__(**kwargs)

    def init_state(self, enc_outputs, *args):
        raise NotImplementedError

    def forward(self, X, state):
        raise NotImplementedError
```

## 把编码器和解码器合并

最后，编码器-解码器结构包含编码器和解码器，并包含可选的额外的参数。在前向传播中，编码器的输出产生“编码状态”，解码器将使用该状态作为其输入之一。

```{.python .input}
#@save
class EncoderDecoder(nn.Block):
    """编码器-解码器结构的基类。"""
    def __init__(self, encoder, decoder, **kwargs):
        super(EncoderDecoder, self).__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, enc_X, dec_X, *args):
        enc_outputs = self.encoder(enc_X, *args)
        dec_state = self.decoder.init_state(enc_outputs, *args)
        return self.decoder(dec_X, dec_state)
```

```{.python .input}
#@tab pytorch
#@save
class EncoderDecoder(nn.Module):
    """编码器-解码器结构的基类。"""
    def __init__(self, encoder, decoder, **kwargs):
        super(EncoderDecoder, self).__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, enc_X, dec_X, *args):
        enc_outputs = self.encoder(enc_X, *args)
        dec_state = self.decoder.init_state(enc_outputs, *args)
        return self.decoder(dec_X, dec_state)
```

编码器-解码器体系结构中的术语“状态”可能启发你使用具有状态的神经网络来实现该结构。在下一节中，我们将看到如何应用循环神经网络来设计基于这种编码器-解码器结构的序列转换模型。

## 小结

* 编码器-解码器结构可以处理可变长度序列的输入和输出，因此适用于机器翻译等序列转换问题。
* 编码器以可变长度序列作为输入，将其转换为具有固定形状的状态。
* 解码器将固定形状的编码状态映射到可变长度序列。

## 练习

1. 假设我们使用神经网络来实现编解码结构。编码器和解码器必须是同一类型的神经网络吗？
1. 除了机器翻译，你能想到另一个可以应用编码器-解码器结构的应用吗？

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/341)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/1061)
:end_tab:
