# 编码器解码器架构
:label:`sec_encoder-decoder`

正如我们在 :numref:`sec_machine_translation` 中所讨论的那样，机器翻译是序列转导模型的一个主要问题领域，其输入和输出都是可变长度序列。为了处理这种类型的输入和输出，我们可以设计一个包含两个主要组件的架构。第一个组件是 * 编码器 *：它采用可变长度序列作为输入，并将其转换为具有固定形状的状态。第二个组件是 * 解码器 *：它将固定形状的编码状态映射到可变长度序列。这就是所谓的 * 编码器-解码器 * 体系结构，在 :numref:`fig_encoder_decoder` 中描述。

![The encoder-decoder architecture.](../img/encoder-decoder.svg)
:label:`fig_encoder_decoder`

让我们以英语到法语的机器翻译为例。给定英语输入序列:“他们”, “是”, “看”,”。”，此编码器解码器体系结构首先将可变长度输入编码为状态，然后将状态解码以通过令牌生成转换的序列令牌作为输出：“Iils”，“Rededent”，“.”。由于编码器-解码器架构构成了后续章节中不同序列转导模型的基础，因此本节将将此架构转换为稍后实现的接口。

## 编码器

在编码器界面中，我们只需指定编码器采用可变长度序列作为输入 `X`。该实现将由任何继承此基础 `Encoder` 类的模型提供。

```{.python .input}
from mxnet.gluon import nn

#@save
class Encoder(nn.Block):
    """The base encoder interface for the encoder-decoder architecture."""
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
    """The base encoder interface for the encoder-decoder architecture."""
    def __init__(self, **kwargs):
        super(Encoder, self).__init__(**kwargs)

    def forward(self, X, *args):
        raise NotImplementedError
```

## 解码器

在下面的解码器接口中，我们添加了一个额外的 `init_state` 函数，将编码器输出 (`enc_outputs`) 转换为编码状态。请注意，此步骤可能需要额外的输入，例如输入的有效长度，在 :numref:`subsec_mt_data_loading` 中对此进行了说明。要按令牌生成可变长度序列令牌，每次解码器都可以将输入（例如，在上一个时间步长生成的标记）和编码状态映射到当前时间步长的输出标记。

```{.python .input}
#@save
class Decoder(nn.Block):
    """The base decoder interface for the encoder-decoder architecture."""
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
    """The base decoder interface for the encoder-decoder architecture."""
    def __init__(self, **kwargs):
        super(Decoder, self).__init__(**kwargs)

    def init_state(self, enc_outputs, *args):
        raise NotImplementedError

    def forward(self, X, state):
        raise NotImplementedError
```

## 将编码器和解码器组合在一起

最后，编码器-解码器架构包含一个编码器和一个解码器，并可选择附加参数。在正向传播中，编码器的输出用于生成编码状态，解码器将进一步使用此状态作为其输入之一。

```{.python .input}
#@save
class EncoderDecoder(nn.Block):
    """The base class for the encoder-decoder architecture."""
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
    """The base class for the encoder-decoder architecture."""
    def __init__(self, encoder, decoder, **kwargs):
        super(EncoderDecoder, self).__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, enc_X, dec_X, *args):
        enc_outputs = self.encoder(enc_X, *args)
        dec_state = self.decoder.init_state(enc_outputs, *args)
        return self.decoder(dec_X, dec_state)
```

编码器解码器架构中的术语 “状态” 可能启发了您使用具有状态的神经网络来实现这个架构。在下一节中，我们将介绍如何将 RNS 应用于基于此编码器-解码器架构设计序列转导模型。

## 摘要

* 编码器-解码器架构可以处理同时属于可变长度序列的输入和输出，因此适用于序列转导问题，例如机器翻译。
* 编码器采用可变长度序列作为输入，并将其转换为具有固定形状的状态。
* 解码器将固定形状的编码状态映射到可变长度序列。

## 练习

1. 假设我们使用神经网络来实现编码器解码器架构。编码器和解码器是否必须是相同类型的神经网络？
1. 除了机器翻译之外，你能想到另一个可以应用编码器-解码器架构的应用吗？

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/341)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/1061)
:end_tab:
