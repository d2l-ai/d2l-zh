# 编码器解码器架构
:label:`sec_encoder-decoder`

正如我们在 :numref:`sec_machine_translation` 中所讨论的那样，机器翻译是序列转导模型的主要问题领域，其输入和输出都是可变长度序列。为了处理这种类型的输入和输出，我们可以设计一个包含两个主要组件的架构。第一个组件是 *编码器 *：它采用可变长度序列作为输入，然后将其转换为具有固定形状的状态。第二个组件是 *decoder*：它将固定形状的编码状态映射到可变长度序列。这被称为 * 编码器-解码器 * 体系结构，在 :numref:`fig_encoder_decoder` 中进行了描述。

![The encoder-decoder architecture.](../img/encoder-decoder.svg)
:label:`fig_encoder_decoder`

让我们以英语到法语的机器翻译为例。给定英语输入序列：“他们”、“是”、“看”，”。“，这种编码器解码器架构首先将可变长度输入编码为一个状态，然后解码状态，然后通过令牌生成翻译的序列标记作为输出：“Ir”、“视察”、“.”。由于编码器解码器体系结构构成了后续章节中不同序列转导模型的基础，因此本节将此架构转换为稍后实现的接口。

## 编码器

在编码器界面中，我们只需指定编码器采用可变长度序列作为输入 `X`。该实现将由继承此基础 `Encoder` 类的任何模型提供。

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

在下面的解码器界面中，我们添加了一个额外的 `init_state` 函数，将编码器输出 (`enc_outputs`) 转换为编码状态。请注意，此步骤可能需要额外的输入，例如输入的有效长度，在 :numref:`subsec_mt_data_loading` 中对此进行了解释。要通过令牌生成可变长度序列令牌，每次解码器都可能在当前时间步将输入（例如，上一个时间步生成的令牌）和编码状态映射到输出令牌时。

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

## 将编码器和解码器放在一起

最后，编码器解码器架构包含编码器和解码器，并可选择附加参数。在向前传播中，编码器的输出用于产生编码状态，解码器将进一步使用此状态作为其输入之一。

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

编码器解码器架构中的 “状态” 一词可能激发了你使用带状态的神经网络来实现这种架构。在下一节中，我们将了解如何应用 RNN 来设计基于此编码器解码器架构的序列转导模型。

## 摘要

* 编码器解码器架构可以处理同时属于可变长度序列的输入和输出，因此适用于机器翻译等序列转导问题。
* 编码器采用可变长度序列作为输入，并将其转换为具有固定形状的状态。
* 解码器将固定形状的编码状态映射到可变长度序列。

## 练习

1. 假设我们使用神经网络来实现编码器解码器架构。编码器和解码器必须是同一类型的神经网络吗？
1. 除了机器翻译之外，你能想到另一个可以应用编码器-解码器架构的应用程序吗？

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/341)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/1061)
:end_tab:
