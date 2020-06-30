

<!--
 × @version:
 × @Author:  StevenJokes https://github.com/StevenJokes
 × @Date: 2020-06-30 13:43:26
 * @LastEditors:  StevenJokes https://github.com/StevenJokes
 * @LastEditTime: 2020-06-30 15:30:09
 × @Description:translate
 × @TODO::
 × @Reference:http://preview.d2l.ai/d2l-en/PR-1080/chapter_generative-adversarial-networks/dcgan.html
-->

# 深度卷积生成对抗网络

在本节中，我们将演示如何使用GANs生成逼真的图像。我们的模型将基于[Radford et al.， 2015](http://preview.d2l.ai/d2l-en/PR-1080/chapter_references/zreferences.html#radford-metz-chintala-2015)中介绍的deep convolutional GANs (DCGAN)。我们将借用卷积架构，已经证明如此成功的甄别计算机视觉问题，并展示如何通过GANs，他们可以被用来产生逼真的图像。

## Pokemon宠物小精灵数据集

我们将使用的数据集是Pokemon宠物小精灵从[Pokemon db](https://pokemondb.net/sprites)获得的。首先下载、提取和加载此数据集。

TODO:CODE

我们调整每个图像到维度为64×64。ToTensor变换将像素值投影到$[0,1]$。而我们的生成器将使用tanh函数来获取$[1,1]$中的输出。因此，我们以$0.5$均值和$0.5$标准差对数据进行归一化，以匹配取值范围。

TODO:CODE

让我们将前20幅图像可视化。

TODO:CODE

## 生成器

生成器需要映射噪声变量$z∈R^d$，一个d维向量，到一个$64×64$的RGB图像。在[13.11节](http://preview.d2l.ai/d2l-en/PR-1080/chapter_computer-vision/fcn.html#sec-fcn)中，我们介绍了使用转置卷积层来扩大输入大小的全卷积网络(参见[13.10节](http://preview.d2l.ai/d2l-en/PR-1080/chapter_computer-vision/transposed-conv.html#sec-transposed-conv))。生成器的基本块包含一个转置卷积层，然后是批处理归一化和ReLU激活。

在默认情况下，转置卷积层使用$k_h=k_w=4$的核，一个$s_h=s_w=2$的步进strides，和一个$p_h=p_w=1$的填充padding。当输入形状为$n_h×n_w =16×16$时，生成块将输入的宽度和高度加倍。

TODO:CODE

如果把转置的卷积层换成一个$4×4$核，$1×1$的步进strides和$0$的填充padding。当输入大小为$1×1$时，输出的宽度和高度将分别增加3。

TODO:CODE

生成器由四个基本块组成，这些块将输入的宽度和高度从1增加到32。同时，它首先将潜变量投影到$64×8$通道中，然后每次将通道减半。 最后，转置的卷积层用于生成输出。 它将宽度和高度进一步加倍以匹配所需的$64×64$形状，并将通道大小减小到$3$。 tanh激活功能适用于将输出值投影到$(-1,1)$范围内。

TODO:CODE

生成一个100维的潜在变量来验证生成器的输出形状。

TODO:CODE

## 鉴别器

可以看出，如果$a =0$，则ReLU正常;如果$a =1$，则为identity函数。对于$α∈(0,1)$，leaky ReLU是一个非线性函数，它为负输入提供非零输出。它旨在解决"dying ReLU"问题，即由于ReLU的梯度为0，神经元可能总是输出一个负值，因此无法取得任何进展。

TODO:CODE

带有默认设置的基本块将把输入的宽度和高度减半，正如我们在6.3节中演示的那样。例如，给定输入形状$n_h=n_w=16$，核形状$k_h=k_w=4$，步幅形状$s_h=s_w=2$，填充形状$p_h=p_w=1$，则输出形状为
鉴别器是生成器的镜像。

TODO:MATH

TODO:CODE

它使用输出通道$1$作为最后一层的卷积层来获得单个预测值。

TODO:CODE


## 训练

与[17.1节](http://preview.d2l.ai/d2l-en/PR-1080/chapter_generative-adversarial-networks/gan.html#sec-basic-gan)中的基本GAN相比，我们对生成器和鉴别器使用相同的学习速率，因为它们彼此相似。此外，我们将Adam([第11.10节](http://preview.d2l.ai/d2l-en/PR-1080/chapter_optimization/adam.html#sec-adam))中的$β1$从$0.9$改为$0.5$。它降低了动量的平顺性，过去梯度的指数加权移动平均，以照顾快速变化的梯度，因为产生器和鉴别器互相争斗。另外，随机产生的噪声Z，是一个四维张量，我们使用GPU来加速计算。

TODO:CODE

我们用少量的epochs来训练模型，只是为了演示。为了获得更好的性能，变量num epoch可以设置为更大的数字。

TODO:CODE

## 小结

- DCGAN结构的鉴别器有四个卷积层，发生器有四个分条纹卷积层。
- 鉴别器是一个4层条纹卷积与批处理规范化(除了它的输入层)和泄漏继电器激活。
- leaky ReLU是一个非线性函数，在负输入时给出非零输出。它的目标是修复逐渐消失的ReLU问题，并帮助梯度更容易地通过架构。

## 练习

1. 如果我们使用标准ReLU激活而不是leaky ReLU，会发生什么?
1. 将DCGAN应用到Fashion-MNIST上，看看哪个类别比较好，哪个不合适。
