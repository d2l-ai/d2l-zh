# 序言

就在几年前，大型公司和初创公司没有一批深度学习科学家开发智能产品和服务。当我们中最年轻的人（作者）进入这个领域时，机器学习并没有在日报上获得头条新闻。我们的父母不知道什么是机器学习，更不用说为什么我们更喜欢它，而不是喜欢医学或法律。机器学习是一门具有前瞻性的学术学科，其中包括一系列狭窄的实际应用程序。这些应用程序，例如语音识别和计算机视觉，需要太多的领域知识，以至于它们往往被视为完全独立的领域，机器学习只是一个小组成部分。然后，神经网络，我们在这本书中关注的深度学习模型的鼻祖，被认为是过时的工具。

在过去的五年里，深度学习让世界感到惊讶，推动了计算机视觉、自然语言处理、自动语音识别、强化学习和统计建模等不同领域的快速发展。随着这些进步，我们现在可以打造出比以往任何时候都更具自主性的汽车（而且自主性不如一些公司想象的那样），自动起草简单的电子邮件智能回复系统，帮助人们从压迫性的大型收件箱中挖掘出来，以及软件代理人在围棋游戏中打败了世界上最棒的人，如 Go，这一壮举曾经被认为是几十年之遥。这些工具已经对工业和社会产生了越来越广泛的影响，改变了电影制作方式，诊断疾病，并在从天体物理学到生物学的基础科学中扮演着越来越大的角色。

## 关于本书

这本书代表了我们试图让深度学习变得平易近人，教你 *概念*、*背景* 和 *代码*。

### 一种结合代码、数学和 HTML 的媒介

为了使任何计算技术发挥全面影响，它必须得到充分理解、有充分记录，并得到成熟、维护良好的工具的支持。应当清楚地提炼关键的想法，最大限度地减少新从业人员所需的入门时间，让新从业人员了解最新情况。成熟的库应该自动执行常见任务，示例代码应该使从业人员能够轻松修改、应用和扩展常见应用程序以满足他们的需求。以动态 Web 应用程序为样本。尽管像亚马逊这样的许多公司在 1990 年代开发了成功的数据库驱动的 Web 应用程序，但在过去十年中，这项技术在帮助初创企业方面的潜力得到了更大程度的实现，部分原因是开发了强大的、有文件记录的框架.

测试深度学习的潜力带来了独特的挑战，因为任何一个应用程序都汇集了不同的学科。应用深度学习需要同时了解 (一) 以特定方式解决问题的动机；(二) 特定模型方法的数学；(三) 将模型与数据拟合的优化算法；(四) 以及有效训练模型所需的工程，浏览数字计算的缺陷，并充分利用可用硬件。教授制定问题所需的批判性思维技巧、解决问题的数学技巧以及实施这些解决方案的软件工具都是艰巨的挑战。我们在这本书中的目标是提供一个统一的资源，以帮助潜在的从业者加快速度。

在我们开始这本书项目的时候，没有资源可以同时 (i)是最新的；(ii) 涵盖了现代机器学习的全部内容，具有相当的技术深度；(iii) 交叉展示了一本引人入胜的教科书所期望的质量，它包含了人们期望在实际操作教程中找到的干净的可运行代码。我们发现了很多代码示例，用于如何使用给定的深度学习框架（例如，如何在 TensorFlow 中使用矩阵进行基本数字计算）或实现特定技术（例如 LenNet，AlexNet，ResNet 等的代码片段）分布在各种博客文章和 GitHub 存储库中。但是，这些示例通常侧重于
*如何* 实施给定的方法，
但忽视了 *为什么* 某些算法决策的讨论。虽然一些互动资源偶尔出现，以解决某个特定主题，例如在网站 [Distill](http://distill.pub) 上发布的引人入胜的博客文章或个人博客，但它们只涉及深度学习中的选定主题，而且往往缺乏相关代码。另一方面，虽然出现了一些教科书，最值得注意的是 :cite:`Goodfellow.Bengio.Courville.2016`，它对深度学习背后的概念进行了全面的调查，但这些资源并没有将描述与代码中的概念的实现结合起来，有时让读者对如何实现这些概念毫不知道。此外，太多的资源隐藏在商业课程提供者的支付壁垒后面。

我们着手创造一种资源，以便 (i) 每个人都可以免费使用；(ii) 提供足够的技术深度，为实际成为应用机器学习科学家的道路提供一个起点；(iii) 包括可运行的代码，向读者展示如何解决实践中的问题；(iv) 允许以获得我们和整个社区的快速更新；以及 (v) 辅以[论坛](http://discuss.d2l.ai)，用于互动讨论技术细节和回答问题。

这些目标往往发生冲突。公示、定理和引用最好在 LaTeX 中管理和布局。代码在 Python 中最好地描述。网页是原生的 HTML 和 JavaScript。此外，我们希望这些内容既可以作为可执行代码、实体书籍、可下载的 PDF，也可以作为网站在互联网上访问。目前没有工具，也没有完全适合这些需求的工作流程，所以我们必须组装自己的工作流程。我们在 :numref:`sec_how_to_contribute` 中详细描述了我们的方法。我们通过 GitHub 来共享源代码并允许编辑，Jupyter notebooks用于混合代码、公式和文本，Sphinx作为渲染引擎来生成多个输出，Discourse作为论坛。虽然我们的系统还不完善，但这些选择在相互竞争的关注点之间提供了一个很好的折衷方案。
我们相信这可能是第一本使用这种集成工作流程出版的书。

### 在实践中学习

许多教科书教授一系列主题，每一个都详尽无遗。样本，克里斯·毕夏普出色的教科书 :cite:`Bishop.2006`，教授每个主题如此彻底，到达关于线性回归的章节需要大量的工作。虽然专家喜欢这本书正是因为它的彻底性，对于初学者来说，这个属性限制了它作为一个介绍性文本的有用性。

在这本书中，我们将教授大多数概念 *在需要的时候*。换句话说，你将在为了实现一些实际目的而需要的时候，学习概念。虽然我们在一开始就需要一些时间来教授基础知识，比如线性代数和概率，我们希望您在担心更深奥的概率分布之前，先体验一下训练第一个模型的满足感。

除了一些提供速成课程的初步笔记本。在基本的数学背景下，接下来的每一章都会介绍合理数量的新概念，并提供一个独立的工作实例——使用真实的数据集。这是一个组织方面的挑战。有些模型可以逻辑地组合在一个笔记本中。一些想法可能最好通过连续执行几个模型来教授。另一方面，遵守 *1可工作的示例，1笔记本* 的政策有一个很大的优势：这使您可以通过利用我们的代码来启动自己的研究项目。只需复制笔记本并开始修改它。

我们将根据需要将可运行的代码与背景材料交替。一般来说，在完全解释之前，我们经常会在提供工具方面出错（我们将在稍后解释背景）。在完全解释为什么它有用或为什么它有效之前，我们可能会使用 *随机梯度下降*。这有助于给从业者提供必要的弹药，以快速解决问题，但代价是要求读者信任我们，并提供一些策展决策。

这本书将从头开始教授深度学习概念。有时，我们希望深入研究有关模型的细节，这些模型通常会通过深度学习框架的高级抽象隐藏在用户身上。这尤其是在基本教程中出现，我们希望您了解给定层或优化器中发生的一切。在这些情况下，我们经常提供两个版本的样本：一个版本我们从头开始实现所有内容，仅依赖 NumPy 接口和自动微分，另一个更实用的例子，我们使用Gluon编写简洁的代码。一旦我们教你一些组件是如何工作的，我们可以在后续教程中使用 Gluon版本。

### 内容和结构

这本书可以大致分为三个部分，这些部分是由不同的颜色在 :numref:`fig_book_org`：

![Book structure](../img/book-org.svg)
:label:`fig_book_org`

* 第一部分包括基础知识和预备。
:numref:`chap_introduction` 提供了深度学习的介绍。然后，在 :numref:`chap_preliminaries` 中，我们可以快速了解深度学习所需的先决条件，例如如何存储和操作数据，以及如何根据线性代数、微积分和概率等基本概念应用各种数值运算。:numref:`chap_linear` 和 :numref:`chap_perceptrons` 涵盖了最多的深度学习的基本概念和技术, 例如线性回归, 多层感知和正则化.

* 接下来的五章重点介绍现代深度学习技巧。
:numref:`chap_computation` 描述了深度学习计算的各种关键组成部分，并为我们随后实施更复杂的模型奠定了基础。接下来，在 :numref:`chap_cnn` 和 :numref:`chap_modern_cnn` 中，我们介绍了卷积神经网络（CNN），这个功能强大的工具构成了大多数现代计算机视觉系统的骨干。随后在 :numref:`chap_rnn` 和 :numref:`chap_modern_rnn` 中，我们引入了循环神经网络 (RNN)，这些模型利用数据中的时间或顺序结构，通常用于自然语言处理和时间序列预测。在 :numref:`chap_attention` 中，我们引入了一类新的模型，它们采用了一种称为注意力机制的技术，并且它们最近开始在自然语言处理中取代RNN系列。这些部分将帮助您了解大多数现代深度学习应用背后的基本工具。

* 第三部分讨论可扩展性、效率和应用程序。
首先，在 :numref:`chap_optimization` 中，我们讨论了用于训练深度学习模型的几种常见优化算法。下一章 :numref:`chap_performance` 探讨了影响深度学习代码计算性能的几个关键因素。在 :numref:`chap_cv` 中，我们展示了深度学习在计算机视觉中的主要应用。在 :numref:`chap_nlp_pretrain` 和 :numref:`chap_nlp_app` 中，我们演示了如何预训练语言表示模型并将其应用于自然语言处理任务。

### 代码
:label:`sec_code`

本书的大部分章节都以可执行代码为特色，因为我们相信在深度学习中交互式学习体验的重要性。目前，某些直觉只能通过试错、微调代码和观察结果来发展。理想情况下，优雅的数学理论可能会告诉我们如何调整我们的代码以达到预期的结果。不幸的是，目前我们还没有找到这样优雅的理论。
尽管我们尽了最大的努力，但对各种技术的正式解释仍然缺乏，一方面是因为描述这些模型的数学非常困难，另一方面是因为对这些主题的严肃研究最近才进入高潮。
我们希望，随着深度学习理论的进展，这本书的未来版本将能够提供深刻的见解，在目前的版本不能。

有时，为了避免不必要的重复，我们将经常导入和引用的函数、类等封装在本书 `d2l` 包中。对于要保存在软件包中的任何块，如函数、类或多个导入，我们将标记为 `# @save `。我们在 :numref:`sec_d2l` 中提供了这些函数和类的详细概述。`d2l` 软件包重量轻，只需要以下软件包和模块作为依赖项：

```{.python .input  n=1}
#@tab all
#@save
import collections
from collections import defaultdict
from IPython import display
import math
from matplotlib import pyplot as plt
import os
import pandas as pd
import random
import re
import shutil
import sys
import tarfile
import time
import requests
import zipfile
import hashlib
d2l = sys.modules[__name__]
```

:begin_tab:`mxnet`
本书中的大部分代码都基于 Apache MxNet。MxNet 是深度学习的开源框架，是 AWS（Amazon Web Services）以及许多学校和公司的首选选择。本书中的所有代码都通过了最新 MxNet 版本的测试。然而，由于深度学习的快速发展，一些代码在*未来版本*的MxNet中可能无法正常工作。
但是，我们计划保持在线版本最新。如果您遇到任何此类问题，请查看 :ref:`chap_installation` 以更新您的代码和运行环境。

以下是我们如何从 MxNet 导入模块。
:end_tab:

:begin_tab:`pytorch`
这本书中的大部分代码都基于 PyTorch。PyTorch 是深度学习的开源框架，在研究界非常流行。本书中的所有代码都通过了最新的 PyTorch 下的测试。然而，由于深度学习的快速发展，一些代码
在*未来版本*的 PyTorch 中可能无法正常工作。
但是，我们计划保持在线版本最新。如果您遇到任何此类问题，请查看 :ref:`chap_installation` 以更新您的代码和运行环境。

以下是我们如何从 PyTorch 导入模块。
:end_tab:

:begin_tab:`tensorflow`
本书中的大部分代码都基于 TensorFlow。TensorFlow 是深度学习的开源框架，在研究界和工业界都非常受欢迎。本书中的所有代码都在最新的 TensorFlow 下通过了测试。然而，由于深度学习的快速发展，一些代码
在*未来版本*的 TensorFlow 中可能无法正常工作。
但是，我们计划保持在线版本最新。如果您遇到任何此类问题，请查看 :ref:`chap_installation` 以更新您的代码和运行时环境。

以下是我们如何从 TensorFlow 导入模块。
:end_tab:

```{.python .input  n=1}
#@save
from mxnet import autograd, context, gluon, image, init, np, npx
from mxnet.gluon import nn, rnn
```

```{.python .input  n=1}
#@tab pytorch
#@save
import numpy as np
import torch
import torchvision
from torch import nn
from torch.nn import functional as F
from torch.utils import data
from torchvision import transforms
```

```{.python .input  n=1}
#@tab tensorflow
#@save
import numpy as np
import tensorflow as tf
```

### 目标受众

这本书适用于学生（本科或研究生）、工程师和研究人员，他们寻求深度学习的实用技巧的坚实掌握。因为我们从头开始解释每一个概念，所以不需要已有的深度学习或机器学习背景。充分解释深度学习的方法需要一些数学和编程，但我们只会假设你有一些基础知识，包括非常基本的线性代数，微积分，概率和 Python 编程。此外，在附录中，我们提供了关于本书所涵盖的大部分数学的进修。大多数情况下，我们将优先考虑直觉和想法，而不是数学的严谨性。有许多非常棒的书籍，可以导致感兴趣的读者进一步。例如，Bela Bollobas :cite:`Bollobas.1999` 的线性分析非常深入地涵盖了线性代数和函数分析。All of Statistics :cite:`Wasserman.2013` 是一个非常棒的统计数据指南。如果你之前没有使用过 Python，你可能想要细看这个 [Python tutorial](http://learnpython.org/)。

### 论坛

我们推出了一个与本书相关的论坛，位于 [discuss.d2l.ai](https://discuss.d2l.ai/)。当您对书籍的任何部分有疑问时，您可以在每章结尾处找到关联的讨论页面链接。

## 表示感谢

我们感谢数百名英文和中文草稿的贡献者。他们帮助改进了内容，并提供了宝贵的反馈。具体而言，我们感谢这份英文草稿的每一位撰稿人，让每个人都更好。他们的 GitHub ID 或名称是 (在没有特定的顺序): alxnorden, avinashingit, bowen0701, brettkoonce, Chaitanya Prakash Bapat,
cryptonaut, Davide Fiocco, edgarroman, gkutiel, John Mitro, Liang Pu,
Rahul Agarwal, Mohamed Ali Jamaoui, Michael (Stu) Stewart, Mike Müller,
NRauschmayr, Prakhar Srivastav, sad-, sfermigier, Sheng Zha, sundeepteki,
topecongiro, tpdi, vermicelli, Vishaal Kapoor, Vishwesh Ravi Shrimali, YaYaB, Yuhong Chen,
Evgeniy Smirnov, lgov, Simon Corston-Oliver, Igor Dzreyev, Ha Nguyen, pmuens,
Andrei Lukovenko, senorcinco, vfdev-5, dsweet, Mohammad Mahdi Rahimi, Abhishek Gupta,
uwsd, DomKM, Lisa Oakley, Bowen Li, Aarush Ahuja, Prasanth Buddareddygari, brianhendee,
mani2106, mtn, lkevinzc, caojilin, Lakshya, Fiete Lüer, Surbhi Vijayvargeeya,
Muhyun Kim, dennismalmgren, adursun, Anirudh Dagar, liqingnz, Pedro Larroy,
lgov, ati-ozgur, Jun Wu, Matthias Blume, Lin Yuan, geogunow, Josh Gardner,
Maximilian Böther, Rakib Islam, Leonard Lausen, Abhinav Upadhyay, rongruosong,
Steve Sedlmeyer, Ruslan Baratov, Rafael Schlatter, liusy182, Giannis Pappas,
ati-ozgur, qbaza, dchoi77, Adam Gerson, Phuc Le, Mark Atwood, christabella, vn09,
Haibin Lin, jjangga0214, RichyChen, noelo, hansent, Giel Dops, dvincent1337, WhiteD3vil,
Peter Kulits, codypenta, joseppinilla, ahmaurya, karolszk, heytitle, Peter Goetz, rigtorp,
tiepvupsu, sfilip, mlxd, Kale-ab Tessera, Sanjar Adilov, MatteoFerrara, hsneto,
Katarzyna Biesialska, Gregory Bruss, duythanhvn, paulaurel, graytowne, minhduc0711,
sl7423, Jaedong Hwang, Yida Wang, cys4, clhm, Jean Kaddour, austinmw, trebeljahr, tbaums,
cuongvng, pavelkomarov, vzlamal, NotAnotherSystem, J-Arun-Mani, jancio, eldarkurtic,
the-great-shazbot, doctorcolossus, gducharme, cclauss, Daniel-Mietchen, hoonose, biagiom,
abhinavsp0730, jonathanhrandall, ysraell, Nodar Okroshiashvili, UgurKap, Jiyang Kang,
StevenJokes, Tomer Kaftan, liweiwp, netyster, ypandya, NishantTharani, heiligerl.

我们感谢Amazon Web Services，尤其是Swami Sivasubramanian、Raju Gulabani、Charlie Bell,和Andrew Jassy在撰写这本书时给予的慷慨支持。如果没有可用的时间、资源、与同事的讨论以及不断的鼓励，这本书就不会问世。

## 总结

* 深度学习给模式识别带来了革命性的变化，它引入的技术如今为计算机视觉、自然语言处理、自动语音识别等多种技术提供了动力。
* 要成功地应用深度学习，你必须了解如何解决问题、数学建模、将模型与数据匹配的算法，以及实现所有这些的工程技术。
* 这本书提供了一个全面的资源，包括散文、数字、数学和代码，所有这些都在一个地方。
* 要回答与本书相关的问题，请访问我们的论坛 https://discuss.d2l.ai/。
* 所有notebooks都可以在 GitHub 上下载。

## 练习

1. 在这本书的论坛上注册一个帐户 [discuss.d2l.ai](https://discuss.d2l.ai/).
1. 在计算机上安装Python。
1. 点击论坛部分底部的链接，在那里您将能够寻求帮助，讨论这本书，并通过作者和更广泛的社区找到您的问题的答案。

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/18)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/20)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/186)
:end_tab:
