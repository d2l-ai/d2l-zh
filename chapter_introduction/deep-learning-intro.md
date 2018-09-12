# 深度学习简介

你可能已经接触过编程，并开发过一两款程序。同时你可能读到了关于深度学习或者是机器学习的铺天盖地的报道（很多时候它们被赋予了更广义的名字：人工智能），从而下定决心来了解它们。实际上，或者说幸运的是，大部分的程序并不需要深度学习或者是更广义上的人工智能技术。例如我们要为一只微波炉编写一个用户界面，只需要一点点功夫我们便能设计出十几个按钮以及一系列能精确描述微波炉在各种情况下的表现的规则。抑或是我们要编写一个电子邮件客户端，这样的程序比微波炉要更复杂一些，但我们还是可以沉下心来一步一步思考：客户端的用户界面将需要几个输入框用来接受收件人、主题、邮件正文等，程序将监听键盘输入并写入一个缓冲区，然后将它们显示在相应的输入框中。当用户点击“发送”按钮时，我们需要检查收件人邮箱地址的格式是否合法，检查邮件主题是否为空（或在主题为空时警告用户），而后用相应的协议传送邮件。

值得注意的是，在以上两个例子中，我们都不需要收集真实世界中的数据，也不需要系统地提取这些数据的特征。只要有充足的时间，我们的常识与编程技巧已经足够让我们完成它们。

与此同时，我们很容易就能找到一些即使世界上最好的程序员仅用编程技巧和常识无法解决的简单问题。例如，假设我们想要编写一个判定一张图像中有没有猫的程序。这件事听起来好像很简单，对不对？程序只需要对每张输入图像输出`True`（表示有猫）或者`False`（表示无猫）即可。但令人惊讶的是，即使是世界上最优秀的计算机科学家和程序员也不懂如何编写这样的程序。

我们从哪里入手呢？我们先进一步简化这个问题：若假设所有的图像的高和宽都是同样的400像素大小，一个像素由红绿蓝三个值构成，那么一张图像就由近50万个数值表示。那么哪些数值隐藏着我们必要的信息呢？是所有数值的平均数？还是四个角的数值？抑或是图像中的某一个特别的点？事实上，要想解读图像中的内容，你需要找寻仅仅在结合成千上万的数值时才会出现的特征，比如边缘、质地、形状、眼睛、鼻子等，最终找到图像中是否含有猫。

一种解决以上问题的思路是逆向思考。与其设计一个解决问题的程序，我们不如从最终的需求入手来寻找一个解决方案。事实上，这也是目前的机器学习和深度学习应用共同的核心思想，我们可以称其为“用数据编程”。与其枯坐在房间里思考怎么设计一个识别猫的程序，不如利用人类肉眼在图像中识别猫的能力。我们可以收集一些已知包含猫与不包含猫的真实图像，然后我们的目标就转化成如何从这些图像入手来得到一个可以推断出图像中是否含有猫的函数。这个函数的形式通常通过我们的知识来对针对特定问题来选定（例如我们使用一个二次函数来判断图像中是否含有猫），但函数里参数的具体值（二次函数系数的值）则是通过数据来确定。

机器学习是一门讨论各式各样适用于不同问题的函数形式，以及如何使用数据来有效地获取函数参数具体值的学科。深度学习是指机器学习中的一类函数，它们的形式为多层神经网络。近年来，仰赖于大数据集和强悍的硬件，深度学习已逐渐成为处理像图像、文本语料和声音信号等复杂高维度数据的主要方法。

我们现在正处于一个程序设计越来越多得到深度学习的帮助（甚至有时被深度学习取代）的时代，这可以说是计算机科学历史上的一个分水岭。深度学习在你的手机里：拼写校正、语音识别、还能认出社交媒体照片里的好友们。得益于优秀的算法、快速而廉价的算力、前所未有的大量数据、以及强大的软件工具，如今大多数软件工程师都有能力建立复杂的模型来解决十年前连最优秀的科学家们都觉得棘手的问题。

本书希望能帮助你进入深度学习的浪潮中。我们希望结合数学、代码和样例让深度学习变得触手可及。本书不要求你具有高深的数学和编程背景，我们将随着章节的前进逐一解释所需要的知识。更值得一提的是，本书的每一节都是一个可以独立运行的Jupyter记事本，你可以从网上获得这些记事本，并且可以在笔记本或云端服务器上执行它们。这样你可以随意改动书中的代码并得到及时的结果。我们希望本书能帮助和启发新一代的程序员、创业家、统计学家、生物学家、及所有对使用深度学习来解决问题感兴趣的人。

## 起源

虽然深度学习似乎是最近几年刚兴起的名词。但它的核心思想——用数据编程和神经网络——已经被研究了数百年。自古以来，人类就一直渴望能从数据中分析出预知未来的窍门。实际上，数据分析正是大部分自然科学的本质，我们希望从日常的观测中提取规则，并找寻不确定性。

早在17世纪[雅各比·伯努利 (1655-1705)](https://en.wikipedia.org/wiki/Jacob_Bernoulli) 提出了描述只有两种结果的随机过程（例如抛掷一枚硬币）的伯努利分布。大约一个世纪之后，[卡尔·弗里德里希·高斯 (1777-1855)](https://en.wikipedia.org/wiki/Carl_Friedrich_Gauss) 发明了今日仍广泛使用在从保险计算到医学诊断等领域的最小二乘法。概率论、统计学和模式识别等工具帮助自然科学的实验学家们从数据回归到自然定律，从而发现了如欧姆定律（描述电阻两端电压和流经电阻电流关系的定律）这类可以用线性模型完美表达的一系列自然法则。

即使是在中世纪，数学家们也热衷于利用统计学来做出估计。例如在[雅各比·科贝尔 (1460-1533)](https://www.maa.org/press/periodicals/convergence/mathematical-treasures-jacob-kobels-geometry)的几何书中记载了使用16名男子的平均脚长来估计男子的平均脚长。

![估计脚的长度](../img/koebel.jpg)

在这个研究中，16位成年男子被要求在离开教堂时站成一排并把脚贴在一起，而后他们脚的总长度被除以16来得到的一个估计（这个数字大约相当于今日的一英尺）。这个算法之后又被改进以应对特异形状的脚：最长和最短的脚不计入，只对剩余的脚长取平均值，即裁剪平均值的雏形。

现代统计学在20世纪的真正起飞要归功于数据的收集和发布。统计学巨匠之一[罗纳德·费雪 (1890-1962)](https://en.wikipedia.org/wiki/Ronald_Fisher)对于统计学理论和统计学在基因学中的应用功不可没。许多他发明的算法（例如线性判别分析）和公式（例如费希尔信息矩阵）仍经常被使用（即使是他在1936年发布的Iris数据集，仍然偶尔被用于展示机器学习算法）。

信息论 [(克劳德·香农, 1916-2001)](https://en.wikipedia.org/wiki/Claude_Shannon) 以及 [阿兰·图灵 (1912-1954)](https://en.wikipedia.org/wiki/Allan_Turing)的计算理论也对机器学习有深远影响。图灵在他著名的论文[计算机器与智能](https://www.jstor.org/stable/2251299) (Mind, October 1950) [1] 中提出了“机器可以思考吗？”这样一个问题。在他称为“图灵测试”的测验中，如果一个人类评审在对话的过程中不能区分他的对话对象到底是人类还是机器的话，那么即可认为这台机器是有智能的。时至今日，智能机器的发展日新月异。

另一个对深度学习有重大影响的领域是神经科学与心理学。既然人类显然能够展现出智能，那么对于解释并逆向工程人类智能机理的探究也是合情合理的。最早的算法之一是由[唐纳德·赫布 (1904-1985)](https://en.wikipedia.org/wiki/Donald_O._Hebb)正式提出的。

在他开创性的著作[《行为的组织》](http://s-f-walker.org.uk/pubsebooks/pdfs/The_Organization_of_Behavior-Donald_O._Hebb.pdf) (John Wiley & Sons, 1949) [2] 中，他提出神经是通过正向强化来学习的，即赫布理论。赫布理论是感知机学习算法的原型，并成为支撑今日深度学习的许许多多的随机梯度下降算法的基石：强化合意的行为、惩罚不合意的行为，最终获得优良的神经网络参数。

来源于生物学的灵感是神经网络名字的由来。研究者们（可以追溯到超过一世纪前Alexander Bain, 1873 和 James Sherrington, 1890的模型）尝试组建模仿神经元互动的计算电路。随着时间发展，神经网络的生物学解释被稀释，但仍保留了这个名字。时至今日，绝大多数网络都包含以下的核心原则：

* 交替使用线性与非线性处理单元，经常被称为“层”。
* 使用链式法则（即反向传播）来更新网络的参数

在最初的快速发展之后，自约1995年起至2005年，大部分的机器学习研究者的视线从*神经网络*上移开了，这是由于多种原因。首先，训练神经网络需要极多的算力。尽管上世纪末内存已经足够，算力却是不够充足。其次，当时使用的数据集也相对小得多。费雪在1932年发布的的Iris数据集仅有150个样本，并被广泛用于测试算法的性能。具有6万个样本的MNIST数据集在当时已经被认为是非常庞大了（如今MNIST已经被认为是典型的简单数据集）。由于数据和算力的稀缺，从经验上来说如核方法、决策树和概率图模型等统计工具更优，他们不像神经网络一样需要长时间的训练，并且在强大的理论保证下提供可以预测的结果。

## 深度学习

随着因为互联网的崛起、价廉物美的传感器和低价的存储器而得到的大量数据，以及便宜的计算力（尤其是原本为电脑游戏设计的GPU），上文描述的情况改变了许多。一瞬间，原本被认为不可能的算法和模型变得触手可及。这样的发展趋势从如下表格中可见一斑：

|                 | 1970               | 1980              | 1990                 | 2000             | 2010               | 2020               |
| --------------- | ------------------ | ----------------- | -------------------- | ---------------- | ------------------ | ------------------ |
| 数据 (样本数量) | 100 (Iris)         | 1 K（波士顿房价） | 10 K（手写字符识别） | 10 M（网页）     | 10 G（广告）       | 1 T （社交网络）   |
| 内存            | 1 KB               | 100 KB            | 10 MB                | 100 MB           | 1 GB               | 100 GB             |
| 每秒浮点计算数  | 100 K (Intel 8080) | 1M (Intel 80186)  | 10 M (Intel 80486)   | 1 G (Intel Core) | 1 T (Nvidia C2050) | 1 P (Nvidia DGX-2) |

很显然，存储容量没能跟上数据量增长的步伐。与此同时，计算力的增长又盖过了数据量的增长。这样的趋势使得统计模型可以在优化参数上投资更多的计算力，但同时需要提高利用存储的效率（通常使用非线性单元来达到这个目的）。这也相应导致了机器学习和统计学的最优选择从广义线性模型及核方法变化到了深度多层神经网络上。这样的变化正是诸如多层感知机(e.g. McCulloch & Pitts, 1943)、卷积神经网络 (Le Cun, 1992) 、长-短期记忆序列神经网络 (Hochreiter & Schmidhuber, 1997) 和Q-Learning (Watkins, 1989)等深度学习的支柱模型在过去十年从坐了数十年的冷板凳上站起来被“重新发现”的原因。

近年来在统计模型、应用和算法上的进展常被拿来与寒武纪大爆发（历史上物种数量大爆发的一个时期）做比较。但这些进展不仅仅是因为可用资源变多了让我们得以用新瓶装旧酒，下面的列表仅仅能涵盖近十年来深度学习长足发展的部分原因：

* 优秀的容量控制方法，例如丢弃层 [(Srivastava et al., 2014)](http://jmlr.org/papers/volume15/srivastava14a.old/srivastava14a.pdf) [3] 使得大型网络的训练不再受制于过拟合（大型神经网络学会记忆大部分训练数据的行为）。这是靠在整个网络中注入噪音（训练时随机将权重替换为随机的数字）[(Bishop, 1995)](https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/bishop-tikhonov-nc-95.pdf) [4] 来达到的。

* 注意力机制解决了另一个困扰统计学超过一个世纪的问题：如何在不增加参数的情况下扩展一个系统的记忆容量和复杂度。注意力机制[Bahdanau, Cho and Bengio, in 2014](https://arxiv.org/pdf/1409.0473.pdf)[5] 使用了一个可学习的指针结构来构建出一个精妙的解决方法。也就是说，与其在例如机器翻译这样的任务中记忆整个句子，不如记忆指向翻译的中间状态的指针。由于生成译文前不需要再存储整句原文的信息，这样的结构使准确翻译长句变得可能。

* 例如记忆网络[(Sukhbataar et al., 2015)](https://arxiv.org/pdf/1503.08895.pdf) [6] 和神经编码器-解释器[(Reed and de Freitas, 2016)](https://arxiv.org/abs/1511.06279) [7] 这样的多阶设计使得针对推理过程的迭代建模方法变得可能。这些模型允许重复修改深度网络的内部状态，这样就能模拟出推理链条上的各个步骤，就好像处理器在计算过程中修改内存一样。

* 另一个重大发展是对抗生成网络[(Goodfellow et al., 2014)](https://papers.nips.cc/paper/5423-generative-adversarial-nets.pdf) [8] 的发明。传统上，用在概率分布估计和生成模型上的统计方法更多地关注于找寻正确的概率分布，以及正确的采样算法。对抗生成网络的关键创新在于将采样部分替换成了任意的含有可微参数的算法，这些参数将被训练到使得辨别器不能再分辨真的和生成的样本。对抗生成网络可使用任意算法来生成输出的特性为许多技巧打开了新的大门。像是生成奔跑的斑马[(Zhu et al., 2017)](https://junyanz.github.io/CycleGAN/)[9]和生成名流的照片[(Karras et al., 2018)](http://research.nvidia.com/publication/2017-10_Progressive-Growing-of) [10] 都是对抗生成网络发展的见证。

* 许多情况下单个GPU已经不能满足在大型数据集上训练的需要。过去十年内我们构建分布式并行训练算法的能力已经有了极大的提升。设计可扩展算法的最大瓶颈在于深度学习优化算法的核心——随机梯度下降需要相对更小的批量。与此同时，更小的批量也会降低GPU的效率。如果我们使用1024个GPU，每个GPU的批量大小为32个样例，那么单步训练的批量大小将是32000个以上。近年来[Li, 2016](https://www.cs.cmu.edu/~muli/file/mu-thesis.pdf) [11]、 [You et al, 2017](https://arxiv.org/pdf/1708.03888.pdf)[12] 及 [Jia et al, 2018](https://arxiv.org/pdf/1807.11205.pdf)[13]的工作将批量大小推向多达64000个样例，并将在ImageNet数据集上训练ResNet50的时间降到了7分钟。与之对比，最初的训练时间需要以天来计算。

* 并行计算的能力也为强化学习（至少在可以采用模拟的情况下）的发展贡献了力量。并行计算帮助计算机在围棋、雅达利游戏、星际争霸和物理模拟上达到了超过人类的水准。

* 深度学习框架也在传播深度学习思想的过程中扮演了重要角色。例如[Caffe](https://github.com/BVLC/caffe), [Torch](https://github.com/torch) 和 [Theano](https://github.com/Theano/Theano)这样的第一代框架使得建模变得简单，许多开创性的论文都用到了这些框架。如今它们已经被[TensorFlow](https://github.com/tensorflow/tensorflow) (经常是以高层API [Keras](https://github.com/keras-team/keras)的形式被使用）、[CNTK](https://github.com/Microsoft/CNTK)、 [Caffe 2](https://github.com/caffe2/caffe2) 和 [Apache MxNet](https://github.com/apache/incubator-mxnet)所取代。第三代，即命令式深度学习框架，是由用类似Numpy的语法来定义模型的 [Chainer](https://github.com/chainer/chainer) 开创。这样的思想后来被 [PyTorch](https://github.com/pytorch/pytorch) 和 MXNet的 [Gluon API](https://github.com/apache/incubator-mxnet) 采用，后者也正是本书用来教学深度学习的工具。

系统研究者负责构建更好的工具，统计学家建立更好的模型，这样的分工使得工作大大简化。举例来说，在2014年时，训练一个逻辑回归模型曾是卡耐基梅隆大学布置给机器学习方向的新入学博士生的作业问题。时至今日，这个问题只需要少于10行的代码便可以完成，普通的程序员都可以做到。

## 成功案例

长期以来机器学习都能达成其他方法难以达成的目的，例如，自上世纪90年代起，邮件的分拣就开始使用光学字符识别（实际上这正是知名的MNIST和USPS手写数字数据集的来源）。机器学习也是电子支付系统的支柱，可以用于读取银行支票、进行授信评分以及防止金融欺诈。机器学习算法在网络上被用来提供搜索结果、个性化推荐和网页排序。尽管长期处于公众视野之外，机器学习已经渗透到了方方面面。直到近年来在此前认为无法被解决的问题以及直接关系到消费者的问题上的突破性进展，机器学习才逐渐变成公众的焦点。这些进展基本归功于深度学习：

* 诸如苹果公司的Siri、亚马逊的Alexa和谷歌助手一类的智能助手能以可观的准确率回答口头提出的问题，甚至包括从简单的开关灯具（对残疾群体帮助很大）到提供语音对话帮助。智能助手的出现或许可以作为人工智能开始影响我们生活的标志。

* 智能助手的关键是需要能够精确识别语音，而这类系统在某些应用上的精确度已经渐渐增长到可以与人类并肩[(Xiong et al., 2017)](https://arxiv.org/abs/1708.06073) [14]。

* 物体识别也经历了漫长的发展过程。在2010年从图像中识别出物体的类别仍是一个相当有挑战性的任务。当年在ImageNet基准测试上 [Lin et al., 2010](http://www.image-net.org/challenges/LSVRC/2010/ILSVRC2010_NEC-UIUC.pdf) [15] 取得了28%的top-5错误率。到2017年 [Hu et al.](https://arxiv.org/abs/1709.01507) [16] 将这个数字降低到了2.25%。在鸟类辨别和皮肤癌诊断上，也取得了同样惊世骇俗的成绩。

* 游戏曾被认为是人类智能最后的堡垒。自使用时间差分强化学习玩双陆棋的TDGammon之始，算法和算力的发展催生了一系列在游戏上使用的新算法。与双陆棋不同，国际象棋有更复杂的状态空间和更多的可选动作。“深蓝”用大量的并行、专用硬件和游戏树的高效搜索打败了加里·卡斯帕罗夫 [(Campbell et al., 2002)](https://dl.acm.org/citation.cfm?id=512152) [17]。围棋因其庞大的状态空间被认为是更难的游戏，AlphaGo在2015年[Silver et al.](https://www.nature.com/articles/nature16961) [18] 用结合深度学习与蒙特卡洛树采样的方法达到了人类水准。对德州扑克游戏而言，除了巨大的状态空间之外，更大的挑战是游戏的信息并不完全可见（看不到对手的牌）。“冷扑大师”[(Brown and Sandholm, 2017)](https://www.cs.cmu.edu/~noamb/papers/17-IJCAI-Libratus.pdf) [19] 用高效的策略体系超越了人类玩家的表现。以上的例子都体现出了先进的算法是人工智能在游戏上的表现提升的重要原因。

* 机器学习进步的另一个标志是自动驾驶汽车的发展。尽管距离完全的自主驾驶还有很长的路要走，但诸如 [Momenta](http://www.momenta.com)、 [Tesla](http://www.tesla.com)、 [NVIDIA](http://www.nvidia.com)、 [MobilEye](http://www.mobileye.com) 和 [Waymo](http://www.waymo.com) 这样的公司交出的具有部分自主驾驶功能的产品展示出了这个领域巨大的进步。完全自主驾驶的难点在于它需要将感知、思考和规则整合在同一个系统中。目前，深度学习主要被应用在计算机视觉的部分，剩余的部分还是需要工程师们的大量调试。

以上的列出的仅仅是近年来深度学习所取得的成果的冰山一角。机器人学、物流管理、计算生物学、粒子物理学和天文学近年来的发展也有部分要归功于深度学习。可以看到，深度学习已经逐渐演变成一个工程师和科学家皆可使用的普适工具。

## 关键组成

因为算法和应用都多到让人眼花缭乱，所以很难说清深度学习的核心组成部分应该是什么，正像是我们也很难说清披萨饼的组成部分应该是什么——每一个部分都是可以被替换的。例如，也许有人认为多层感知机是一个必要的组成部分，但实际上也有完全由卷积层组成的计算机视觉模型。

所有方法最主要的共同点应该说是端到端的训练。也就是说，并不是将单独调试的部分拼凑起来组成一个系统，而是将整个系统组建好之后一起训练。比如说，计算机视觉科学家们之前曾一度将特征构造与机器学习模型的构建分开处理，像是 Canny 边缘探测 [(Canny, 1986)](https://ieeexplore.ieee.org/document/4767851/) [20] 和 SIFT 特征提取 [(Lowe, 2004)](https://www.cs.ubc.ca/~lowe/papers/ijcv04.pdf) [21] 曾占据统治性地位达10年以上，但这也就是人类能找到的最好方法了。当深度学习进入这个领域，这些特征提取方法就被性能更强的自动优化的滤波器替代了。

相似地，在自然语言处理领域，词袋模型[(Salton and McGill, 1993)](https://dl.acm.org/citation.cfm?id=576628) [22]多年来都被认为是不二之选。词袋模型是将一个句子映射到一个词频向量的模型，但这样的做法完全忽视了单词的排列顺序或者是句中的标点符号。不幸的是，我们也没有能力来手工构建更好的特征。但是自动化的算法反而可以从所有可能的特征设计中搜寻最好的那个，这也带来了极大的进步。例如，语义相关的词嵌入能够在向量空间中完成如下推理：“Berlin - Germany + Italy = Rome”。可以看出，这些都是端到端训练整个系统带来的效果。

除了端到端的训练以外，我们也正在经历从含参数统计描述向完全无参数的模型。当数据非常稀缺时，我们需要通过简化对现实的假设来得到实用的模型。当数据充足时，我们就可以用能更好地拟合现实的无参数模型来替代这些含参数模型。这也使得我们可以得到更精确的模型，尽管需要牺牲一些可解释性。

另一个与此前工作的区别是对于非最优解的包容、非凸非线性优化的使用以及勇于尝试没有被证明过的方法。这种在处理统计问题上的新经验主义风潮与大量人才的涌入，带来了在实际问题上的高速进展（尽管大部分情况下需要修改甚至重新发明已经存在数十年的工具）。

最后，深度学习社区长期以来以在学界和企业之间分享工具而自豪，开源了许多优秀的软件库、统计模型和预训练网络。正是本着开放开源的精神，本书和基于它的教学视频可以自由下载和随意分享。我们致力于为所有人降低学习深度学习的门槛，并希望大家从中获益。

## 练习

* 你现在正在编写的代码有没有可以被“学习”的部分？也就是说，是否有可以被机器学习改进的部分？
* 你在生活中有没有遇到过有许多样例，但无法找到一个特定的自动解决的算法的问题？它们也许是深度学习的最好猎物。
* 如果把人工智能的发展看作是新一次工业革命。那么深度学习和数据的关系是否像是蒸汽机与煤炭的关系呢？为什么？
* 端到端的训练方法还可以用在哪里？物理学？工程学？或是经济学？
* 为什么我们应该让深度网络模仿人脑结构？为什么我们不该让深度网络模仿人脑结构？

## 扫码直达 [讨论区](https://discuss.gluon.ai/t/topic/746)

![](../img/qr_deep-learning-intro.svg)

## 参考文献

[1] Machinery, C. (1950). Computing machinery and intelligence-AM Turing. Mind, 59(236), 433.

[2] Hebb, D. O. (1949). The organization of behavior; a neuropsycholocigal theory. A Wiley Book in Clinical Psychology., 62-78.

[3] Srivastava, N., Hinton, G., Krizhevsky, A., Sutskever, I., & Salakhutdinov, R. (2014). Dropout: a simple way to prevent neural networks from overfitting. The Journal of Machine Learning Research, 15(1), 1929-1958.

[4] Bishop, C. M. (1995). Training with noise is equivalent to Tikhonov regularization. Neural computation, 7(1), 108-116.

[5] Bahdanau, D., Cho, K., & Bengio, Y. (2014). Neural machine translation by jointly learning to align and translate. arXiv preprint arXiv:1409.0473.

[6] Sukhbaatar, S., Weston, J., & Fergus, R. (2015). End-to-end memory networks. In Advances in neural information processing systems (pp. 2440-2448).

[7] Reed, S., & De Freitas, N. (2015). Neural programmer-interpreters. arXiv preprint arXiv:1511.06279.

[8] Goodfellow, I., Pouget-Abadie, J., Mirza, M., Xu, B., Warde-Farley, D., Ozair, S., … & Bengio, Y. (2014). Generative adversarial nets. In Advances in neural information processing systems (pp. 2672-2680).

[9] Zhu, J. Y., Park, T., Isola, P., & Efros, A. A. (2017). Unpaired image-to-image translation using cycle-consistent adversarial networks. arXiv preprint.

[10] Karras, T., Aila, T., Laine, S., & Lehtinen, J. (2017). Progressive growing of gans for improved quality, stability, and variation. arXiv preprint arXiv:1710.10196.

[11] Li, M. (2017). Scaling Distributed Machine Learning with System and Algorithm Co-design (Doctoral dissertation, PhD thesis, Intel).

[12] You, Y., Gitman, I., & Ginsburg, B. Large batch training of convolutional networks. ArXiv e-prints.

[13] Jia, X., Song, S., He, W., Wang, Y., Rong, H., Zhou, F., … & Chen, T. (2018). Highly Scalable Deep Learning Training System with Mixed-Precision: Training ImageNet in Four Minutes. arXiv preprint arXiv:1807.11205.

[14] Xiong, W., Droppo, J., Huang, X., Seide, F., Seltzer, M., Stolcke, A., … & Zweig, G. (2017, March). The Microsoft 2016 conversational speech recognition system. In Acoustics, Speech and Signal Processing (ICASSP), 2017 IEEE International Conference on (pp. 5255-5259). IEEE.

[15] Lin, Y., Lv, F., Zhu, S., Yang, M., Cour, T., Yu, K., … & Huang, T. (2010). Imagenet classification: fast descriptor coding and large-scale svm training. Large scale visual recognition challenge.

[16] Hu, J., Shen, L., & Sun, G. (2017). Squeeze-and-excitation networks. arXiv preprint arXiv:1709.01507, 7.

[17] Campbell, M., Hoane Jr, A. J., & Hsu, F. H. (2002). Deep blue. Artificial intelligence, 134(1-2), 57-83.

[18] Silver, D., Huang, A., Maddison, C. J., Guez, A., Sifre, L., Van Den Driessche, G., … & Dieleman, S. (2016). Mastering the game of Go with deep neural networks and tree search. nature, 529(7587), 484.

[19] Brown, N., & Sandholm, T. (2017, August). Libratus: The superhuman ai for no-limit poker. In Proceedings of the Twenty-Sixth International Joint Conference on Artificial Intelligence.

[20] Canny, J. (1986). A computational approach to edge detection. IEEE Transactions on pattern analysis and machine intelligence, (6), 679-698.

[21] Lowe, D. G. (2004). Distinctive image features from scale-invariant keypoints. International journal of computer vision, 60(2), 91-110.

[22] Salton, G., & McGill, M. J. (1986). Introduction to modern information retrieval.
