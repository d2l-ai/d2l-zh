# 机器学习简介

本书作者跟广大程序员一样，在开始写作前需要去来一杯。我们跳进准备出发，Alex掏出他的安卓喊一身“OK Google”唤醒语言助手，Mu操着他的中式英语命令到“去蓝瓶咖啡店”。手机这时马上显示出识别的命令，并且知道我们需要导航。接着它调出地图应用并给出数条路线方案，每条方案边上会有预估的到达时间并自动选择最快的线路。

好吧，这是一个虚构的例子，因为我们一般在办公室喝自己的手磨咖啡。但这个例子展示了在短短几秒钟里，我们跟数个机器学习模型进行了交互。

如果你从来没有使用过机器学习，你会想，这个不就是编程吗？或者，到底机器学习是什么？首先，我们确实是使用编程语言来实现机器学习模型，我们跟计算机其他领域一样，使用同样的编程语言和硬件。但不是每个程序都用了机器学习。对于第二个问题，精确定义机器学习就像定义什么是数学一样难，但我们试图在这章提供一些直观的解释。


## 一个例子

我们日常交互的大部分计算机程序可以使用最基本的命题来实现。当你把一个商品加进购物车时，你触发了电商的电子商务程序来吧一个商品ID和你的用户ID插入到一个叫做购物车的数据库表格中。你可以在没有见到任何真正客户前来用最基本的程序指令来实现这个功能。如果你发现你可以这么做，那么你就不应该使用机器学习。

对来机器学习科学家来说，幸运的是大部分应用没有那么容易。回到前面那个例子，想象下如何写一个程序来回应唤醒词例如“Okay, Google”，“Siri”，和“Alexa”。如果你在一个只有你和代码编辑器的房间里写这个程序，你改怎么办？你可能会想像下面的程序

```python
# if input_command == 'Okey, Google":
#     run_voice_assistant()
```

但实际上你能拿到的只是麦克风里采集到的原始语音信号，可能是每秒44,000个样本点。那么需要些什么样的规则才能把这些样本点转成一个字符串呢？或者简单点，判断这些信号里是不是就是说了唤醒词。如果你被这个困住了，不用担心，我们也不会从零开始写。这就是我们为什么要机器学习。

虽然我们不知道怎么告诉机器去吧语音信号转成对应的字符串，但我们自己可以。我们可以收集一个巨大的**数据集**里包含了大量语音信号，以及每个语音型号是不是对应我们要的唤醒词。在机器学习里，我们不直接设计一个系统去辨别唤醒词，而是写一个灵活的程序，它的行为可以根据在读取数据集的时候改变。所以我们不是去直接写一个唤醒词辨别器，而是一个程序，当提供一个巨大的有标注的数据集的时候它能辨别唤醒词。你可以认为这种方式是**利用数据编程**。

## 眼花缭乱的机器学习应用

机器学习背后的核心思想是，设计程序使得它可以在执行的时候提升它在某任务上的能力，而不是有着固定行为的程序。机器学习包括多种问题的定义，提供很多不同的算法，能解决不同领域的各种问题。我们之前讲到的是一个讲**监督学习**应用到语言识别的例子。

正因为机器学习提供多种工具可以利用数据来解决简单规则不能或者难以解决的问题，它被广泛的用在了搜索引擎，无人驾驶，机器翻译，医疗诊断，垃圾邮件过滤，玩游戏，人脸识别，数据匹配，信用评级，和给图片加滤镜的各种应用中。

虽然这些问题各式各样，但他们有着共同的模式从而可以被机器学习模型解决。最常见的描述这些问题的方法是通过数学，但不像其他机器学习和神经网络的书那样，我们会主要关注真实数据和代码。下面我们来看点数据和代码。

## 用代码编程和用数据编程

这个例子灵感来自 [Joel Grus](http://joelgrus.com) 的一次 [应聘面试](http://joelgrus.com/2016/05/23/fizz-buzz-in-tensorflow/). 面试官让他写个程序来玩Fizz Buzz. 这是一个小孩子游戏。玩家从1数到100，如果数字被3整除，那么喊'fizz'，如果被5整除就喊'buzz'，如果两个都满足就喊'fizzbuzz'，不然就直接说数字。这个游戏玩起来就像是：

> 1 2 fizz 4 buzz fizz 7 8 fizz buzz 11 fizz 13 14 fizzbuzz 16 ...

传统的实现是这样的：

```{.python .input}
res = []
for i in range(1, 101):
    if i % 15 == 0:
        res.append('fizzbuzz')
    elif i % 3 == 0:
        res.append('fizz')
    elif i % 5 == 0:
        res.append('buzz')
    else:
        res.append(str(i))
print(' '.join(res))
```

对于经验丰富的程序员来说这个太不够一颗赛艇了。所以Joel尝试用机器学习来实现这个。为了让程序能学，他需要准备下面这个数据集：

* 数据 X ``[1, 2, 3, 4, ...]`` 和标注Y ``['fizz', 'buzz', 'fizzbuzz', identity]``
* Training data, i.e. examples of what the system is supposed to do. Such as ``[(2, 2), (6, fizz), (15, fizzbuzz), (23, 23), (40, buzz)]``
* ###### Features that map the data into something that the computer can handle more easily, e.g. ``x -> [(x % 3), (x % 5), (x % 15)]``. This is optional but helps a lot if you have it.

有了这些，Jeol利用TensorFlow写了一个[分类器](https://github.com/joelgrus/fizz-buzz-tensorflow)。对于不按常理出牌的Jeol，面试官一脸黑线。而且这个分类器不是总是的对的。

显然，用原子弹杀鸡了。为什么不直接写几行简单而且保证结果正确的Python代码呢？当然，这里有很多例子一个简单Python脚本不能分类的，即使简单的3岁小孩解决起来毫无压力。

| ![](img/cat1.jpg) | ![](img/cat2.jpg) | ![](img/dog1.jpg) | ![](img/dog2.jpg) |
| :---------------: | :---------------: | :---------------: | :---------------: |
|         喵         |         喵         |         汪         |         喵         |

幸运的是，这个正是机器学习的用武之地。我们通过提供大量的含有猫和狗的图片来编程一个猫狗检测器，一般来说它就是一个函数，它会输出一个大的正数如果图片里面是猫，一个大的负数如实过狗，如果不确信就输出一个0附近的。当然，这是机器学习能做的最简单例子。

## 机器学习最简要素

成功的机器学习有四个要素：数据，转换数据的模型，衡量模型好坏的损失函数，和一个调整模型权重来最小化损失函数的算法。

- **数据**。越多越好。特别的，数据是深度学习复兴的核心，因为复杂的非线性模型比其他机器学习需要更多的数据。数据的例子包括
  - 图片：例如你的手机图片，里面可能包含猫，狗，恐龙，高中同学聚会，或者昨天的晚饭。
  - 文本：邮件，新闻，微信聊天记录
  - 声音：有声书籍，电话记录
  - 结构数据：Jupyter notebook（里面有文本，图片和代码），网页，租车单，电费表
- **模型**。通常数据和我们最终想要的相差很远，例如我们想知道照片中的人是不是在高兴，所以我们需要把一千万像素变成一个高兴度的概率值。通常我们需要在数据上应用数个（通常）非线性函数（例如神经网络）
- **损失函数**。我们需要对比模型的输出和真实值之间的误差。损失函数帮助我们决定2017年底亚马逊股票会不会价值1500美元。取决于我们想短线还是长线，这个函数可以很不一样。
- **训练**。通常一个模型里面有很多参数。我们通过最小化损失函数来学这些参数。不幸的是，即使我们做得很好也不能保证在新的没见过的数据上我们可以任然做很好。
  - **训练误差**。这是模型在评估用来训练模型的数据集上的误差。这个类似于考试前我们在模拟试卷上拿到的分数。有一定的指向性，但不一定保证真实考试分数。
  - **测试误差**。这是模型在没见过的新数据上的误差，可能会跟训练误差不很一样（统计上叫过拟合）。这个类似于考前模考次次拿高分，但实际考起来确失误了。（笔者之一曾今做GRE真题时次次拿高分，高兴之下背了一遍红宝书就真上阵考试了，结果最终拿了一个刚刚够用的低分。后来意识到这是因为红宝书里包含了大量的真题。）

下面我们详细讨论一些不同机器学习应用。

In the following we will discuss a few types of machine learning in some more detail. This helps to understand what exactly one aims to do. We begin with a list of *objectives*, i.e. a list of things that machine learning can do. Note that the objectives are complemented with a set of techniques of *how* to accomplish them, i.e. training, types of data, etc. The list below is really only sufficient to whet the readers' appetite and to give us a common language when we talk about problems. We will introduce a larger number of such problems as we go along.

## Classification

This is one of the simplest tasks. Given data $x \in X$, such as images, text, sound, video, medical diagnostics, performance of a car, motion sensor data, etc., we want to answer the question as to which class $y \in Y$ the data belongs to. In the above case, $X$ are images and $Y = \mathrm{\{cat, dog\}}$. Quite often the confidence of the classifier, i.e. the algorithm that does this, is expressed in the form of probabilities, e.g. $\Pr(y=\mathrm{cat}\mid x) = 0.9$, i.e. the classifier is 90% sure that it's a cat. Whenever we have only two possible outcomes, statisticians call this a *binary classifier*. All other cases are called *multiclass classification*, e.g. the digits `[0, 1, 2, 3 ... 9]` in a digit recognition task. In `MXNet Gluon` the corresponding loss function is the [Cross Entropy Loss](http://mxnet.io/api/python/gluon.html#mxnet.gluon.loss.SoftmaxCrossEntropyLoss).

Note that the most likely class is not necessarily the one that you're going to use for your decision. Assume that you find this beautiful mushroom in your backyard:

| ![](img/death_cap.jpg)  |
| :---------------------: |
| Death cap - do not eat! |

Our (admittedly quite foolish) classifier outputs $\Pr(y=\mathrm{death cap}\mid\mathrm{image}) = 0.2$. In other words, it is quite confident that it *isn't* a death cap. Nonetheless, very few people would be foolhardy enough to eat it, after all, the certain benefit of a delicious dinner isn't worth the potential risk of dying from it. In other words, the effect of the *uncertain risk* by far outweighs the benefit. Let's look at this in math. Basically we need to compute the expected risk that we incur, i.e. we need to multiply the probability of the outcome with the benefit (or harm) associated with it:

$$L(\mathrm{action}\mid x) = \mathbf{E}_{y \sim p(y\mid x)}[\mathrm{loss}(\mathrm{action},y)]$$

Hence, the loss $L$ incurred by eating the mushroom is $L(a=\mathrm{eat}\mid x) = 0.2 * \infty + 0.8 * 0 = \infty$, whereas the cost of discarding it is $L(a=\mathrm{discard}\mid x) = 0.2 * 0 + 0.8 * 1 = 0.8$. We got lucky - as any botanist would tell us, the above actually *is* a death cap.

There are way more fancy classification problems than the ones above. For instance, we might have hierarchies. One of the first examples of such a thing are due to Linnaeus, who applied it to animals. Usually this is referred to as *hierarchical classification*. Typically the cost of misclassification depends on how far you've strayed from the truth, e.g. mistaking a poodle for a schnautzer is no big deal but mistaking it for a dinosaur would be embarrassing. On the other hand, mistaking a rattle snake for a garter snake could be deadly. In other words, the cost might be *nonuniform* over the hierarchy of classes but tends to increase the further away you are from the truth.

![](img/taxonomy.jpg)

## Tagging

It is worth noting that many problems are *not* classification problems. Discerning cats and dogs by computer vision is relatively easy, but what should our poor classifier do in this situation?

![](img/catdog.jpg)

Obviously there's a cat in the picture. And a dog. And a tire, some grass, a door, concrete, rust, individual grass leaves, etc.; Treating it as a binary classification problem is asking for trouble. Our poor classifier will get horribly confused if it needs to decide whether the image is one of two things, if it is actually both.

The above example seems contrived but what about this case: a picture of a model posing in front of a car at the beach. Each of the tags `(woman, car, beach)` would be true. In other words, there are situations where we have multiple tags or attributes of what is contained in an object. Sometimes this is treated as a lot of binary classification problems. But this is problematic, too, since there are just so many tags (often hundreds of thousands or millions) that could apply, e.g. `(ham, green eggs, spam, grinch, ...)` and we would have to *check* all of them and to ensure that they are all accurate.

Suffice it to say, there are better ways of generating tags. For instance, we could try to estimate the probability that $y$ is one of the tags in the set $S_x$ of tags associated with $x$, i.e. $\Pr(y \in S_x\mid x)$. We will discuss them later in this tutorial (with actual code). For now just remember that *tagging is not classification*.

## Regression

Let's assume that you're having your drains repaired and the contractor spends $x=3$ hours removing gunk from your sewage pipes. He then sends you a bill of $y = 350\$ $. Your friend hires the same contractor for $x = 2$ hours and he gets a bill of $y = 250\$ $. You can now both team up and perform a regression estimate to identify the contractor's pricing structure: \$100 per hour plus \$50 to show up at your house. That is, $f(x) = 100 \cdot x + 50$.

More generally, in regression we aim to obtain a real-valued number $y \in \mathbb{R}$ based on data $x$. Here $x$ could be as simple as the number of hours worked, or as complex as last week's news if we want to estimate the gain in a share price. For most of the tutorial, we will be using one of two very common losses, the [L1 loss](http://mxnet.io/api/python/gluon.html#mxnet.gluon.loss.L1Loss) where $l(y,y') = \sum_i |y_i-y_i'|$ and the [L2 loss](http://mxnet.io/api/python/gluon.html#mxnet.gluon.loss.L2Loss) where $l(y,y') = \sum_i (y_i - y_i')^2$. As we will see later, the $L_2$ loss corresponds to the assumption that our data was corrupted by Gaussian Noise, whereas the $L_1$ loss is very robust to malicious data corruption, albeit at the expense of lower efficiency.

## Search and ranking

One of the problems quite different from classifications is ranking. There the goal is less to determine whether a particular page is relevant for a query, but rather, which one of the plethora of search results should be displayed for the user. That is, we really care about the ordering among the relevant search results and our learning algorithm needs to produce ordered subsets of elements from a larger set. In other words, if we are asked to produce the first 5 letters from the alphabet, there is a difference between returning ``A B C D E`` and ``C A B E D``. Even if the result set is the same, the ordering within the set matters nonetheless.

A possible solution to this problem is to score every element in the set of possible sets with a relevance score and then retrieve the top-rated elements. [PageRank](https://en.wikipedia.org/wiki/PageRank) is an early example of such a relevance score. One of the peculiarities is that it didn't depend on the actual query. Instead, it simply helped to order the results that contained the query terms. Nowadays search engines use machine learning and behavioral models to obtain query-dependent relevance scores. There are entire conferences devoted to this subject.

## Recommender systems

Quite closely related to search and ranking are recommender systems. The problems are  similar insofar as the goal is to display a set of relevant items to the user. The main difference is the emphasis on *personalization* to specific users in the context of recommender systems. For instance, for movie recommendation the result page for a SciFi fan and the result page for a connoisseur of Woody Allen comedies might differ significantly.

Such problems occur, e.g. for movie, product or music recommendation. In some cases customers will provide explicit details about how much they liked the product (e.g. Amazon product reviews). In some other cases they might simply provide feedback if they are dissatisfied with the result (skipping titles on a playlist). Generally, such systems strive to estimate some score $y_{ij}$ as a function of user $u_i$ and object $o_j$. The objects $o_j$ with the largest scores $y_{ij}$ are then used as recommendation. Production systems are considerably more advanced and take detailed user activity and item characteristics into account when computing such scores. Below an example of the books recommended for deep learning, based on the author's preferences.

![](img/deeplearning_amazon.png)


## Sequence transformations

Some of the more exciting applications of machine learning are sequence transformations, sometimes also referred as ``seq2seq`` problems. They ingest a sequence of data and emit a new, significantly transformed one. This goes considerably beyond the previous examples where the output essentially had a predermined cardinality and type (e.g. one out of 10 classes, regressing a dollar value, ordering objects). While it is impossible to consider all types of sequence transformations, a number of special cases are worth mentioning:

### Tagging and Parsing

This involves annotating a text sequence with attributes. In other words, the number of inputs and outputs is essentially the same. For instance, we might want to know where the verbs and subjects are, we might want to know which words are the named entities. In general, the goal is to decompose and annotate text $x$ based on structural and grammatical assumptions to get some annotation $y$. This sounds more complex than it actually is. Below is a very simple example of annotating a sentence with tags regarding which word refers to a named entity.

| `Tom wants to have dinner in Washington with Sally.` |
| :--------------------------------------- |
| `E   -     -  -    -      -  E          -    E` |

### Automatic Speech Recognition

Here the input sequence $x$ is the sound of a speaker, and the output $y$ is the textual transcript of what the speaker said. The challenge is that there are many more audio frames (sound is typically sampled at 8kHz or 16kHz), i.e. there is no 1:1 correspondence between audio and text. In other words, this is a seq2seq problem where the output is much shorter than the input.

| `----D----e--e--e-----p----------- L----ea-------r---------ni-----ng-----` |
| :--------------------------------------- |
| ![Deep Learning](img/speech.jpg)         |

### Text to Speech

TTS is the inverse of Speech Recognition. That is, the input $x$ is text and the output $y$ is an audio file. There, the output is *much longer* than the input. While it is easy for *humans* to recognize a bad audio file, this isn't quite so trivial for computers. The challenge is that the audio output is way longer than the input sequence.

### Machine Translation

The goal here is to map text from one language automatically to the other. Unlike in the previous cases where the order of the inputs was preserved, in machine translation order inversion can be vital for a correct result. In other words, while we are still converting one sequence into another, neither the number of inputs and outputs or their order are assumed to be the same. Consider the following example which illustrates the obnoxious fact of German to place the verb at the end.

| German          | Haben Sie sich schon dieses grossartige Lehrwerk angeschaut? |
| :-------------- | :--------------------------------------- |
| English         | Did you already check out this excellent tutorial? |
| Wrong alignment | Did you yourself already this excellent tutorial looked-at? |

There are many more related problems. For instance, the order in which a user reads a webpage is a two-dimensional layout analysis problem. Likewise, for dialog problems we need to take world-knowledge and prior state into account. This is an active area of research.


## Unsupervised learning

All the examples so far are related to *Supervised Learning*, i.e. situations where we know what we want. Quite often, though, we simply want to learn as much about the data as possible. This sounds vague because it is. The type and number of questions we could ask is only limited by the creativity of the statistician asking the question. We will address a number of them later in this tutorial where we will provide matching examples. To whet your appetite, we list a few of them below:

* Is there a small number of prototypes that accurately summarize the data. E.g. given a set of photos, can we group  them into landscape photos, pictures of dogs, babies, cats, mountain peaks, etc.? Likewise, given a collection of users (with their behavior), can we group them into users with similar behavior? This problem is typically known as **clustering**.
* Is there a small number of parameters that accurately captures the relevant properties of the data? E.g. the trajectories of a ball are quite well described by velocity, diameter and mass of the ball. Tailors have developed a small number of parameters that describe human body shape fairly accurately for the purpose of fitting clothes. These problems are referred to as **subspace estimation** problems. If the dependence is linear, it is called **principal component analysis**.
* Is there a representation of (arbitrary structured) objects in Euclidean space (i.e. the space of vectors in $\mathbb{R}^n$) such that symbolic properties can be well matched? This is called **representation learning** and it is used, to describe entities and their relations such as Rome - Italy + France = Paris.
* Is there a description of the root causes of much of the data that we observe? For instance, if we have demographic data about house prices, pollution, crime, location, education, salaries, etc., can we discover how they are related simply based on empirical data? The field of **directed graphical models** and **causality** deals with this.
* An important and exciting recent development are **generative adversarial networks**. They are basically a procedural way of synthesizing data. The underlying statistical mechanisms are tests to check whether real and fake data are the same. We will devote a few notebooks to them.


## Environment

So far we didn't discuss at all yet, where all the data comes from, how we need to interact with the environment, whether it remembers what we did previously, if the environment wants to help us (e.g. a user reading text into a speech recognizer) or if it is out to beat us (e.g. in a game), or if it doesn't care (in most cases). Those problems are usually distinguished by monikers such as batch learning, online learning, control, and reinforcement learning.

We also didn't discuss what happens when training and test data are different (statisticians call this covariate shift). This is a problem that most of us will have experienced painfully when taking exams written by the lecturer, while the homeworks were composed by his TAs. Likewise, there is a large area of situations where we want our tools to be robust against malicious or malformed training data (robustness) or equally abnormal test data. We will introduce these aspects gradually throughout this tutorial to help practitioners deal with them in their work.

## Conclusion

Machine Learning is vast. We cannot possibly cover it all. On the other hand, the chain rule is simple, so it's easy to get started.
