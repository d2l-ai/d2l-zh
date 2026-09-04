# 统计学
:label:`sec_statistics`

毫无疑问，要成为顶尖的深度学习从业者，训练出最先进且准确率高的模型至关重要。然而，模型性能的提升究竟显著，还是仅仅源于训练过程中的随机波动，往往并不明确。为了讨论估计值中的不确定性，我们必须学习一些统计学知识。


关于*统计学*（statistics）的最早记载可以追溯到9世纪的阿拉伯学者肯迪，他详细描述了如何利用统计与频率分析破解加密信息。800年后，现代统计学于18世纪在德国兴起，当时的研究人员主要关注人口和经济数据的收集与分析。如今，统计学是一门研究数据收集、处理、分析、解释和可视化的学科。统计学的核心理论也已广泛应用于学术界、工业界和政府部门的研究中。


更具体地说，统计学可以分为*描述性统计*（descriptive statistics）和*统计推断*（statistical inference）。前者侧重于概括和展示一组观测数据的特征，这组观测数据称为*样本*（sample）。样本取自*总体*（population），总体是实验所关注的同类个体、事物或事件的全集。与描述性统计不同，统计推断假定样本分布在一定程度上能够反映总体分布，进而根据给定样本推断总体的特征。


读者可能会问：“机器学习与统计学的本质区别是什么？”从根本上说，统计学关注推断问题，包括对变量之间的关系建模（如因果推断），以及检验模型参数的统计显著性（如A/B测试）。相比之下，机器学习强调在不显式编写规则、也不必理解每个参数作用的情况下作出准确预测。


本节将介绍三类统计推断方法：评估和比较估计量、进行假设检验以及构造置信区间。这些方法可以帮助我们推断给定总体的特征，即真实参数$\theta$。为简明起见，假设给定总体的真实参数$\theta$是标量。将其推广到$\theta$为向量或张量的情况并不困难，因此不再讨论。



## 评估和比较估计量

在统计学中，*估计量*（estimator）是给定样本的函数，用来估计真实参数$\theta$。观测样本$\{x_1,x_2,\ldots,x_n\}$后得到的$\theta$估计值记为$\hat{\theta}_n=\hat{f}(x_1,\ldots,x_n)$。

我们在:numref:`sec_maximum_likelihood`中已经见过估计量的简单例子。给定来自伯努利随机变量的若干样本，随机变量取1的概率的最大似然估计，可以用观测到的1的个数除以样本总数得到。类似地，一道练习要求证明：给定若干高斯分布样本，其均值的最大似然估计就是所有样本的平均值。这些估计量几乎不会恰好给出参数的真实值，但理想情况下，当样本很多时，估计值会非常接近真实值。

下面给出一个例子，同时展示均值为零、方差为一的高斯随机变量的真实密度，以及从该分布中抽取的一组样本。这里特意构造了$y$坐标，使每个点都清晰可见，并更直观地显示它们与原始密度的关系。

```{.python .input}
#@tab mxnet
from d2l import mxnet as d2l
from mxnet import np, npx
import random
npx.set_np()

# 对数据点采样并构造y坐标
epsilon = 0.1
random.seed(8675309)
xs = np.random.normal(loc=0, scale=1, size=(300,))

ys = [np.sum(np.exp(-(xs[:i] - xs[i])**2 / (2 * epsilon**2))
             / np.sqrt(2*np.pi*epsilon**2)) / len(xs) for i in range(len(xs))]

# 计算真实密度
xd = np.arange(np.min(xs), np.max(xs), 0.01)
yd = np.exp(-xd**2/2) / np.sqrt(2 * np.pi)

# 绘制结果
d2l.plot(xd, yd, 'x', '密度')
d2l.plt.scatter(xs, ys)
d2l.plt.axvline(x=0)
d2l.plt.axvline(x=np.mean(xs), linestyle='--', color='purple')
d2l.plt.title(f'样本均值：{float(np.mean(xs)):.2f}')
d2l.plt.show()
```

```{.python .input}
#@tab pytorch
from d2l import torch as d2l
import torch

torch.pi = torch.acos(torch.zeros(1)) * 2  # 在torch中定义pi

# 对数据点采样并构造y坐标
epsilon = 0.1
torch.manual_seed(8675309)
xs = torch.randn(size=(300,))

ys = torch.tensor(
    [torch.sum(torch.exp(-(xs[:i] - xs[i])**2 / (2 * epsilon**2))\
               / torch.sqrt(2*torch.pi*epsilon**2)) / len(xs)\
     for i in range(len(xs))])

# 计算真实密度
xd = torch.arange(torch.min(xs), torch.max(xs), 0.01)
yd = torch.exp(-xd**2/2) / torch.sqrt(2 * torch.pi)

# 绘制结果
d2l.plot(xd, yd, 'x', '密度')
d2l.plt.scatter(xs, ys)
d2l.plt.axvline(x=0)
d2l.plt.axvline(x=torch.mean(xs), linestyle='--', color='purple')
d2l.plt.title(f'样本均值：{float(torch.mean(xs).item()):.2f}')
d2l.plt.show()
```

```{.python .input}
#@tab tensorflow
from d2l import tensorflow as d2l
import tensorflow as tf

tf.pi = tf.acos(tf.zeros(1)) * 2  # 在TensorFlow中定义pi

# 对数据点采样并构造y坐标
epsilon = 0.1
xs = tf.random.normal((300,))

ys = tf.constant(
    [(tf.reduce_sum(tf.exp(-(xs[:i] - xs[i])**2 / (2 * epsilon**2)) \
               / tf.sqrt(2*tf.pi*epsilon**2)) / tf.cast(
        tf.size(xs), dtype=tf.float32)).numpy() \
     for i in range(tf.size(xs))])

# 计算真实密度
xd = tf.range(tf.reduce_min(xs), tf.reduce_max(xs), 0.01)
yd = tf.exp(-xd**2/2) / tf.sqrt(2 * tf.pi)

# 绘制结果
d2l.plot(xd, yd, 'x', '密度')
d2l.plt.scatter(xs, ys)
d2l.plt.axvline(x=0)
d2l.plt.axvline(x=tf.reduce_mean(xs), linestyle='--', color='purple')
d2l.plt.title(f'样本均值：{float(tf.reduce_mean(xs).numpy()):.2f}')
d2l.plt.show()
```

参数的估计量$\hat{\theta}_n$可以有许多种计算方法。本节介绍评估和比较估计量的三种常用指标：均方误差、标准差和统计偏差。

### 均方误差

评估估计量最简单的指标或许是*均方误差*（mean squared error，MSE，也称$l_2$损失），其定义为

$$\textrm{MSE} (\hat{\theta}_n, \theta) = E[(\hat{\theta}_n - \theta)^2].$$
:eqlabel:`eq_mse_est`

它可以量化估计值与真实值之间平方偏差的平均值。均方误差始终非负。如果读过:numref:`sec_linear_regression`，就会认出它是最常用的回归损失函数。用它评估估计量时，值越接近零，估计量就越接近真实参数$\theta$。


### 统计偏差

均方误差提供了一个自然的指标，但许多不同因素都可能使其增大。其中最重要的两个因素是：数据集的随机性引起的估计量波动，以及估计过程引入的系统误差。


首先度量系统误差。对于估计量$\hat{\theta}_n$，*统计偏差*（statistical bias）定义为

$$\textrm{bias}(\hat{\theta}_n) = E(\hat{\theta}_n - \theta) = E(\hat{\theta}_n) - \theta.$$
:eqlabel:`eq_bias`

注意，当$\textrm{bias}(\hat{\theta}_n)=0$时，估计量$\hat{\theta}_n$的期望等于参数的真实值。此时称$\hat{\theta}_n$为无偏估计量。一般来说，无偏估计量优于有偏估计量，因为它的期望与真实参数相同。


不过应当知道，有偏估计量在实践中经常使用。在某些情况下，如果不增加假设，无偏估计量并不存在或无法实际计算。这看似是估计量的重大缺陷，但实践中遇到的大多数估计量至少是渐近无偏的，也就是说，随着可用样本数趋于无穷，偏差趋于零：$\lim_{n\rightarrow\infty}\textrm{bias}(\hat{\theta}_n)=0$。


### 方差和标准差

其次度量估计量的随机性。回顾:numref:`sec_random_variables`，*标准差*（standard deviation，也称*标准误差*）定义为方差的平方根。可以用估计量的标准差或方差衡量其波动程度。

$$\sigma_{\hat{\theta}_n} = \sqrt{\textrm{Var} (\hat{\theta}_n )} = \sqrt{E[(\hat{\theta}_n - E(\hat{\theta}_n))^2]}.$$
:eqlabel:`eq_var_est`

比较:eqref:`eq_var_est`与:eqref:`eq_mse_est`十分重要。在前一个公式中，比较对象不是真实总体值$\theta$，而是$E(\hat{\theta}_n)$，即样本均值的期望。因此，这里度量的不是估计量通常偏离真实值多远，而是估计量本身的波动。


### 偏差—方差权衡

直观上，这两个主要因素都会影响均方误差。令人稍感意外的是，可以证明均方误差能够真正*分解*为这两个因素与第三个因素之和。也就是说，均方误差等于偏差的平方、方差与不可约误差之和。

$$
\begin{aligned}
\textrm{MSE} (\hat{\theta}_n, \theta) &= E[(\hat{\theta}_n - \theta)^2] \\
 &= E[(\hat{\theta}_n)^2] + E[\theta^2] - 2E[\hat{\theta}_n\theta] \\
 &= \textrm{Var} [\hat{\theta}_n] + E[\hat{\theta}_n]^2 + \textrm{Var} [\theta] + E[\theta]^2 - 2E[\hat{\theta}_n]E[\theta] \\
 &= (E[\hat{\theta}_n] - E[\theta])^2 + \textrm{Var} [\hat{\theta}_n] + \textrm{Var} [\theta] \\
 &= (E[\hat{\theta}_n - \theta])^2 + \textrm{Var} [\hat{\theta}_n] + \textrm{Var} [\theta] \\
 &= (\textrm{bias} [\hat{\theta}_n])^2 + \textrm{Var} (\hat{\theta}_n) + \textrm{Var} [\theta].\\
\end{aligned}
$$

上式称为*偏差—方差权衡*（bias-variance trade-off）。均方误差可以分为三个来源：高偏差造成的误差、高方差造成的误差以及不可约误差。偏差误差常见于简单模型（如线性回归模型），这类模型无法提取特征与输出之间的高维关系。当模型存在较大的偏差误差时，通常称其*欠拟合*（underfitting）或缺乏*灵活性*（flexibility），如:numref:`sec_generalization_basics`所述。高方差通常来自过于复杂、因而过度拟合训练数据的模型。因此，*过拟合*（overfitting）模型对数据中的微小波动十分敏感。当模型存在较大的方差时，通常称其过拟合或缺乏*泛化能力*（generalization），同样参见:numref:`sec_generalization_basics`。不可约误差则来自$\theta$本身的噪声。


### 用代码评估估计量

对于张量`a`，只需调用`a.std()`即可计算估计量的标准差，因此这里略去其实现，转而实现统计偏差和均方误差。

```{.python .input}
#@tab mxnet
# 统计偏差
def stat_bias(true_theta, est_theta):
    return(np.mean(est_theta) - true_theta)

# 均方误差
def mse(data, true_theta):
    return(np.mean(np.square(data - true_theta)))
```

```{.python .input}
#@tab pytorch
# 统计偏差
def stat_bias(true_theta, est_theta):
    return(torch.mean(est_theta) - true_theta)

# 均方误差
def mse(data, true_theta):
    return(torch.mean(torch.square(data - true_theta)))
```

```{.python .input}
#@tab tensorflow
# 统计偏差
def stat_bias(true_theta, est_theta):
    return(tf.reduce_mean(est_theta) - true_theta)

# 均方误差
def mse(data, true_theta):
    return(tf.reduce_mean(tf.square(data - true_theta)))
```

为了说明偏差—方差权衡公式，我们从正态分布$\mathcal{N}(\theta,\sigma^2)$中模拟生成10000个样本，其中$\theta=1$且$\sigma=4$。由于估计量是给定样本的函数，这里用样本均值估计该正态分布$\mathcal{N}(\theta,\sigma^2)$中的真实参数$\theta$。

```{.python .input}
#@tab mxnet
theta_true = 1
sigma = 4
sample_len = 10000
samples = np.random.normal(theta_true, sigma, sample_len)
theta_est = np.mean(samples)
theta_est
```

```{.python .input}
#@tab pytorch
theta_true = 1
sigma = 4
sample_len = 10000
samples = torch.normal(theta_true, sigma, size=(sample_len, 1))
theta_est = torch.mean(samples)
theta_est
```

```{.python .input}
#@tab tensorflow
theta_true = 1
sigma = 4
sample_len = 10000
samples = tf.random.normal((sample_len, 1), theta_true, sigma)
theta_est = tf.reduce_mean(samples)
theta_est
```

下面通过计算估计量的偏差平方与方差之和来验证权衡公式。首先计算估计量的均方误差。

```{.python .input}
#@tab all
mse(samples, theta_true)
```

接下来计算$\textrm{Var}(\hat{\theta}_n)+[\textrm{bias}(\hat{\theta}_n)]^2$。可以看到，两个值在数值精度范围内一致。

```{.python .input}
#@tab mxnet
bias = stat_bias(theta_true, theta_est)
np.square(samples.std()) + np.square(bias)
```

```{.python .input}
#@tab pytorch
bias = stat_bias(theta_true, theta_est)
torch.square(samples.std(unbiased=False)) + torch.square(bias)
```

```{.python .input}
#@tab tensorflow
bias = stat_bias(theta_true, theta_est)
tf.square(tf.math.reduce_std(samples)) + tf.square(bias)
```

## 进行假设检验


统计推断中最常见的主题是假设检验。虽然假设检验到20世纪初才得到普及，但其首次应用可以追溯到18世纪的约翰·阿布斯诺特。他研究了伦敦80年的出生记录，得出每年出生的男性多于女性的结论。此后，现代显著性检验继承了多位学者的成果：卡尔·皮尔逊提出了$p$值和皮尔逊卡方检验，威廉·戈塞特创立了学生$t$分布，罗纳德·费希尔则开创了零假设与显著性检验。

*假设检验*（hypothesis test）是一种评估证据是否反对关于总体的默认陈述的方法。这个默认陈述称为*零假设*（null hypothesis）$H_0$，我们尝试利用观测数据将其拒绝。$H_0$是统计显著性检验的起点。*备择假设*（alternative hypothesis）$H_A$（或$H_1$）是与零假设相反的陈述。零假设通常以陈述句形式提出某种变量关系。它应尽可能简洁明确，并且能够用统计理论检验。

假设你是一位化学家。在实验室中投入数千小时后，你研制出一种能够显著提高数学理解能力的新药。为了证明它的神奇功效，需要对其进行检验。自然，你可以招募一些志愿者服药，观察它能否帮助他们更好地学习数学。那么该如何开始呢？

首先，需要谨慎地随机选出两组志愿者，使他们按照某些指标衡量的数学理解能力没有差异。这两组通常称为试验组和对照组。*试验组*（test group，也称*处理组*）中的个体将服用药物；*对照组*（control group）则作为基准，除不服用该药物外，其环境设置与试验组完全相同。这样可以尽量减小其他所有变量的影响，只保留处理过程中自变量的影响。

其次，在服药一段时间后，需要使用相同指标衡量两组人的数学理解能力，例如让志愿者学习一个新数学公式后完成同一套测试。随后收集他们的成绩并比较结果。在这个例子中，零假设是两组之间没有差异，备择假设则是两组之间存在差异。

这仍不是完全形式化的描述，还有许多细节需要仔细考虑。例如，使用什么指标检验数学理解能力才合适？需要多少志愿者才能有把握宣称药物有效？试验应持续多久？如何判断两组之间是否存在差异？只关注平均成绩，还是也要关注成绩的波动范围？诸如此类。

由此，假设检验为实验设计以及判断观测结果的确定性提供了一个框架。如果能够表明零假设为真的可能性极低，就可以有把握地拒绝它。

为了完整说明假设检验的工作方式，下面引入一些术语，并将上述概念形式化。


### 统计显著性

*统计显著性*（statistical significance）衡量在不应拒绝零假设$H_0$时却错误地将其拒绝的概率，即

$$ \textrm{统计显著性}=1-\alpha=1-P(\textrm{拒绝 }H_0\mid H_0\textrm{ 为真}).$$

这种错误也称为*第一类错误*（type I error）或*假阳性*（false positive）。$\alpha$称为*显著性水平*（significance level），常用取值为$5\%$，即$1-\alpha=95\%$。显著性水平可以理解为拒绝一个真实的零假设时，我们愿意承担的风险程度。

:numref:`fig_statistical_significance`展示了双样本假设检验中，给定正态分布的观测值与概率。如果某个观测样本位于$95\%$阈值之外，那么在零假设成立的条件下，它极不可能出现。因此，零假设可能有问题，我们将拒绝它。

![统计显著性。](../img/statistical-significance.svg)
:label:`fig_statistical_significance`


### 统计功效

*统计功效*（statistical power，也称*灵敏度*）衡量应当拒绝零假设$H_0$时正确将其拒绝的概率，即

$$ \textrm{统计功效}=1-\beta=1-P(\textrm{未拒绝 }H_0\mid H_0\textrm{ 为假}).$$

回顾一下，第一类错误是在零假设为真时将其拒绝；*第二类错误*（type II error）则是在零假设为假时未能将其拒绝。第二类错误通常记为$\beta$，因此相应的统计功效为$1-\beta$。


直观地说，统计功效表示在期望的统计显著性水平下，检验发现某种最小幅度真实差异的可能性。$80\%$是常用的统计功效阈值。统计功效越高，发现真实差异的可能性就越大。

统计功效最常见的用途之一是确定所需样本数。当零假设为假时，拒绝它的概率取决于它偏离真实情况的程度（称为*效应量*（effect size））以及样本数。正如可以预料的，效应量很小时，需要大量样本才能以较高概率检测出来。详细推导超出了本附录的范围，这里举一个例子：假设零假设认为样本来自均值为零、方差为一的高斯分布，而我们认为样本的实际均值接近一，那么只需$8$个样本就能以可接受的错误率拒绝零假设。然而，如果认为样本总体的真实均值接近$0.01$，则需要近80000个样本才能检测出差异。

可以把统计功效想象成净水器。在这个类比中，高功效的假设检验就像高质量的净水系统，会尽可能滤除水中的有害物质；较小的差异则像尺寸较小的杂质，容易从低质量滤网的缝隙中漏过。同理，如果统计功效不够高，检验就可能无法发现较小的差异。


### 检验统计量

*检验统计量*（test statistic）$T(x)$是概括样本数据某种特征的标量。定义这种统计量的目的，是让我们能够区分不同分布并进行假设检验。回到化学家的例子，如果想证明一个总体的表现优于另一个总体，把均值作为检验统计量可能很合理。选择不同的检验统计量，可能使统计检验的统计功效产生巨大差异。

在零假设下，$T(X)$（检验统计量在零假设下的分布）通常至少近似服从某种常见概率分布，如正态分布。如果能够显式推导出该分布，再在数据集上计算检验统计量，那么当统计量远远落在预期范围之外时，就可以有把握地拒绝零假设。将这一判断量化，便得到$p$值的概念。


### $p$值

假设零假设*为真*，$p$值（或*概率值*（probability value））是$T(X)$至少与观测到的检验统计量$T(x)$同样极端的概率，即

$$ p\textrm{-value} = P_{H_0}(T(X) \geq T(x)).$$

如果$p$值小于或等于预先设定的统计显著性水平$\alpha$，就可以拒绝零假设；否则，只能认为没有足够证据拒绝零假设。对于给定的总体分布，*拒绝域*（region of rejection）是由$p$值小于统计显著性水平$\alpha$的所有点组成的区域。


### 单侧检验和双侧检验

显著性检验通常分为单侧检验和双侧检验。当零假设与备择假设只有一个方向时，适合使用*单侧检验*（one-sided test，也称*单尾检验*）。例如，零假设可以声称真实参数$\theta$小于或等于某个值$c$，备择假设则是$\theta$大于$c$。也就是说，拒绝域只位于抽样分布的一侧。与之相对，当拒绝域位于抽样分布的两侧时，适合使用*双侧检验*（two-sided test，也称*双尾检验*）。例如，零假设可以声称真实参数$\theta$等于$c$，而备择假设是$\theta$不等于$c$。


### 假设检验的一般步骤

熟悉上述概念后，下面梳理假设检验的一般步骤。

1. 陈述问题并建立零假设$H_0$。
2. 设定统计显著性水平$\alpha$和统计功效$1-\beta$。
3. 通过实验获得样本。所需样本数取决于统计功效和预期效应量。
4. 计算检验统计量和$p$值。
5. 根据$p$值和统计显著性水平$\alpha$，决定保留还是拒绝零假设。

进行假设检验时，首先定义零假设以及愿意承担的风险水平。然后计算样本的检验统计量，并把检验统计量的极端值视为反对零假设的证据。如果检验统计量落入拒绝域，就可以拒绝零假设而接受备择假设。

假设检验适用于临床试验和A/B测试等多种场景。


## 构造置信区间


估计参数$\theta$的值时，$\hat\theta$这样的点估计量作用有限，因为它们没有表达不确定性。如果能给出一个以高概率包含真实参数$\theta$的区间，就会有用得多。如果一个世纪前就对这一思想感兴趣，那么一定会兴奋地读到耶日·内曼的《基于经典概率理论的统计估计理论纲要》:cite:`Neyman.1937`，他于1937年首次提出置信区间的概念。

在给定确定性程度下，置信区间应尽可能小才有实用价值。下面看看如何推导它。


### 定义

在数学上，真实参数$\theta$的*置信区间*（confidence interval）是一个由样本数据计算得到的区间$C_n$，满足

$$P_{\theta} (C_n \ni \theta) \geq 1 - \alpha, \forall \theta.$$
:eqlabel:`eq_confidence`

其中$\alpha\in(0,1)$，$1-\alpha$称为区间的*置信水平*（confidence level）或*覆盖率*（coverage）。这里的$\alpha$与前文显著性水平中的$\alpha$相同。

注意，:eqref:`eq_confidence`讨论的是随机变量$C_n$，而不是固定的$\theta$。为了强调这一点，写成$P_{\theta}(C_n\ni\theta)$，而不是$P_{\theta}(\theta\in C_n)$。

### 解释

人们很容易把$95\%$置信区间解释为“真实参数有$95\%$的概率位于这个区间内”，但遗憾的是，这并不正确。真实参数是固定的，随机的是区间。因此，更准确的解释是：如果按照这一过程生成大量置信区间，其中$95\%$的区间会包含真实参数。

这看似是在咬文嚼字，却会实际影响结果的解释。特别是，只要这种情况发生得足够少，即使构造出我们*几乎确信*不包含真实值的区间，也仍可能满足:eqref:`eq_confidence`。下面列出三个看似合理但错误的说法。关于这些问题的深入讨论可参见:citet:`Morey.Hoekstra.Rouder.ea.2016`。

* **谬误1**：置信区间越窄，就说明参数估计越精确。
* **谬误2**：置信区间内的值比区间外的值更可能是真实值。
* **谬误3**：某个已经观测到的$95\%$置信区间包含真实值的概率是$95\%$。

总之，置信区间是一个微妙的概念。不过，只要始终明确其正确解释，它就是一种强大的工具。

### 高斯分布示例

下面讨论最经典的例子：均值和方差未知的高斯分布，其均值的置信区间。假设从高斯分布$\mathcal{N}(\mu,\sigma^2)$中收集$n$个样本$\{x_i\}_{i=1}^n$。均值和方差的估计量可以计算为

$$\hat\mu_n = \frac{1}{n}\sum_{i=1}^n x_i, \qquad \hat\sigma^2_n = \frac{1}{n-1}\sum_{i=1}^n (x_i - \hat\mu)^2.$$

现在考虑随机变量

$$
T = \frac{\hat\mu_n - \mu}{\hat\sigma_n/\sqrt{n}},
$$

它服从一个著名的分布，称为自由度为$n-1$的*学生$t$分布*（Student's t-distribution）。

这一分布得到了充分研究。例如，已知当$n\rightarrow\infty$时，它近似于标准高斯分布。因此，查阅高斯分布累积分布函数表可知，$T$至少有$95\%$的概率落在区间$[-1.96,1.96]$内。当$n$有限时，所需区间会稍大一些，但这些区间也早已算出并列入表中。

因此，当$n$较大时，可以得到

$$
P\left(\frac{\hat\mu_n - \mu}{\hat\sigma_n/\sqrt{n}} \in [-1.96, 1.96]\right) \ge 0.95.
$$

将不等式各部分乘以$\hat\sigma_n/\sqrt{n}$，再加上$\hat\mu_n$，得到

$$
P\left(\mu \in \left[\hat\mu_n - 1.96\frac{\hat\sigma_n}{\sqrt{n}}, \hat\mu_n + 1.96\frac{\hat\sigma_n}{\sqrt{n}}\right]\right) \ge 0.95.
$$

由此得到$95\%$置信区间：
$$\left[\hat\mu_n - 1.96\frac{\hat\sigma_n}{\sqrt{n}}, \hat\mu_n + 1.96\frac{\hat\sigma_n}{\sqrt{n}}\right].$$
:eqlabel:`eq_gauss_confidence`

可以说，:eqref:`eq_gauss_confidence`是统计学中最常用的公式之一。最后用代码实现它。为简单起见，假设当前处于渐近情形。$N$较小时，应当以编程方式或查$t$分布表得到正确的`t_star`值。

```{.python .input}
#@tab mxnet
# 样本数
N = 1000

# 对数据集采样
samples = np.random.normal(loc=0, scale=1, size=(N,))

# 查学生t分布的累积分布函数
t_star = 1.96

# 构造区间
mu_hat = np.mean(samples)
sigma_hat = samples.std(ddof=1)
(mu_hat - t_star*sigma_hat/np.sqrt(N), mu_hat + t_star*sigma_hat/np.sqrt(N))
```

```{.python .input}
#@tab pytorch
# PyTorch默认使用贝塞尔校正，即ddof=1
# 而NumPy默认为ddof=0，可以用unbiased=False模拟ddof=0

# 样本数
N = 1000

# 对数据集采样
samples = torch.normal(0, 1, size=(N,))

# 查学生t分布的累积分布函数
t_star = 1.96

# 构造区间
mu_hat = torch.mean(samples)
sigma_hat = samples.std(unbiased=True)
(mu_hat - t_star*sigma_hat/torch.sqrt(torch.tensor(N, dtype=torch.float32)),\
 mu_hat + t_star*sigma_hat/torch.sqrt(torch.tensor(N, dtype=torch.float32)))
```

```{.python .input}
#@tab tensorflow
# 样本数
N = 1000

# 对数据集采样
samples = tf.random.normal((N,), 0, 1)

# 查学生t分布的累积分布函数
t_star = 1.96

# 构造区间
mu_hat = tf.reduce_mean(samples)
sigma_hat = tf.math.reduce_std(samples)
(mu_hat - t_star*sigma_hat/tf.sqrt(tf.constant(N, dtype=tf.float32)), \
 mu_hat + t_star*sigma_hat/tf.sqrt(tf.constant(N, dtype=tf.float32)))
```

## 小结

* 统计学关注推断问题，而深度学习强调在不显式编写规则和理解参数作用的情况下作出准确预测。
* 三类常见的统计推断方法是：评估和比较估计量、进行假设检验以及构造置信区间。
* 评估估计量的三个常用指标是统计偏差、标准差和均方误差。
* 置信区间是根据给定样本构造的、对总体真实参数所在范围的估计。
* 假设检验是一种评估证据是否反对关于总体的默认陈述的方法。


## 练习

1. 设$X_1,X_2,\ldots,X_n\overset{\textrm{iid}}{\sim}\textrm{Unif}(0,\theta)$，其中“iid”表示*独立同分布*（independent and identically distributed）。考虑以下两个$\theta$的估计量：
$$\hat{\theta} = \max \{X_1, X_2, \ldots, X_n \};$$
$$\tilde{\theta} = 2 \bar{X_n} = \frac{2}{n} \sum_{i=1}^n X_i.$$
    * 求$\hat{\theta}$的统计偏差、标准差和均方误差。
    * 求$\tilde{\theta}$的统计偏差、标准差和均方误差。
    * 哪个估计量更好？
1. 对于引言中的化学家示例，在统计显著性水平$\alpha=0.05$、统计功效$1-\beta=0.8$时，请给出进行双侧假设检验的5个步骤。
1. 令$N=2$、$\alpha=0.5$，对100个独立生成的数据集运行置信区间代码并绘制所得区间（此时`t_star = 1.0`）。可以看到若干非常短、远未包含真实均值$0$的区间。这是否违背了置信区间的解释？你是否愿意用短区间表示高精度估计？

:begin_tab:`mxnet`
[讨论](https://discuss.d2l.ai/t/419)
:end_tab:

:begin_tab:`pytorch`
[讨论](https://discuss.d2l.ai/t/1102)
:end_tab:

:begin_tab:`tensorflow`
[讨论](https://discuss.d2l.ai/t/1103)
:end_tab:
