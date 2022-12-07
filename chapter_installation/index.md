# 安装
:label:`chap_installation`

我们需要配置一个环境来运行 Python、Jupyter Notebook、相关库以及运行本书所需的代码，以快速入门并获得动手学习经验。

## 安装 Miniconda

最简单的方法就是安装依赖Python 3.x的[Miniconda](https://conda.io/en/latest/miniconda.html)。
如果已安装conda，则可以跳过以下步骤。访问Miniconda网站，根据Python3.x版本确定适合的版本。

如果我们使用macOS，假设Python版本是3.9（我们的测试版本），将下载名称包含字符串“MacOSX”的bash脚本，并执行以下操作：

```bash
# 以Intel处理器为例，文件名可能会更改
sh Miniconda3-py39_4.12.0-MacOSX-x86_64.sh -b
```


如果我们使用Linux，假设Python版本是3.9（我们的测试版本），将下载名称包含字符串“Linux”的bash脚本，并执行以下操作：

```bash
# 文件名可能会更改
sh Miniconda3-py39_4.12.0-Linux-x86_64.sh -b
```


接下来，初始化终端Shell，以便我们可以直接运行`conda`。

```bash
~/miniconda3/bin/conda init
```


现在关闭并重新打开当前的shell。并使用下面的命令创建一个新的环境：

```bash
conda create --name d2l python=3.9 -y
```


现在激活 `d2l` 环境：

```bash
conda activate d2l
```


## 安装深度学习框架和`d2l`软件包

在安装深度学习框架之前，请先检查计算机上是否有可用的GPU。
例如可以查看计算机是否装有NVIDIA GPU并已安装[CUDA](https://developer.nvidia.com/cuda-downloads)。
如果机器没有任何GPU，没有必要担心，因为CPU在前几章完全够用。
但是，如果想流畅地学习全部章节，请提早获取GPU并且安装深度学习框架的GPU版本。


:begin_tab:`mxnet`

安装MXNet的GPU版本，首先需要知道已安装的CUDA版本。
（可以通过运行`nvcc --version`或`cat /usr/local/cuda/version.txt`来检验。）
假设已安装CUDA 10.1版本，请执行以下命令：

```bash
# 对于Linux和macOS用户
pip install mxnet-cu101==1.7.0

# 对于Windows用户
pip install mxnet-cu101==1.7.0 -f https://dist.mxnet.io/python
```


可以根据CUDA版本更改如上`mxnet-cu101`的最后一位数字，
例如：CUDA 10.0是`cu100`， CUDA 9.0是`cu90`。


如果机器没有NVIDIA GPU或CUDA，可以按如下方式MXNet的CPU版本：

```bash
pip install mxnet==1.7.0.post1
```


:end_tab:


:begin_tab:`pytorch`

我们可以按如下方式安装PyTorch的CPU或GPU版本：

```bash
pip install torch==1.12.0
pip install torchvision==0.13.0
```


:end_tab:

:begin_tab:`tensorflow`
我们可以按如下方式安装TensorFlow的CPU或GPU版本：

```bash
pip install tensorflow==2.8.0
pip install tensorflow-probability==0.16.0
```


:end_tab:

:begin_tab:`paddle`
安装PaddlePaddle的GPU版本，首先需要知道已安装的CUDA版本。
（可以通过运行`nvcc --version`或`cat /usr/local/cuda/version.txt`来检验。）
假设已安装CUDA 11.2版本，请执行以下命令：

```bash
python -m pip install paddlepaddle-gpu==2.3.2.post112 -f https://www.paddlepaddle.org.cn/whl/linux/mkl/avx/stable.html
```


如果机器没有NVIDIA GPU或CUDA，可以按如下方式PaddlePaddle的CPU版本：

```bash
python -m pip install paddlepaddle==2.3.2 -i https://pypi.tuna.tsinghua.edu.cn/simple
```


:end_tab:

我们的下一步是安装`d2l`包，以方便调取本书中经常使用的函数和类：

```bash
pip install d2l==0.17.6
```


## 下载 D2L Notebook

接下来，需要下载这本书的代码。
可以点击本书HTML页面顶部的“Jupyter 记事本”选项下载后解压代码，或者可以按照如下方式进行下载：


:begin_tab:`mxnet`

```bash
mkdir d2l-zh && cd d2l-zh
curl https://zh-v2.d2l.ai/d2l-zh-2.0.0.zip -o d2l-zh.zip
unzip d2l-zh.zip && rm d2l-zh.zip
cd mxnet
```


注意：如果没有安装`unzip`，则可以通过运行`sudo apt install unzip`进行安装。

:end_tab:


:begin_tab:`pytorch`

```bash
mkdir d2l-zh && cd d2l-zh
curl https://zh-v2.d2l.ai/d2l-zh-2.0.0.zip -o d2l-zh.zip
unzip d2l-zh.zip && rm d2l-zh.zip
cd pytorch
```


注意：如果没有安装`unzip`，则可以通过运行`sudo apt install unzip`进行安装。

:end_tab:


:begin_tab:`tensorflow`

```bash
mkdir d2l-zh && cd d2l-zh
curl https://zh-v2.d2l.ai/d2l-zh-2.0.0.zip -o d2l-zh.zip
unzip d2l-zh.zip && rm d2l-zh.zip
cd tensorflow
```


注意：如果没有安装`unzip`，则可以通过运行`sudo apt install unzip`进行安装。

:end_tab:


:begin_tab:`paddle`

```bash
mkdir d2l-zh && cd d2l-zh
curl https://zh-v2.d2l.ai/d2l-zh-2.0.0.zip -o d2l-zh.zip
unzip d2l-zh.zip && rm d2l-zh.zip
cd paddle
```


注意：如果没有安装`unzip`，则可以通过运行`sudo apt install unzip`进行安装。

:end_tab:


安装完成后我们可以通过运行以下命令打开Jupyter笔记本（在Window系统的命令行窗口中运行以下命令前，需先将当前路径定位到刚下载的本书代码解压后的目录）：

```bash
jupyter notebook
```


现在可以在Web浏览器中打开<http://localhost:8888>（通常会自动打开）。
由此，我们可以运行这本书中每个部分的代码。
在运行书籍代码、更新深度学习框架或`d2l`软件包之前，请始终执行`conda activate d2l`以激活运行时环境。
要退出环境，请运行`conda deactivate`。



:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/2082)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/2083)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/2084)
:end_tab:

:begin_tab:`paddle`
[Discussions](https://discuss.d2l.ai/t/11679)
:end_tab:
