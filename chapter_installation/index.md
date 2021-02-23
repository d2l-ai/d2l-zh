# 安装
:label:`chap_installation`

我们需要设置一个环境来运行 Python，Jupyter Notebook，相关库以及运行本书所需的代码，以快速入门并获得动手学习经验。

## 安装 Miniconda

最简单的方法就是安装依赖 Python 3.x 的 [Miniconda](https://conda.io/en/latest/miniconda.html)。如果已安装 conda，则可以跳过以下步骤。从网站下载相应的 Miniconda sh 文件，然后使用 `sh <FILENAME> -b` 从命令行执行安装。对于 macOS 用户：

```bash
# The file name is subject to changes
sh Miniconda3-latest-MacOSX-x86_64.sh -b
```

对于 Linux 用户：

```bash
# The file name is subject to changes
sh Miniconda3-latest-Linux-x86_64.sh -b
```

接下来，初始化终端 Shell，以便我们可以直接运行 `conda`。

```bash
~/miniconda3/bin/conda init
```

现在关闭并重新打开当前的 shell。你应该能够创建一个新的环境，如下所示：

```bash
conda create --name d2l python=3.8 -y
```

## 下载 D2L Notebook

接下来，需要下载这本书的代码。你可以点击任何 HTML 页面顶部的 “Jupyter 笔记本文件” 选项卡下载，并解压代码。或者可以按照如下方式进行下载：

```bash
mkdir d2l-zh && cd d2l-zh
curl https://d2l.ai/d2l-zh.zip -o d2l-zh.zip
unzip d2l-zh.zip && rm d2l-zh.zip
```

注意：如果没有安装 `unzip`，则可以通过运行 `sudo apt install unzip` 进行安装。

现在我们要激活 `d2l` 环境。

```bash
conda activate d2l
```

## 安装框架和 `d2l` 软件包

在安装深度学习框架之前，请先检查你的计算机上是否有正确的 GPU（在标准笔记本电脑上为显示器提供电源的 GPU 不计入我们的目的）。如果要在 GPU 服务器上安装，请继续执行 :ref:`subsec_gpu` 以获取有关安装 GPU 支持版本的说明。

或者，你可以按照如下方法安装CPU版本。这将足够帮助你完成前几章，但你需要在运行更大的模型之前访问 GPU。

:begin_tab:`mxnet`

```bash
pip install mxnet==1.7.0.post1
```
:end_tab:

:begin_tab:`pytorch`

```bash
pip install torch torchvision -f https://download.pytorch.org/whl/torch_stable.html
```
:end_tab:

:begin_tab:`tensorflow`
你可以通过以下方式安装具有 CPU 和 GPU 支持的 TensorFlow：

```bash
pip install tensorflow tensorflow-probability
```
:end_tab:

我们还安装了 `d2l` 软件包，它封装了本书中常用的函数和类。

```bash
# -U：将所有包升级到最新的可用版本
pip install -U d2l
```

安装完成后，我们现在通过运行以下操作打开 Jupyter 笔记本：

```bash
jupyter notebook
```

此时，你可以在 Web 浏览器中打开 <http://localhost:8888>（通常会自动打开）。然后我们可以运行这本书的每个部分的代码。在运行书籍代码或更新深度学习框架或 `d2l` 软件包之前，请始终执行 `conda activate d2l` 以激活运行时环境。要退出环境，请运行 `conda deactivate`。

## GPU 支持
:label:`subsec_gpu`

:begin_tab:`mxnet`
默认情况下，安装MXNet时不支持 GPU，以确保它在任何计算机（包括大多数笔记本电脑）上运行。本书的一部分要求或建议使用 GPU 运行。如果你的计算机具有 NVIDIA 显卡并且已安装 [CUDA](https://developer.nvidia.com/cuda-downloads)，则应安装启用 GPU 的版本。如果你已经安装了仅 CPU 版本，则可能需要首先通过运行以下操作将其删除：

```bash
pip uninstall mxnet
```


然后，我们需要找到你安装的 CUDA 版本。你可以通过 `nvcc --version` 或 `cat /usr/local/cuda/version.txt` 查看它。假定你已安装 CUDA 10.1，则可以使用以下命令进行安装：


```bash
# For Windows users
pip install mxnet-cu101==1.7.0 -f https://dist.mxnet.io/python

# For Linux and macOS users
pip install mxnet-cu101==1.7.0
```


你可以根据你的 CUDA 版本更改最后一位数字，例如：CUDA 10.0 的 `cu100` 和 CUDA 9.0 的 `cu90`。
:end_tab:

:begin_tab:`pytorch,tensorflow`
默认情况下，深度学习框架安装了GPU支持。
如果你的计算机有NVIDIA GPU，并且已经安装了[CUDA](https://developer.nvidia.com/cuda-downloads)，那么你应该已经设置好了。
:end_tab:

## 练习

1. 下载该书的代码并安装运行时环境。

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/2082)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/2083)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/2084)
:end_tab:
