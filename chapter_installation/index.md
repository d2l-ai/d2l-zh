# 安装
:label:`chap_installation`

为了让你开始运行并获得动手学习体验，我们需要为您配置一个能够运行Python、Jupyter notebook、相关库以及运行本书所需代码的环境。

## 安装Miniconda

最简单的途径是安装 [Miniconda](https://conda.io/en/latest/miniconda.html)。这里必须依赖Python 3.x 版本。如果已安装 conda，则可以跳过以下步骤。从网站下载相应的 Miniconda sh 文件，然后使用 `sh <FILENAME> -b` 从命令行执行安装。

对于 macOS 用户：

```bash
# 文件名可能会更改
sh Miniconda3-latest-MacOSX-x86_64.sh -b
```

对于 Linux 用户：

```bash
# 文件名可能会更改
sh Miniconda3-latest-Linux-x86_64.sh -b
```

接下来，初始化shell，使我们可以直接运行 `conda` 命令。

```bash
~/miniconda3/bin/conda init
```

现在关闭并重新打开当前的shell。你应该能用如下命令创建一个新的环境：

```bash
conda create --name d2l -y
```

## 下载 D2L 记事本

接下来，我们需要下载这本书的代码。您可以点击任何 HTML 页面顶部的 “Jupyter记事本文件” 选项卡下载并解压代码。如果你有 `unzip`（通过 `sudo apt install unzip` 安装）可以：

```bash
mkdir d2l-en && cd d2l-en
curl https://d2l.ai/d2l-en.zip -o d2l-en.zip
unzip d2l-en.zip && rm d2l-en.zip
```

现在我们激活 `d2l` 环境并安装 `pip`。
在执行此命令后的询问中输入 `y`。

```bash
conda activate d2l
conda install python=3.7 pip -y
```

## 安装框架和 `d2l` 包

:begin_tab:`mxnet,pytorch`
在安装深度学习框架之前，请先检查您的计算机上是否有正确的 GPU（在标准笔记本电脑上为显示器提供供电的 GPU 不算）。如果要在 GPU 服务器上安装，请查看 :ref:`subsec_gpu` 以获取有关安装GPU支持版本的说明。

否则，您可以安装 CPU 版本。这足够帮助您完成前几章，但在运行大型模型之前，你需要gpu。
:end_tab:

:begin_tab:`mxnet`
```bash
pip install mxnet==1.6.0
```
:end_tab:

:begin_tab:`pytorch`
```bash
pip install torch==1.5.1 torchvision -f https://download.pytorch.org/whl/torch_stable.html
```
:end_tab:

:begin_tab:`tensorflow`
您可以通过以下命令安装具有 CPU 和 GPU 支持的tensorflow：

```bash
pip install tensorflow==2.2.0 tensorflow-probability==0.10.0
```
:end_tab:

我们还需要安装 `d2l` 软件包，它封装了本书中常用的函数和类。

```bash
pip install -U d2l
```

安装完成后，我们现在可以通过运行以下命令打开 Jupyter notebook：

```bash
jupyter notebook
```

此时，您可以在 Web 浏览器中打开 http://localhost:8888（通常会自动打开）。然后我们可以运行这本书的每部分的代码。在运行书籍代码、更新深度学习框架或`d2l` 软件包之前，请执行 `conda activate d2l` 以激活运行环境。要退出环境，请运行 `conda deactivate`。

## GPU 支持
:label:`subsec_gpu`

:begin_tab:`mxnet,pytorch`
默认情况下，安装深度学习框架时不支持 GPU，以确保它在任何计算机（包括大多数笔记本电脑）上运行。本书的部分内容必需或建议使用GPU运行。如果你的计算机有NVIDIA显卡并且已安装 [CUDA](https://developer.nvidia.com/cuda-downloads)，则应安装GPU版本。如果你已经安装了仅CPU版本，则可能需要首先通过运行以下命令将其删除：
:end_tab:

:begin_tab:`tensorflow`
默认情况下，TensorFlow安装了GPU支持。如果你的计算机有NVIDIA显卡并且已安装 [CUDA](https://developer.nvidia.com/cuda-downloads)，那你就准备好了。
:end_tab:

:begin_tab:`mxnet`
```bash
pip uninstall mxnet
```
:end_tab:

:begin_tab:`pytorch`
```bash
pip uninstall torch
```
:end_tab:

:begin_tab:`mxnet,pytorch`
然后，我们需要找到您安装的CUDA 版本。你可以通过 `nvcc --version` 或 `cat /usr/local/cuda/version.txt` 查看它。假定您已安装CUDA 10.1，则可以使用以下命令进行安装：
:end_tab:

:begin_tab:`mxnet`
```bash
# For Windows users
pip install mxnet-cu101==1.6.0b20190926

# For Linux and macOS users
pip install mxnet-cu101==1.6.0
```
:end_tab:

:begin_tab:`pytorch`
```bash
pip install torch==1.5.1+cu101 -f https://download.pytorch.org/whl/torch_stable.html
```
:end_tab:

:begin_tab:`mxnet,pytorch`
您可以根据您的CUDA版本更改最后几位数字，例如：CUDA 10.0 的 `cu100` 和 CUDA 9.0 的 `cu90`。
:end_tab:

## 练习

1. 下载该书的代码并安装运行环境。

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/23)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/24)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/436)
:end_tab:
