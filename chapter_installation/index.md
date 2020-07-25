# 安装
:label:`chap_installation`

为了让您开始运行并获得动手学习体验，我们需要为您设置一个运行 Python、Jupyter 笔记本电脑、相关库以及运行本书所需的代码的环境。

## 安装米尼康达

最简单的方法就是安装 [Miniconda](https://conda.io/en/latest/miniconda.html)。Python 3.x 版本是必需的。如果已安装 conda，则可以跳过以下步骤。从网站下载相应的 Miniconda sh 文件，然后使用 `sh <FILENAME> -b` 从命令行执行安装。对于 macOS 用户：

```bash
# The file name is subject to changes
sh Miniconda3-latest-MacOSX-x86_64.sh -b
```

对于 Linux 用户：

```bash
# The file name is subject to changes
sh Miniconda3-latest-Linux-x86_64.sh -b
```

接下来，初始化外壳，以便我们可以直接运行 `conda`。

```bash
~/miniconda3/bin/conda init
```

现在关闭并重新打开当前的 shell。您应该能够创建一个新的环境，如下所示：

```bash
conda create --name d2l -y
```

## 下载 D2L 笔记本电脑

接下来，我们需要下载这本书的代码。您可以点击任何 HTML 页面顶部的 “所有笔记本” 选项卡下载并解压代码。或者，如果您有 `unzip`（否则运行 `sudo apt install unzip`）可用：

```bash
mkdir d2l-en && cd d2l-en
curl https://d2l.ai/d2l-en.zip -o d2l-en.zip
unzip d2l-en.zip && rm d2l-en.zip
```

现在我们要激活 `d2l` 环境并安装 `pip`。在此命令后面的查询中输入 `y`。

```bash
conda activate d2l
conda install python=3.7 pip -y
```

## 安装框架和 `d2l` 软件包

:begin_tab:`mxnet,pytorch`
在安装深度学习框架之前，请先检查您的计算机上是否有正确的 GPU（在标准笔记本电脑上为显示器提供电源的 GPU 不计入我们的目的）。如果要在 GPU 服务器上安装，请继续执行 :ref:`subsec_gpu` 以获取有关安装 GPU 支持版本的说明。

否则，您可以安装 CPU 版本。这将是足够的马力来帮助您完成前几章，但您需要在运行更大的模型之前访问 GPU。
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
您可以通过以下方式安装具有 CPU 和 GPU 支持的腾讯流：

```bash
pip install tensorflow==2.2.0 tensorflow-probability==0.10.0
```
:end_tab:

我们还安装了 `d2l` 软件包，它封装了本书中常用的函数和类。

```bash
pip install -U d2l
```

安装完成后，我们现在通过运行以下操作打开 Jupyter 笔记本：

```bash
jupyter notebook
```

此时，您可以在 Web 浏览器中打开 http://localhost:8888（通常会自动打开）。然后我们可以运行这本书的每个部分的代码。在运行书籍代码或更新深度学习框架或 `d2l` 软件包之前，请始终执行 `conda activate d2l` 以激活运行时环境。要退出环境，请运行 `conda deactivate`。

## GPU 支持
:label:`subsec_gpu`

:begin_tab:`mxnet,pytorch`
默认情况下，安装深度学习框架时不支持 GPU，以确保它在任何计算机（包括大多数笔记本电脑）上运行。本书的一部分要求或建议使用 GPU 运行。如果您的计算机具有 NVIDIA 显卡并且已安装 [CUDA](https://developer.nvidia.com/cuda-downloads)，则应安装启用 GPU 的版本。如果您已经安装了仅 CPU 版本，则可能需要首先通过运行以下操作将其删除：
:end_tab:

:begin_tab:`tensorflow`
默认情况下，TensorFlow 安装了 GPU 支持。如果您的计算机具有 NVIDIA 显卡并且已安装 [CUDA](https://developer.nvidia.com/cuda-downloads)，那么您都可以完成。
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
然后，我们需要找到您安装的 CUDA 版本。你可以通过 `nvcc --version` 或 `cat /usr/local/cuda/version.txt` 查看它。假定您已安装 CUDA 10.1，则可以使用以下命令进行安装：
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
您可以根据您的 CUDA 版本更改最后一位数字，例如：CUDA 10.0 的 `cu100` 和 CUDA 9.0 的 `cu90`。
:end_tab:

## 练习

1. 下载该书的代码并安装运行时环境。

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/23)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/24)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/436)
:end_tab:
