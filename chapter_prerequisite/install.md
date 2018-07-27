# 安装和运行

为了动手学深度学习，我们需要获取本书代码，安装并运行Python、MXNet、`gluonbook`、Jupyter notebook等工具。


## 进入命令行模式

在这一节中，我们将描述安装和运行所需要的命令。执行命令需要进入命令行模式：Windows用户可以在文件资源管理器的地址栏输入`cmd`并按回车键，Linux/macOS用户可以打开Terminal应用。


## 获取代码并安装运行环境

我们可以通过Conda来获取本书代码并安装运行环境。Windows和Linux/macOS用户请分别参照以下步骤。


### Windows用户

第一步，根据操作系统下载并安装Miniconda（网址：https://conda.io/miniconda.html ），在安装过程中需要勾选“Add Anaconda to my PATH environment variable”选项。

第二步，下载包含本书全部代码的压缩包。我们可以在浏览器的地址栏中输入以下地址并按回车键进行下载：

> https://zh.gluon.ai/gluon_tutorials_zh-1.0.zip

下载完成后，创建文件夹“gluon_tutorials_zh-1.0”并将以上压缩包解压到这个文件夹。在该目录文件资源管理器的地址栏输入`cmd`进入命令行模式。

第三步，安装运行所需的软件包并激活该运行环境。我们可以先通过运行下面命令来配置下载源，从而使用国内镜像加速下载:

```
# 优先使用清华 conda 镜像。
conda config --prepend channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free/

# 或者选用科大 conda 镜像。
conda config --prepend channels http://mirrors.ustc.edu.cn/anaconda/pkgs/free/
```

然后运行以下命令安装并激活运行环境。

```
conda env create -f environment.yml
activate gluon
```

第四步，打开Juputer notebook。运行下面命令。

```
jupyter notebook
```

这时在浏览器打开 http://localhost:8888 （通常会自动打开）就可以查看和运行本书中每一节的代码了。

第五步（可选项），如果你是国内用户，建议使用国内Gluon镜像加速数据集和预训练模型的下载。运行下面命令。

```
set MXNET_GLUON_REPO=https://apache-mxnet.s3.cn-north-1.amazonaws.com.cn/ jupyter notebook
```

### Linux/macOS用户

第一步，根据操作系统下载并安装Miniconda（网址：https://conda.io/miniconda.html ）。

安装时会显示使用条款，按“↓”继续阅读，按“Q”退出阅读。之后需要回答下面几个问题：

```
Do you accept the license terms? [yes|no]
[no] >>> yes
Do you wish the installer to prepend the Miniconda3 install location
to PATH in your /xx/yyy/.zzzz ? [yes|no]
[no] >>> yes
```

安装完成后，我们需要让conda生效。Linux用户需要运行一次`source ~/.bashrc`或重启命令行；macOS用户需要运行一次`source ~/.bash_profile`或重启命令行。

第二步，下载包含本书全部代码的压缩包，解压后进入文件夹。运行如下命令。

```
mkdir gluon_tutorials_zh-1.0 && cd gluon_tutorials_zh-1.0
curl https://zh.gluon.ai/gluon_tutorials_zh-1.0.tar.gz -o tutorials.tar.gz
tar -xzvf tutorials.tar.gz && rm tutorials.tar.gz
```

第三步，安装运行所需的软件包并激活该运行环境。我们可以先通过运行下面命令来配置下载源，从而使用国内镜像加速下载:

```
# 优先使用清华 conda 镜像。
conda config --prepend channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free/

# 或者选用科大 conda 镜像。
conda config --prepend channels http://mirrors.ustc.edu.cn/anaconda/pkgs/free/
```

然后运行以下命令安装并激活运行环境。

```
conda env create -f environment.yml
source activate gluon
```

由于教程会使用`matplotlib.plot`函数作图，macOS用户需要创建或访问`~/.matplotlib/matplotlibrc`文件并添加一行代码：`backend: TkAgg`。

第四步，打开Juputer notebook。运行下面命令。

```
jupyter notebook
```

这时在浏览器打开 http://localhost:8888 （通常会自动打开）就可以查看和运行本书中每一节的代码了。

第五步（可选项），如果你是国内用户，建议使用国内Gluon镜像加速数据集和预训练模型的下载。运行下面命令。

```
MXNET_GLUON_REPO=https://apache-mxnet.s3.cn-north-1.amazonaws.com.cn/ jupyter notebook
```

## 激活运行环境

运行环境中已安装了运行书中代码所需的Python、MXNet、`gluonbook`、Jupyter notebook等工具。我们可以在下载并解压的代码包里的文件“gluon_tutorials_zh-1.0/environment.yml”中查看它们。在运行书中代码前，我们需要激活运行环境。Windows和Linux/macOS用户请分别参照以下步骤激活并退出运行环境。

### Windows用户

首先进入之前解压得到的文件夹“gluon_tutorials_zh-1.0”。然后在该目录文件资源管理器的地址栏输入`cmd`进入命令行模式。最后运行以下命令激活安装环境。

```
activate gluon
```

如需退出激活环境，运行以下命令。

```
deactivate
```

###  Linux/macOS用户

首先在命令行模式下进入之前解压得到的文件夹“gluon_tutorials_zh-1.0”（例如运行`cd gluon_tutorials_zh-1.0`），然后运行以下命令激活安装环境。

```
source activate gluon
```

如需退出激活环境，运行以下命令。

```
source deactivate
```

## 更新代码和运行环境

为了适应深度学习和MXNet的快速发展，本书的开源内容将定期发布新版本。我们推荐大家定期更新本书的开源内容（例如代码）和相应的运行环境（例如新版MXNet）。以下是更新的具体步骤。


第一步，重新下载最新的包含本书全部代码的包。下载地址可以从以下二者之间选择。

* https://zh.gluon.ai/gluon_tutorials_zh.zip
* https://zh.gluon.ai/gluon_tutorials_zh.tar.gz

解压后进入文件夹“gluon_tutorials_zh”。

第二步，使用下面命令更新运行环境。

```
conda env update -f environment.yml
```

## 使用GPU版的MXNet

通过前面介绍的方式安装的MXNet只支持CPU计算。本书中有部分章节需要或推荐使用GPU来运行。如果你的电脑上有Nvidia显卡并安装了CUDA，建议使用GPU版的MXNet。

我们在完成获取代码并安装运行环境的步骤后，需要先激活运行环境。然后卸载CPU版本的MXNet：

```
pip uninstall mxnet
```

接下来，退出运行环境。使用文本编辑器打开之前解压得到的代码包里的文件“gluon_tutorials_zh-1.0/environment.yml”。如果电脑上装的是8.0版本的CUDA，将该文件中的字符串“mxnet”改为“mxnet-cu80”。如果电脑上安装了其他版本的CUDA（比如7.5、9.0、9.2等），对该文件中的字符串“mxnet”做类似修改（比如改为“mxnet-cu75”、“mxnet-cu90”、“mxnet-cu92”等）。然后，使用下面命令更新运行环境。

```
conda env update -f environment.yml
```

之后，我们只需要再激活安装环境就可以使用GPU版的MXNet运行书中代码了。

### 更新代码和运行环境

如果使用GPU版的MXNet，更新代码和运行环境可参照以下步骤：

第一步，重新下载最新的包含本书全部代码的包。下载地址可以从以下二者之间选择。

* https://zh.gluon.ai/gluon_tutorials_zh.zip
* https://zh.gluon.ai/gluon_tutorials_zh.tar.gz

解压后进入文件夹“gluon_tutorials_zh”。

第二步，使用文本编辑器打开文件夹“gluon_tutorials_zh”中的环境配置文件“environment.yml”。如果电脑上装的是8.0版本的CUDA，将该文件中的字符串“mxnet”改为“mxnet-cu80”。如果电脑上安装了其他版本的CUDA（比如7.5、9.0、9.2等），对该文件中的字符串“mxnet”做类似修改（比如改为“mxnet-cu75”、“mxnet-cu90”、“mxnet-cu92”等）。


第三步，使用下面命令更新运行环境。

```
conda env update -f environment.yml
```

## 小结

* 为了能够动手学深度学习，我们需要获取本书代码并安装运行环境。
* 我们建议大家定期更新代码和运行环境。


## 练习

* 获取本书代码并安装运行环境。如果你在安装时遇到任何问题，请扫一扫本节二维码。在讨论区，你可以查阅疑难问题汇总或者提问。


## 扫码直达[讨论区](https://discuss.gluon.ai/t/topic/249)

![](../img/qr_install.svg)
