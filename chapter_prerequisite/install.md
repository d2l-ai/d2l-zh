# 获取和运行本书代码

本节将介绍如何获取本书代码和安装运行所需软件。虽然跳过本节不会影响后面的阅读，但我们强烈建议你按照下面的步骤来动手操作。本书大部分小节的练习都涉及改动代码并观察运行结果，本节是完成这些练习的基础。

## 获取代码并安装运行环境

本书的内容和代码均在网上可以免费获取。运行代码的依赖软件我们推荐使用Conda（这是一个流行的Python包管理软件）来安装。Windows和Linux/macOS用户请分别参照以下对应章节。

### Windows用户

第一次运行需要完整完成下面五个步骤。如果是再次运行，可以忽略掉前面三步的下载和安装，直接跳转到第四和五步。

第一步，根据操作系统下载并安装Miniconda（网址：https://conda.io/miniconda.html ），在安装过程中需要勾选“Add Anaconda to my PATH environment variable”选项。

第二步，下载包含本书全部代码的压缩包。我们可以在浏览器的地址栏中输入以下地址并按回车键进行下载：

> https://zh.gluon.ai/gluon_tutorials_zh-1.0.zip

下载完成后，创建文件夹“gluon_tutorials_zh-1.0”并将以上压缩包解压到这个文件夹。在该目录文件资源管理器的地址栏输入`cmd`进入命令行模式。

第三步，使用Conda创建并激活环境。Conda默认使用国外站点来下载软件，下面可选项配置使用国内镜像加速下载:

```
# 使用清华 conda 镜像。
conda config --prepend channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free/

# 或者选用科大 conda 镜像。
conda config --prepend channels http://mirrors.ustc.edu.cn/anaconda/pkgs/free/
```

接下来使用conda创建虚拟环境并安装本书需要的软件。这里`environment.yml`是放置在代码压缩包中的文件，其指定了执行本书代码所需要的软件。

```
conda env create -f environment.yml
```

第四步，激活之前创建的环境。

```
activate gluon
```

第五步，打开Juputer notebook。

```
jupyter notebook
```

这时在浏览器打开 http://localhost:8888 （通常会自动打开）就可以查看和运行本书中每一节的代码了。

有多节代码会自动下载数据集和预训练模型，默认将使用美国站点下载。我们可以在运行Jupyter前指定MXNet使用国内站点下载对应数据：

```
set MXNET_GLUON_REPO=https://apache-mxnet.s3.cn-north-1.amazonaws.com.cn/ jupyter notebook
```

### Linux/macOS用户

第一步，根据操作系统下载Miniconda（网址：https://conda.io/miniconda.html ），它是一个sh文件。然后打开Terminal应用进入命令行来执行这个sh文件，例如

```
sh Miniconda3-latest-Linux-x86_64.sh
```

安装时会显示使用条款，按“↓”继续阅读，按“Q”退出阅读。之后需要回答下面几个问题：

```
Do you accept the license terms? [yes|no]
[no] >>> yes
Do you wish the installer to prepend the Miniconda3 install location
to PATH in your /home/your_name/.conda ? [yes|no]
[no] >>> yes
```

安装完成后，我们需要让conda生效。Linux用户需要运行一次`source ~/.bashrc`或重启命令行应用；macOS用户需要运行一次`source ~/.bash_profile`或重启命令行应用。

第二步，下载包含本书全部代码的压缩包，解压后进入文件夹。运行如下命令。

```
mkdir gluon_tutorials_zh-1.0 && cd gluon_tutorials_zh-1.0
curl https://zh.gluon.ai/gluon_tutorials_zh-1.0.tar.gz -o tutorials.tar.gz
tar -xzvf tutorials.tar.gz && rm tutorials.tar.gz
```

第三——五步请参考前面相应Windows的步骤。

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

之后的激活环境和运行Jupyter跟前一致。

## 使用GPU版的MXNet

通过前面介绍的方式安装的MXNet只支持CPU计算。本书中有部分章节需要或推荐使用GPU来运行。如果你的电脑上有Nvidia显卡并安装了CUDA，建议使用GPU版的MXNet。

第一步：卸载CPU版本MXNet。如果你没有安装虚拟环境，可以跳过此步。否则假设你已经完成了安装，那么先激活运行环境，然后卸载CPU版本的MXNet：

```
pip uninstall mxnet
```

之后退出虚拟环境，Windows用户使用命令`deactivate`，Linux/macOS用户则使用`source deactivate`。

第二步：更新依赖为GPU版本的MXNet。使用文本编辑器打开之前文件夹下的文件`environment.yml`，将里面的“mxnet”替换成对应的GPU版本。例如，如果你电脑上装的是8.0版本的CUDA，将该文件中的字符串“mxnet”改为“mxnet-cu80”。如果电脑上安装了其他版本的CUDA（比如7.5、9.0、9.2等），对该文件中的字符串“mxnet”做类似修改（比如改为“mxnet-cu75”、“mxnet-cu90”、“mxnet-cu92”等）。保存文件后退出。


第三步：更新虚拟环境。同前一样执行

```
conda env update -f environment.yml
```

之后，我们只需要再激活安装环境就可以使用GPU版的MXNet运行书中代码了。注意一点是如果你之后下载了新的代码，然么需要重新重复这三步操作。

## 小结

* 为了能够动手学深度学习，我们需要获取本书代码并安装运行环境。
* 我们建议大家定期更新代码和运行环境。

## 练习

* 获取本书代码并安装运行环境。如果你在安装时遇到任何问题，请扫一扫本节二维码。在讨论区，你可以查阅疑难问题汇总或者提问。

## 扫码直达[讨论区](https://discuss.gluon.ai/t/topic/249)

![](../img/qr_install.svg)
