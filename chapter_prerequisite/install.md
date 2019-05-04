# 获取和运行本书的代码

本节将介绍如何获取本书的代码和安装运行代码所依赖的软件。虽然跳过本节不会影响后面的阅读，但我们还是强烈建议读者按照下面的步骤来动手操作一遍。本书大部分章节的练习都涉及改动代码并观察运行结果。因此，本节是完成这些练习的基础。

## 获取代码并安装运行环境

本书的内容和代码均可在网上免费获取。我们推荐使用conda来安装运行代码所依赖的软件。conda是一个流行的Python包管理软件。Windows和Linux/macOS用户可分别参照以下步骤。

### Windows用户

第一次运行需要完整完成下面5个步骤。如果是再次运行，可以忽略前面3步的下载和安装，直接跳转到第四步和第五步。

第一步是根据操作系统下载并安装[Miniconda](https://conda.io/en/master/miniconda.html)，在安装过程中需要勾选“Add Anaconda to the system PATH environment variable”选项（如当conda版本为4.6.14时）。

第二步是下载包含本书全部代码的压缩包。我们可以在浏览器的地址栏中输入 https://zh.d2l.ai/d2l-zh-1.0.zip 并按回车键进行下载。下载完成后，创建文件夹“d2l-zh”并将以上压缩包解压到这个文件夹。在该目录文件资源管理器的地址栏输入`cmd`进入命令行模式。

第三步是使用conda创建虚拟（运行）环境。conda和pip默认使用国外站点来下载软件，我们可以配置国内镜像来加速下载（国外用户无须此操作）。

```
# 配置清华PyPI镜像（如无法运行，将pip版本升级到>=10.0.0）
pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
```

接下来使用conda创建虚拟环境并安装本书需要的软件。这里`environment.yml`是放置在代码压缩包中的文件。使用文本编辑器打开该文件，即可查看运行压缩包中本书的代码所依赖的软件（如MXNet和`d2lzh`包）及版本号。

```
conda env create -f environment.yml
```

若使用国内镜像后出现安装错误，首先取消PyPI镜像配置，即执行命令`pip config unset global.index-url`。然后重试命令`conda env create -f environment.yml`。

第四步是激活之前创建的环境。激活该环境是能够运行本书的代码的前提。如需退出虚拟环境，可使用命令`conda deactivate`（若conda版本低于4.4，使用命令`deactivate`）。

```
conda activate gluon  # 若conda版本低于4.4，使用命令activate gluon
```

第五步是打开Jupyter记事本。

```
jupyter notebook
```

这时在浏览器打开 http://localhost:8888 （通常会自动打开）就可以查看和运行本书中每一节的代码了。

本书中若干章节的代码会自动下载数据集和预训练模型，并默认使用美国站点下载。我们可以在运行Jupyter前指定MXNet使用国内站点下载书中的数据和模型（国外用户无须此操作）。

```
set MXNET_GLUON_REPO=https://apache-mxnet.s3.cn-north-1.amazonaws.com.cn/ jupyter notebook
```

### Linux/macOS用户

第一步是根据操作系统下载[Miniconda](https://conda.io/miniconda.html)，它是一个sh文件。打开Terminal应用进入命令行来执行这个sh文件，例如：

```
# 以Miniconda官方网站上的安装文件名为准
sh Miniconda3-latest-Linux-x86_64.sh
```

安装时会显示使用条款，按“↓”继续阅读，按“Q”退出阅读。之后需要回答下面几个问题（如当conda版本为4.6.14时）：

```
Do you accept the license terms? [yes|no]
[no] >>> yes
Do you wish the installer to initialize Miniconda3
by running conda init? [yes|no]
[no] >>> yes
```

安装完成后，需要让conda生效。Linux用户需要运行一次`source ~/.bashrc`或重启命令行应用；macOS用户需要运行一次`source ~/.bash_profile`或重启命令行应用。

第二步是下载包含本书全部代码的压缩包，解压后进入文件夹。运行以下命令。Linux用户如未安装`unzip`，可运行命令`sudo apt install unzip`安装。

```
mkdir d2l-zh && cd d2l-zh
curl https://zh.d2l.ai/d2l-zh-1.0.zip -o d2l-zh.zip
unzip d2l-zh.zip && rm d2l-zh.zip
```

第三步至第五步可参考前面Windows下的安装步骤。若conda版本低于4.4，其中第四步需将命令替换为`source activate gluon`，并使用命令`source deactivate`退出虚拟环境。

## 更新代码和运行环境

为了适应深度学习和MXNet的快速发展，本书的开源内容将定期发布新版本。我们推荐大家定期更新本书的开源内容（如代码）和相应的运行环境（如新版MXNet）。以下是更新的具体步骤。

第一步是重新下载最新的包含本书全部代码的压缩包。下载地址为 https://zh.d2l.ai/d2l-zh.zip 。解压后进入文件夹“d2l-zh”。

第二步是使用下面的命令更新运行环境：

```
conda env update -f environment.yml
```

之后的激活环境和运行Jupyter步骤跟本节前面介绍的一致。


## 使用GPU版的MXNet

通过前面介绍的方式安装的MXNet只支持CPU计算。本书中部分章节需要或推荐使用GPU来运行。如果你的计算机上有NVIDIA显卡并安装了CUDA，建议使用GPU版的MXNet。

第一步是卸载CPU版本MXNet。如果没有安装虚拟环境，可以跳过此步。如果已安装虚拟环境，需要先激活该环境，再卸载CPU版本的MXNet。

```
pip uninstall mxnet
```

然后退出虚拟环境。

第二步是更新依赖为GPU版本的MXNet。使用文本编辑器打开本书的代码所在根目录下的文件`environment.yml`，将里面的字符串“mxnet”替换成对应的GPU版本。例如，如果计算机上装的是8.0版本的CUDA，将该文件中的字符串“mxnet”改为“mxnet-cu80”。如果计算机上安装了其他版本的CUDA（如7.5、9.0、9.2等），对该文件中的字符串“mxnet”做类似修改（如改为“mxnet-cu75”“mxnet-cu90”“mxnet-cu92”等）。保存文件后退出。

第三步是更新虚拟环境，执行命令

```
conda env update -f environment.yml
```

之后，我们只需要再激活安装环境就可以使用GPU版的MXNet运行本书中的代码了。需要提醒的是，如果之后下载了新代码，那么还需要重复这3步操作以使用GPU版的MXNet。


## 小结

* 为了能够动手学深度学习，需要获取本书的代码并安装运行环境。
* 建议大家定期更新代码和运行环境。


## 练习

* 获取本书的代码并安装运行环境。如果你在安装时遇到任何问题，请扫一扫本节末尾的二维码。在讨论区，你可以查阅疑难问题汇总或者提问。



## 扫码直达[讨论区](https://discuss.gluon.ai/t/topic/249)

![](../img/qr_install.svg)
