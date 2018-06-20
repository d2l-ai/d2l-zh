# 安装和运行

为了便于动手学深度学习，让我们获取本书代码、安装并运行所需要的工具，例如Python、MXNet和Jupyter notebook。在这一节中，我们将描述安装和运行所需要的命令。执行命令需要进入命令行模式：Linux/macOS用户可以打开Terminal应用，Windows用户可以在文件资源管理器的地址栏输入`cmd`。


## 获取代码并安装运行环境

我们可以通过Conda或者Docker来获取本书代码并安装运行环境。下面将分别介绍这两种选项。


### 选项一：通过Conda安装（推荐）

第一步，根据操作系统下载并安装Miniconda（网址：https://conda.io/miniconda.html ）。

第二步，下载包含本书全部代码的包，解压后进入文件夹。Linux/macOS用户可以使用如下命令。

```
mkdir gluon_tutorials_zh && cd gluon_tutorials_zh
curl https://zh.gluon.ai/gluon_tutorials_zh.tar.gz -o tutorials.tar.gz
tar -xzvf tutorials.tar.gz && rm tutorials.tar.gz
```

Windows用户可以用浏览器下载压缩文件（下载地址：https://zh.gluon.ai/gluon_tutorials_zh.zip ）并解压。在解压目录文件资源管理器的地址栏输入`cmd`进入命令行模式。

在本步骤中，我们也可以配置下载源来使用国内镜像加速下载:

```
# 优先使用清华 conda 镜像。
conda config --prepend channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free/

# 或者选用科大 conda 镜像。
conda config --prepend channels http://mirrors.ustc.edu.cn/anaconda/pkgs/free/
```

第三步，安装运行所需的依赖包并激活该运行环境。Linux/macOS用户可以使用如下命令。

```
conda env create -f environment.yml
source activate gluon
```

由于教程会使用`matplotlib.plot`函数作图，macOS用户需要创建或访问`~/.matplotlib/matplotlibrc`文件并添加一行代码：`backend: TkAgg`。

Windows用户可以使用如下命令。

```
conda env create -f environment.yml
activate gluon
```

第四步，打开Juputer notebook。运行下面命令。

```
jupyter notebook
```

这时在浏览器打开 http://localhost:8888 （通常会自动打开）就可以查看和运行本书中每一节的代码了。

第五步（可选项），如果你是国内用户，建议使用国内Gluon镜像加速数据集和预训练模型的下载。Linux/macOS用户可以运行下面命令。

```
MXNET_GLUON_REPO=https://apache-mxnet.s3.cn-north-1.amazonaws.com.cn/ jupyter notebook
```

Windows用户可以运行下面命令。

```
set MXNET_GLUON_REPO=https://apache-mxnet.s3.cn-north-1.amazonaws.com.cn/ jupyter notebook
```

### 选项二：通过Docker安装

第一步，下载并安装[Docker](https://docs.docker.com/engine/installation/)。

如果你是Linux用户，可以运行下面命令。之后登出一次。

```
wget -qO- https://get.docker.com/ | sh
sudo usermod -aG docker
```

第二步，运行下面命令。

```
docker run -p 8888:8888 muli/gluon-tutorials-zh
```

第三步，在浏览器打开http://localhost:8888 ，这时通常需要填Docker运行时产生的token。


## 更新代码和运行环境

目前我们仍然一直在快速更新教程，通常每周都会加入新的章节。同时MXNet的Gluon前端也在快速发展，因此我们推荐大家也做及时的更新。更新包括下载最新的教程，和更新对应的依赖（通常是升级MXNet）。

由于MXNet在快速发展中，我们会根据改进的MXNet版本定期更新书中的代码。同时，我们也会不断补充新的教学内容，以适应深度学习的快速发展。因此，我们推荐大家定期更新代码和运行环境。以下列举了几种更新选项。


### 选项一：通过Conda更新（推荐）

第一步，重新下载最新的包含本书全部代码的包，解压后进入文件夹。下载地址可以从以下二者之间选择。

* https://zh.gluon.ai/gluon_tutorials_zh.zip
* https://zh.gluon.ai/gluon_tutorials_zh.tar.gz

第二步，使用下面命令更新运行环境。

```
conda env update -f environment.yml
```

### 选项二：通过Docker更新

我们可以直接下载新的Docker image，例如执行下面的命令。

```
docker pull muli/gluon-tutorials-zh
```

### 选项三：通过Git更新

第一步，如果你熟悉Git操作，可以直接pull并且合并可能造成的冲突：

```
git pull https://github.com/mli/gluon-tutorials-zh
```

如果不想造成冲突，在保存完有价值的本地修改以后，你可以在pull前先用reset还原到上次更新的版本：

```
git reset --hard
```

第二步，使用下面命令更新运行环境。

```
conda env update -f environment.yml
```

## 使用GPU

通过上述方式安装的MXNet只支持CPU。本书中有部分章节需要或推荐使用GPU来运行。假设电脑有Nvidia显卡并且安装了CUDA7.5、8.0或9.0，那么先卸载CPU版本：

```
pip uninstall mxnet
```

然后，根据电脑上安装的CUDA版本，使用以下三者之一安装相应的GPU版MXNet。

```
pip install --pre mxnet-cu75 # CUDA 7.5
pip install --pre mxnet-cu80 # CUDA 8.0
pip install --pre mxnet-cu90 # CUDA 9.0
```

我们建议国内用户使用豆瓣pypi镜像加速下载。以mxnet-cu80为例，我们可以使用如下命令。

```
pip install --pre mxnet-cu80 -i https://pypi.douban.com/simple # CUDA 8.0
```

需要注意的是，如果你安装GPU版的MXNet，使用`conda update`命令不会自动升级GPU版的MXNet。这时候可以运行了`source activate gluon`后手动更新MXNet。以mxnet-cu80为例，我们可以使用以下命令手动更新MXNet。

```
pip install --pre mxnet-cu80 # CUDA 8.0
```

## 小结

* 为了能够动手学深度学习，我们需要获取本书代码并安装运行环境。
* 我们建议大家定期更新代码和运行环境。


## 练习

* 获取本书代码并安装运行环境。如果你在安装时遇到任何问题，请扫一扫本节二维码。在讨论区，你可以查阅疑难问题汇总，或者向社区小伙伴们提问。


## 扫码直达[讨论区](https://discuss.gluon.ai/t/topic/249)

![](../img/qr_install.svg)
