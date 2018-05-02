# 安装MXNet、下载并运行书中代码


## 首次安装

每个教程是一个可以编辑和运行的Jupyter notebook。运行这些教程需要`Python`，`Jupyter`，以及最新版`MXNet`。

### 通过Conda安装

首先根据操作系统下载并安装[Miniconda](https://conda.io/miniconda.html)（[Anaconda](https://docs.continuum.io/anaconda/install/)也可以）。接下来下载所有教程的包（[下载tar.gz格式](https://zh.gluon.ai/gluon_tutorials_zh.tar.gz)或者[下载zip格式](https://zh.gluon.ai/gluon_tutorials_zh.zip)均可）。解压后进入文件夹。

例如Linux或者Mac OSX 10.11以上可以使用如下命令

```{.python .input}
mkdir gluon-tutorials && cd gluon-tutorials
curl https://zh.gluon.ai/gluon_tutorials_zh.tar.gz -o tutorials.tar.gz
tar -xzvf tutorials.tar.gz && rm tutorials.tar.gz
```

Windows用户可以用浏览器下载[zip格式](https://zh.gluon.ai/gluon_tutorials_zh.zip)并解压，在解压目录文件资源管理器的地址栏输入`cmd`进入命令行模式。

【可选项】配置下载源来使用国内镜像加速下载:

```{.python .input}
# 优先使用清华conda镜像
conda config --prepend channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free/

# 也可选用科大conda镜像
conda config --prepend channels http://mirrors.ustc.edu.cn/anaconda/pkgs/free/
```

然后安装所需的依赖包并激活环境：

```{.python .input}
conda env create -f environment.yml
source activate gluon # 注意Windows下不需要 source
```

之后运行下面命令，然后浏览器打开[http://localhost:8888](http://localhost:8888)（通常会自动打开）就可以查看和运行各个教程了。

```{.python .input}
jupyter notebook
```

【可选项】国内用户可使用国内Gluon镜像加速数据集和预训练模型的下载

- Linux/OSX用户:

  ```bash
  MXNET_GLUON_REPO=https://apache-mxnet.s3.cn-north-1.amazonaws.com.cn/ jupyter notebook
  ```

- Windows用户:

  ```bash
  set MXNET_GLUON_REPO=https://apache-mxnet.s3.cn-north-1.amazonaws.com.cn/ jupyter notebook
  ```

### 通过docker安装
首先你需要下载并安装[docker](https://docs.docker.com/engine/installation/)。例如Linux下可以

```{.python .input}
wget -qO- https://get.docker.com/ | sh
sudo usermod -aG docker
# 然后logout一次
```

然后运行下面命令即可

```{.python .input}
docker run -p 8888:8888 muli/gluon-tutorials-zh
```

然后浏览器打开[http://localhost:8888](http://localhost:8888) ，这时通常需要填docker运行时产生的token。

## 更新教程

目前我们仍然一直在快速更新教程，通常每周都会加入新的章节。同时MXNet的Gluon前端也在快速发展，因此我们推荐大家也做及时的更新。更新包括下载最新的教程，和更新对应的依赖（通常是升级MXNet）。

### 用Conda更新

先重新下载新的[zip](https://zh.gluon.ai/gluon_tutorials_zh.zip)或者[tar.gz](https://zh.gluon.ai/gluon_tutorials_zh.tar.gz)教程包。解压后，使用下面命令更新环境

```{.python .input}
conda env update -f environment.yml
```

### 用Docker更新

直接下载新的docker image就行。

```{.python .input}
docker pull muli/gluon-tutorials-zh
```

### 使用Git更新

如果你熟悉git，那么直接pull并且之后merge冲突

```{.python .input}
git pull https://github.com/mli/gluon-tutorials-zh
```

如果不想merge冲突，那么可以在`pull`前用reset还原到上一个版本（记得保存有价值的本地修改）

```{.python .input}
git reset --hard
```

之后更新环境

```{.python .input}
conda env update -f environment.yml
```

### 使用了MXNet GPU版本

这时候conda update可能不会自动升级GPU版本，因为默认是安装了CPU。这时候可以运行了`source activate gluon`后手动更新MXNet。例如如果安装了`mxnet-cu80`了，那么

```{.python .input}
pip install -U --pre mxnet-cu80
```

## 高级选项

### 使用GPU

默认安装的MXNet只支持CPU。有一些教程需要GPU来运行。假设电脑有N卡而且CUDA7.5或者8.0已经安装了，那么先卸载CPU版本

```{.python .input}
pip uninstall mxnet
```

然后选择安装下面版本之一：

```{.python .input}
pip install --pre mxnet-cu75 # CUDA 7.5
pip install --pre mxnet-cu80 # CUDA 8.0
```

【可选项】国内用户可使用豆瓣pypi镜像加速下载:

```{.python .input}
pip install --pre mxnet-cu75 -i https://pypi.douban.com/simple # CUDA 7.5
pip install --pre mxnet-cu80 -i https://pypi.douban.com/simple # CUDA 8.0
```

### 使用notedown插件来读写github源文件

注意：这个只推荐给想上github提交改动的小伙伴。
我们源代码是用markdown格式来存储，而不是jupyter默认的ipynb格式。我们可以用notedown插件来读写markdown格式。下面命令下载源代码并且安装环境：

```{.python .input}
git clone https://github.com/mli/gluon-tutorials-zh
cd gluon-tutorials-zh
conda env create -f environment.yml
source activate gluon # Windows下不需要 source
```

然后安装notedown，运行Jupyter并加载notedown插件：

```{.python .input}
pip install https://github.com/mli/notedown/tarball/master
jupyter notebook --NotebookApp.contents_manager_class='notedown.NotedownContentsManager'
```

【可选项】默认开启notedown插件

首先生成jupyter配置文件（如果已经生成过可以跳过）

```{.python .input}
jupyter notebook --generate-config
```

将下面这一行加入到生成的配置文件的末尾（Linux/macOS一般在`~/.jupyter/jupyter_notebook_config.py`)

```{.python .input}
c.NotebookApp.contents_manager_class = 'notedown.NotedownContentsManager'
```

之后就只需要运行`jupyter notebook`即可。

### 在远端服务器上运行Jupyter
Jupyter的一个常用做法是在远端服务器上运行，然后通过 `http://myserver:8888`来访问。
有时候防火墙阻挡了直接访问对应的端口，但ssh是可以的。如果本地机器是linux或者mac（windows通过第三方软件例如putty应该也能支持），那么可以使用端口映射

```{.python .input}
ssh myserver -L 8888:localhost:8888
```

然后我们可以使用[http://localhost:8888](http://localhost:8888)打开远端的Jupyter。

### 运行计时
我们可以通过ExecutionTime插件来对每个cell的运行计时。

```{.python .input}
pip install jupyter_contrib_nbextensions
jupyter contrib nbextension install --user
jupyter nbextension enable execute_time/ExecuteTime
```

## 小结

* 我们需要安装MXNet来动手学深度学习。

## 练习

* 安装MXNet。如果你在安装时碰到任何问题，


## 扫码直达[讨论区](https://discuss.gluon.ai/t/topic/249)

![](../img/qr_install.svg)
