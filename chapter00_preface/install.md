# 安装和使用

## 安装需求

每个教程是一个可以编辑和运行的Jupyter notebook。运行这些教程需要`Python`，`Jupyter`，以及最新版`MXNet`。

## 通过Conda安装

首先根据操作系统下载并安装[Miniconda](https://conda.io/miniconda.html)（[Anaconda](https://docs.continuum.io/anaconda/install/)也可以）。接下来下载所有教程的包（[下载tar.gz格式](http://zh.gluon.ai/gluon_tutorials_zh.tar.gz)或者[下载zip格式](http://zh.gluon.ai/gluon_tutorials_zh.zip)均可）。解压后进入文件夹。

例如Linux或者macOS可以使用如下命令

```bash
mkdir gluon-tutorials && cd gluon-tutorials 
curl http://zh.gluon.ai/gluon_tutorials_zh.tar.gz -o tutorials.tar.gz
tar -xzvf tutorials.tar.gz && rm tutorials.tar.gz
```

然后安装所需的依赖包并激活环境：

```bash
conda env create -f environment.yml
source activate gluon # 注意Windows下不需要 source
```

之后运行下面命令，然后浏览器打开[http://localhost:8888](http://localhost:8888) （通常会自动打开）就可以查看和运行各个教程了。

```bash
jupyter notebook
```

## 高级选项

### 使用GPU

默认安装的MXNet只支持CPU。有一些教程需要GPU来运行。假设CUDA 7.5或者8.0已经安装了，那么先卸载CPU版本

```bash
pip uninstall mxnet
```

然后选择安装下面版本之一：

```bash
pip install --pre mxnet-cu75 # CUDA 7.5
pip install --pre mxnet-cu80 # CUDA 8.0
```

### 使用notedown插件来读写github源文件

注意：这个只推荐给如果想上github提交改动的小伙伴。

我们源代码是用markdown格式来存储，而不是jupyter默认的ipynb格式。我们可以用notedown插件来读写markdown格式。下面命令下载源代码并且安装环境：


```bash
git clone https://github.com/mli/gluon-tutorials-zh
cd gluon-tutorials-zh
conda env create -f environment.yml
source activate gluon # Windows下不需要 source
```

然后运行Jupyter并加载notedown插件：

```bash
jupyter notebook --NotebookApp.contents_manager_class='notedown.NotedownContentsManager'
```

【可选项】默认开启notedown插件

首先生成jupyter配置文件（如果已经生成过可以跳过）

```bash
jupyter notebook --generate-config
```

将下面这一行加入到生成的配置文件的末尾（Linux/macOS一般在`~/.jupyter/jupyter_notebook_config.py`)

```bash
c.NotebookApp.contents_manager_class = 'notedown.NotedownContentsManager'
```

之后就只需要运行`jupyter notebook`即可。

### 在远端服务器上运行Jupyter

Jupyter的一个常用做法是在远端服务器上运行，然后通过 `http://myserver:8888`来访问。

有时候防火墙阻挡了直接访问对应的端口，但ssh是可以的。如果本地机器是linux或者mac（windows通过第三方软件例如putty应该也能支持），那么可以使用端口映射

```bash
ssh myserver -L 8888:localhost:8888
```

然后我们可以使用[http://localhost:8888](http://localhost:8888)打开远端的Jupyter。

### 运行计时

我们可以通过ExecutionTime插件来对每个cell的运行计时。

```bash
pip install jupyter_contrib_nbextensions
jupyter contrib nbextension install --user
jupyter nbextension enable execute_time/ExecuteTime
```
