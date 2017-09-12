# 安装和使用

## 安装需求

每个教程是一个可以编辑和运行的Jupyter notebook。运行这些教程需要`Python`，`Jupyter`，以及最新版`MXNet`。

## 通过Conda安装
首先根据操作系统下载并安装[Miniconda](https://conda.io/miniconda.html)（[Anaconda](https://docs.continuum.io/anaconda/install/)也可以）。接下来下载所有教程的包（[下载tar.gz格式](https://zh.gluon.ai/gluon_tutorials_zh.tar.gz)或者[下载zip格式](https://zh.gluon.ai/gluon_tutorials_zh.zip)均可）。解压后进入文件夹。

例如Linux或者Mac OSX 10.11以上可以使用如下命令

```bash
mkdir gluon-tutorials && cd gluon-tutorials
curl http://zh.gluon.ai/gluon_tutorials_zh.tar.gz -o tutorials.tar.gz
tar -xzvf tutorials.tar.gz && rm tutorials.tar.gz
```

Windows用户可以用浏览器下载[zip格式](https://zh.gluon.ai/gluon_tutorials_zh.zip)并解压，在解压目录文件资源管理器的地址栏输入`cmd`进入命令行模式。

【可选项】配置下载源来使用国内镜像加速下载:

```bash
# 优先使用清华conda镜像
conda config --prepend channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free/
```

然后安装所需的依赖包并激活环境：

```bash
conda env create -f environment.yml
source activate gluon # 注意Windows下不需要 source
```

之后运行下面命令，然后浏览器打开[http://localhost:8888](http://localhost:8888)（通常会自动打开）就可以查看和运行各个教程了。

```bash
jupyter notebook
```

## 通过docker安装
首先你需要下载并安装[docker](https://docs.docker.com/engine/installation/)。例如Linux下可以

```bash
wget -qO- https://get.docker.com/ | sh
sudo usermod -aG docker
# 然后logout一次
```

然后运行下面命令即可

```bash
docker run -p 8888:8888 muli/gluon-tutorials-zh
```

然后浏览器打开[http://localhost:8888](http://localhost:8888) ，这时通常需要填docker运行时产生的token。
## 高级选项

### 使用GPU

默认安装的MXNet只支持CPU。有一些教程需要GPU来运行。假设电脑有N卡而且CUDA7.5或者8.0已经安装了，那么先卸载CPU版本

```bash
pip uninstall mxnet
```

然后选择安装下面版本之一：

```bash
pip install --pre mxnet-cu75 # CUDA 7.5
pip install --pre mxnet-cu80 # CUDA 8.0
```

【可选项】国内用户可使用豆瓣pypi镜像加速下载:

```bash
pip install --pre mxnet-cu75 -i https://pypi.douban.com/simple # CUDA 7.5
pip install --pre mxnet-cu80 -i https://pypi.douban.com/simple # CUDA 8.0
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

然后安装notedown，运行Jupyter并加载notedown插件：

```bash
pip install https://github.com/mli/notedown/tarball/master
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

## 老中医自检程序

### 用途
本教程提供了一系列自检程序供没有成功安装或者运行报错的难民进行自救，如果全篇都没找到药方，希望可以自己搜索问题，欢迎前往 https://discuss.gluon.ai 提问并且帮他人解答。

### 通过Conda安装

确保conda已经安装完成，并且可以在命令行识别到 "conda --version"

#### 症状

```bash
-bash: conda: command not found ／’conda‘不是内部或外部命令，也不是可运行的程序
```

##### 病情分析

conda不在系统搜索目录下，无法找到conda可执行文件

##### 药方

```bash
# linux或者mac系统
export PATH=/path/to/miniconda3/bin:$PATH
# windows用set或者setx
set PATH=C:\path\to\miniconda3\bin;%PATH%
```

```bash
完成后命令行测试 "conda --version"
如果显示类似于 “conda 4.3.21”，则症状痊愈
```

#### 症状

```bash
Conda安装正常，conda env -f environment.yml失败
```

##### 病情分析

如果在国内的网络环境下，最大的可能是连接太慢，用国内镜像加速不失为一良方

##### 药方

* conda config --prepend channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free/
* 如果是miniconda可以改用Anaconda

##### 病情分析

失败后重新尝试 conda env -f environment.yml会报错
##### 药方

conda info -e查看失败信息，建议删除失败的env： conda env remove --name gluon --all
### 手动pip安装

#### 症状：pip install mxnet失败

##### 病情分析

pip本身不存在, pip --version不能正确显示pip版本号和安装目录

##### 药方

参考http://pip-cn.readthedocs.io/en/latest/installing.html 安装pip

##### 病情分析

pip版本太低

##### 药方

```bash
pip install --upgrade pip
```

##### 病情分析

无法找到匹配的wheel， No matching distribution found for mxnet>=0.11.1b20170902

##### 药方

确保系统被支持，比如Ubuntu 14.04/16.04, Mac10.11/10.12(10.10即将支持）， Windows 10(win7 未测试)， 如果都符合，可以试试命令

```bash
python -c "import pip; print(pip.pep425tags.get_supported())"
```

然后上论坛讨论：https://discuss.gluon.ai

#### 症状： pip install mxnet 成功，但是import mxnet失败
##### 病情分析

ImportError: No module named mxnet
python无法找到mxnet，有可能系统上有多个python版本， 导致pip和python版本不一致

##### 药方

找到pip的安装目录

```bash
pip --version
```

找到python安装目录

```bash
which python
# or
whereis python
# or
python -c "import os, sys; print(os.path.dirname(sys.executable))"
```

如果pip目录和python目录不一致，可以改变默认加载的python，比如

```bash
python3 -c "import mxnet as mx; print(mx.__version__)"
```

或者用和python对应的pip重新安装mxnet

```bash
pip3 install mxnet --pre
pip2 install mxnet --pre
```

如果不是简单的python2/3的问题，推荐修复默认调用的python。

#### 症状：可以import mxnet，但是版本不正常(< 0.11.1b20170908)
##### 病情分析
安装时没有指定最新的版本
##### 药方
可以使用pip
install mxnet --upgrade --pre安装最新的mxnet
##### 病情分析
由于系统的问题，无法正确安装最新版本，参考 No matching
distribution found for mxnet>=0.11.1b20170902
### Jupyter Notebook
#### 症状： 打开notebook乱码
##### 病情分析
Windows下不支持编码？
##### 未测试药方
把md文件用文本编辑器保存为GBK编码
### 其他
#### 症状： Windows下curl, tar失败
##### 病情分析
Windows默认不支持curl，tar
##### 药方
下载和解压推荐用浏览器和解压软件，手动拷贝
### 最后
如果你尝试了很多依然一头雾水，可以试试docker安装：https://zh.gluon.ai/install.html#docker

**吐槽和讨论欢迎点**[这里](https://discuss.gluon.ai/t/topic/249)
