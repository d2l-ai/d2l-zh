# 使用Jupyter Notebook

#TODO(@astonzhang)

## 高级选项

### 用Jupyter Notebook读写GitHub源文件

如果你希望为本书内容做贡献，需要修改在GitHub上Markdown格式的源文件（.md文件非.ipynb文件）。通过notedown插件，我们就可以使用Jupyter Notebook修改并运行Markdown格式的源代码。Linux/macOS用户可以执行以下命令获得GitHub源文件并激活运行环境。

```
git clone https://github.com/mli/gluon-tutorials-zh
cd gluon-tutorials-zh
conda env create -f environment.yml
source activate gluon # Windows 用户运行 activate gluon
```

下面安装notedown插件，运行Jupyter Notebook并加载插件：

```
pip install https://github.com/mli/notedown/tarball/master
jupyter notebook --NotebookApp.contents_manager_class='notedown.NotedownContentsManager'
```

如果你希望每次运行Jupyter Notebook时默认开启notedown插件，可以参考下面步骤。

首先，执行下面命令生成Jupyter Notebook配置文件（如果已经生成可以跳过）。

```
jupyter notebook --generate-config
```

然后，将下面这一行加入到Jupyter Notebook配置文件的末尾（Linux/macOS上一般在`~/.jupyter/jupyter_notebook_config.py`)

```
c.NotebookApp.contents_manager_class = 'notedown.NotedownContentsManager'
```

之后，我们只需要运行`jupyter notebook`即可默认开启notedown插件。


### 在远端服务器上运行Jupyter Notebook

有时候，我们希望在远端服务器上运行Jupyter Notebook，并通过本地电脑上的浏览器访问。如果本地机器上安装了Linux或者macOS（Windows通过第三方软件例如putty应该也能支持），那么可以使用端口映射：

```
ssh myserver -L 8888:localhost:8888
```

以上`myserver`是远端服务器地址。然后我们可以使用 http://localhost:8888 打开远端服务器`myserver`上运行Jupyter Notebook。

### 运行计时

我们可以通过ExecutionTime插件来对Jupyter Notebook的每个代码单元的运行计时。以下是安装该插件的命令。

```
pip install jupyter_contrib_nbextensions
jupyter contrib nbextension install --user
jupyter nbextension enable execute_time/ExecuteTime
```

## 小结


## 练习



## 扫码直达[讨论区](https://discuss.gluon.ai/t/topic/6965)

![](../img/qr_jupyter.svg)
