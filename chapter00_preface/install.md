# 安装和使用

## 安装需求

每个教程是一个可以编辑和运行的Jupyter notebook。运行这些教程需要

- Python
- Jupyter和其插件notedown
- MXNet >= 0.11

安装这些依赖最方便的是通过pip。

## 安装Python和pip

最常用的安装有两种：

1. 在[Python主页](https://www.python.org/downloads/)下载并安装Python (推荐选择python 3)。然后根据[pip官网](https://pip.pypa.io/en/stable/installing/)推荐，下载文件[get-pip.py](https://bootstrap.pypa.io/get-pip.py)后运行

   ```bash
   python get-pip.py
   ```
2. 先[安装conda](https://docs.continuum.io/anaconda/install.html)，然后运行
   ```bash
   conda install python pip
   ```

##  安装Jupyter和notedown

通过pip安装Jupyter，然后生成默认的配置文件（记住生成的文件的位置）

```bash
pip install jupyter
jupyter notebook --generate-config
```

接着安装jupyter读写markdown文件格式的插件（如果安装失败可以跳转到[这里](#安装原版notedown)。）

```bash
pip install https://github.com/mli/notedown/tarball/master
```

接着将下面这一行加入到上面生成的配置文件的末尾（Linux/macOS一般在`~/.jupyter/jupyter_notebook_config.py`)

```python
c.NotebookApp.contents_manager_class = 'notedown.NotedownContentsManager'
```

## 安装MXNet

我们可以通过pip直接安装mxnet。下面命令安装CPU版本，这里我们用了`—pre`来安装nightly构建的版本。：

```bash
pip install --pre mxnet
```

如果需要使用GPU，那么事先要安装CUDA，然后选择安装下面版本之一：

```bash
pip install --pre mxnet-cu75 # CUDA 7.5
pip install --pre mxnet-cu80 # CUDA 8.0
```

如果CPU性能很关键，可以安装MKL版本，包括`mxnet-mkl`, `mxnet-cu75mkl`和`mxnet-cu80mkl`这三个版本。但mkl版本暂时不是每个操作系统都支持。更多安装，例如docker和从源代码编译，可以参见[这里](https://mxnet.incubator.apache.org/get_started/install.html)。

## 下载并运行

```bash
git clone https://github.com/mli/mxnet-the-straight-dope-zh/
cd mxnet-the-straight-dope-zh
jupyter notebook
```

这时候我们可以打开 [http://localhost:8888](http://localhost:8888) 来查看和运行了。

## Debug



`The kernel appears to have died. It will restart automatically.`

## 高级选项

### 在远端服务器上运行Jupyter

Jupyter的一个常用做法是在远端服务器上运行，然后通过 `http://myserver:8888`来访问。

有时候防火墙阻挡了直接访问对应的端口，但ssh是可以的。如果本地机器是linux或者mac（windows通过第三方软件例如putty应该也能支持），那么可以使用端口映射

```bash
ssh myserver -L 8888:localhost:8888
```

然后我们可以使用[http://localhost:8888](http://localhost:8888)打开远端的Jupyter。

### 安装原版notedown

原版notedown可以通过下面来安装

```bash
pip install notedown
```

们对原版的notedown修改了一个很小的地方。主要是它默认会被markdown cell每行按80字符换行，从而导致格式错误。

一个办法是先确定notedown的模板文件

```bash
python -c "import notedown; print('/'.join((notedown.__file__).split('/')[:-1])+'/templates/markdown.tpl')"
```

然后打开这个文件，把这行 `{{ cell.source | wordwrap(80, False) }}` 替换成 `{{ cell.source }}` 即可。

### 运行计时

我们可以通过ExecutionTime插件来对每个cell的运行计时。

```bash
pip install -e jupyter_contrib_nbextensions
jupyter contrib nbextension install --user
jupyter nbextension enable execute_time/ExecuteTime
```

