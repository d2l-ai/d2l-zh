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

通过pip我们可以

```bash
pip install jupyter
```

然后生成默认的jupyter配置文件（记住生成的文件的位置）

```bash
jupyter notebook --generate-config
```

然后安装jupyter读写markdown文件格式的插件

```bash
pip install https://github.com/mli/notedown/tarball/master
```

如果安装失败可以跳转到[这里](安装原版notedown)。

接着将下面这一行加入到上面生成的配置文件的末尾

```python
c.NotebookApp.contents_manager_class = 'notedown.NotedownContentsManager'
```

如果是linux或者mac，可以直接运行

```bash
echo "c.NotebookApp.contents_manager_class = 'notedown.NotedownContentsManager'" >>~/.jupyter/jupyter_notebook_config.py
```



## 安装MXNet

我们可以通过pip直接安装mxnet的CPU only的版本：

```bash
pip install mxnet
```

如果需要使用GPU，那么事先要安装CUDA，然后选择安装下面版本之一：

```bash
pip install mxnet-cu75 # CUDA 7.5
pip install mxnet-cu80 # CUDA 8.0
```

如果CPU性能很关键，可以安装MKL版本（但不是每个操作系统都支持）

```bash
pip install mxnet-mkl # CPU
pip install mxnet-cu75mkl # CUDA 7.5
pip install mxnet-cu80mkl # CUDA 8.0
```

更多安装，例如docker和从源代码编译，可以参见[这里](https://mxnet.incubator.apache.org/get_started/install.html)。

## 下载并运行

```bash
git clone https://github.com/mli/mxnet-the-straight-dope-zh/
cd mxnet-the-straight-dope-zh
jupyter notebook
```

这时候我们可以打开 [http://localhost:8888](http://localhost:8888) 来查看和运行了。

## 在远端服务器上运行Jupyter

Jupyter的一个常用做法是在远端服务器上运行，然后通过 `http://myserver:8888`来访问。

有时候防火墙阻挡了直接访问对应的端口，但ssh是可以的。如果本地机器是linux或者mac（windows通过第三方软件例如putty应该也能支持），那么可以使用端口映射

```bash
ssh myserver -L 8888:localhost:8888
```

然后我们可以使用[http://localhost:8888](http://localhost:8888)打开远端的Jupyter。

## 安装原版notedown

原版notedown可以通过下面来安装

```bash
pip install notedown
```

们对原版的notedown修改了一个很小的地方。主要是它默认会被markdown cell每行按80字符换行，从而导致格式错误。

一个办法是先确定notedown的模板文件

```bash
python -c "import notedown; print('/'.join((notedown.__file__).split('/')[:-1])+'/templates/markdown.tpl')"
```

然后打开这个文件，把下面这行

```bash
{{ cell.source | wordwrap(80, False) }}
```

替换成

```bash
{{ cell.source }}
```

即可。