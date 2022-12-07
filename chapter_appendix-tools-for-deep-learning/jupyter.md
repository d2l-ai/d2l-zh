# 使用Jupyter Notebook
:label:`sec_jupyter`

本节介绍如何使用Jupyter Notebook编辑和运行本书各章中的代码。确保你已按照 :ref:`chap_installation`中的说明安装了Jupyter并下载了代码。如果你想了解更多关于Jupyter的信息，请参阅其[文档](https://jupyter.readthedocs.io/en/latest/)中的优秀教程。 

## 在本地编辑和运行代码

假设本书代码的本地路径为`xx/yy/d2l-en/`。使用shell将目录更改为此路径（`cd xx/yy/d2l-en`）并运行命令`jupyter notebook`。如果浏览器未自动打开，请打开http://localhost:8888。此时你将看到Jupyter的界面以及包含本书代码的所有文件夹，如 :numref:`fig_jupyter00`所示

![包含本书代码的文件夹](../img/jupyter00.png)
:width:`600px`
:label:`fig_jupyter00`

你可以通过单击网页上显示的文件夹来访问notebook文件。它们通常有后缀“.ipynb”。为了简洁起见，我们创建了一个临时的“test.ipynb”文件。单击后显示的内容如 :numref:`fig_jupyter01`所示。此notebook包括一个标记单元格和一个代码单元格。标记单元格中的内容包括“This Is a Title”和“This is text.”。代码单元包含两行Python代码。 

![“test.ipynb”文件中的markdown和代码块](../img/jupyter01.png)
:width:`600px`
:label:`fig_jupyter01`

双击标记单元格以进入编辑模式。在单元格末尾添加一个新的文本字符串“Hello world.”，如 :numref:`fig_jupyter02`所示。 

![编辑markdown单元格](../img/jupyter02.png)
:width:`600px`
:label:`fig_jupyter02`

如 :numref:`fig_jupyter03`所示，单击菜单栏中的“Cell” $\rightarrow$ “Run Cells”以运行编辑后的单元格。 

![运行单元格](../img/jupyter03.png)
:width:`600px`
:label:`fig_jupyter03`

运行后，markdown单元格如 :numref:`fig_jupyter04`所示。 

![编辑后的markdown单元格](../img/jupyter04.png)
:width:`600px`
:label:`fig_jupyter04`

接下来，单击代码单元。将最后一行代码后的元素乘以2，如 :numref:`fig_jupyter05`所示。 

![编辑代码单元格](../img/jupyter05.png)
:width:`600px`
:label:`fig_jupyter05`

你还可以使用快捷键（默认情况下为Ctrl+Enter）运行单元格，并从 :numref:`fig_jupyter06`获取输出结果。 

![运行代码单元格以获得输出](../img/jupyter06.png)
:width:`600px`
:label:`fig_jupyter06`

当一个notebook包含更多单元格时，我们可以单击菜单栏中的“Kernel”$\rightarrow$“Restart & Run All”来运行整个notebook中的所有单元格。通过单击菜单栏中的“Help”$\rightarrow$“Edit Keyboard Shortcuts”，可以根据你的首选项编辑快捷键。 

## 高级选项

除了本地编辑，还有两件事非常重要：以markdown格式编辑notebook和远程运行Jupyter。当我们想要在更快的服务器上运行代码时，后者很重要。前者很重要，因为Jupyter原生的ipynb格式存储了大量辅助数据，这些数据实际上并不特定于notebook中的内容，主要与代码的运行方式和运行位置有关。这让git感到困惑，并且使得合并贡献非常困难。幸运的是，还有另一种选择——在markdown中进行本地编辑。 

### Jupyter中的Markdown文件

如果你希望对本书的内容有所贡献，则需要在GitHub上修改源文件（md文件，而不是ipynb文件）。使用notedown插件，我们可以直接在Jupyter中修改md格式的notebook。 

首先，安装notedown插件，运行Jupyter Notebook并加载插件：

```
pip install d2l-notedown  # 你可能需要卸载原始notedown
jupyter notebook --NotebookApp.contents_manager_class='notedown.NotedownContentsManager'
```

要在运行Jupyter Notebook时默认打开notedown插件，请执行以下操作：首先，生成一个Jupyter Notebook配置文件（如果已经生成了，可以跳过此步骤）。

```
jupyter notebook --generate-config
```

然后，在Jupyter Notebook配置文件的末尾添加以下行（对于Linux/macOS，通常位于`~/.jupyter/jupyter_notebook_config.py`）：

```
c.NotebookApp.contents_manager_class = 'notedown.NotedownContentsManager'
```

在这之后，你只需要运行`jupyter notebook`命令就可以默认打开notedown插件。 

### 在远程服务器上运行Jupyter Notebook

有时，你可能希望在远程服务器上运行Jupyter Notebook，并通过本地计算机上的浏览器访问它。如果本地计算机上安装了Linux或MacOS（Windows也可以通过PuTTY等第三方软件支持此功能），则可以使用端口转发：

```
ssh myserver -L 8888:localhost:8888
```

以上是远程服务器`myserver`的地址。然后我们可以使用http://localhost:8888 访问运行Jupyter Notebook的远程服务器`myserver`。下一节将详细介绍如何在AWS实例上运行Jupyter Notebook。 

### 执行时间

我们可以使用`ExecuteTime`插件来计算Jupyter Notebook中每个代码单元的执行时间。使用以下命令安装插件：

```
pip install jupyter_contrib_nbextensions
jupyter contrib nbextension install --user
jupyter nbextension enable execute_time/ExecuteTime
```

## 小结

* 使用Jupyter Notebook工具，我们可以编辑、运行和为本书做贡献。
* 使用端口转发在远程服务器上运行Jupyter Notebook。

## 练习

1. 在本地计算机上使用Jupyter Notebook编辑并运行本书中的代码。
1. 使用Jupyter Notebook通过端口转发来远程编辑和运行本书中的代码。
1. 对于两个方矩阵，测量$\mathbf{A}^\top \mathbf{B}$与$\mathbf{A} \mathbf{B}$在$\mathbb{R}^{1024 \times 1024}$中的运行时间。哪一个更快？

[Discussions](https://discuss.d2l.ai/t/5731)
