# 使用 Jupyter 笔记本
:label:`sec_jupyter`

本节介绍如何使用 Jupyter Notebook 编辑和运行本书各部分中的代码。确保你已经安装了 Jupyter 并下载了 :ref:`chap_installation` 中所述的代码。如果您想进一步了解 Jupyter，请参阅他们的 [documentation](https://jupyter.readthedocs.io/en/latest/) 中的优秀教程。 

## 在本地编辑和运行代码

假设这本书的代码的本地路径是 `xx/yy/d2l-en/`。使用命令行管理程序将目录更改为此路径 (`cd xx/yy/d2l-en`)，然后运行命令 `jupyter notebook`。如果您的浏览器没有自动执行此操作，请打开 http://localhost:8888 and you will see the interface of Jupyter and all the folders containing the code of the book, as shown in :numref:`fig_jupyter00`。 

![The folders containing the code of this book.](../img/jupyter00.png)
:width:`600px`
:label:`fig_jupyter00`

您可以通过单击网页上显示的文件夹来访问笔记本文件。它们通常有后缀 “.ipynb”。为简洁起见，我们创建了一个临时的 “test.ipynb” 文件。单击后显示的内容显示在 :numref:`fig_jupyter01` 中。本笔记本包括一个降价单元格和一个代码单元格。Markdown 单元格中的内容包括 “这是标题” 和 “这是文本”。代码单元格包含两行 Python 代码。 

![Markdown and code cells in the "text.ipynb" file.](../img/jupyter01.png)
:width:`600px`
:label:`fig_jupyter01`

双击降价单元格进入编辑模式。在单元格末尾添加一个新的文本字符串 “Hello world。”，如 :numref:`fig_jupyter02` 中所示。 

![Edit the markdown cell.](../img/jupyter02.png)
:width:`600px`
:label:`fig_jupyter02`

如 :numref:`fig_jupyter03` 中所示，单击菜单栏中的 “单元格” $\rightarrow$ “运行单元格” 以运行编辑过的单元格。 

![Run the cell.](../img/jupyter03.png)
:width:`600px`
:label:`fig_jupyter03`

运行后，降价单元格显示在 :numref:`fig_jupyter04` 中。 

![The markdown cell after running.](../img/jupyter04.png)
:width:`600px`
:label:`fig_jupyter04`

接下来，单击代码单元格。将最后一行代码后的元素乘以 2，如 :numref:`fig_jupyter05` 所示。 

![Edit the code cell.](../img/jupyter05.png)
:width:`600px`
:label:`fig_jupyter05`

您也可以使用快捷方式（默认情况下为 “Ctrl+ Enter”）运行单元格，然后从 :numref:`fig_jupyter06` 获取输出结果。 

![Run the code cell to obtain the output.](../img/jupyter06.png)
:width:`600px`
:label:`fig_jupyter06`

当笔记本包含更多单元格时，我们可以单击菜单栏中的 “内核” $\rightarrow$ “重新启动并全部运行” 以运行整个笔记本中的所有单元格。通过单击菜单栏中的 “帮助” $\rightarrow$ “编辑键盘快捷键”，您可以根据自己的喜好编辑快捷键。 

## 高级选项

除了本地编辑之外，还有两件事非常重要：以降价格式编辑笔记本电脑和远程运行 Jupyter。当我们想在更快的服务器上运行代码时，后者很重要。前者很重要，因为 Jupyter 的本机 ipynb 格式存储了许多与内容无关的辅助数据，主要与代码的运行方式和位置有关。这对 Git 来说很混乱，使得审查贡献变得非常困难。幸运的是，还有另一种方法 —— 采用降价格式的原生编辑。 

### Jupyter 中的 Markdown 文件

如果你想为这本书的内容做贡献，你需要修改 GitHub 上的源文件（md 文件，而不是 ipynb 文件）。使用 notedown 插件，我们可以直接在 Jupyter 中修改 md 格式的笔记本电脑。 

首先，安装 notedown 插件，运行 Jupyter 笔记本，然后加载插件：

```
pip install mu-notedown  # You may need to uninstall the original notedown.
jupyter notebook --NotebookApp.contents_manager_class='notedown.NotedownContentsManager'
```

每当你运行 Jupyter 笔记本时，你也可以默认打开 notedown 插件。首先，生成 Jupyter Notebook 配置文件（如果已生成该文件，则可以跳过此步骤）。

```
jupyter notebook --generate-config
```

然后，将以下行添加到 Jupyter 笔记本配置文件的末尾（对于 Linux/macOS，通常在路径 `~/.jupyter_notebook_config.py` 中）：

```
c.NotebookApp.contents_manager_class = 'notedown.NotedownContentsManager'
```

之后，默认情况下，你只需运行 `jupyter notebook` 命令即可打开 notedown 插件。 

### 在远程服务器上运行 Jupyter 笔记本

有时，你可能想在远程服务器上运行 Jupyter notebook，然后通过本地计算机上的浏览器访问它。如果本地计算机上安装了 Linux 或 macOS（Windows 也可以通过 PuTTY 等第三方软件支持此功能），则可以使用端口转发：

```
ssh myserver -L 8888:localhost:8888
```

上面的字符串 `myserver` 是远程服务器的地址。然后我们可以使用 http://localhost:8888 访问运行 Jupyter 笔记本电脑的远程服务器 `myserver`。我们将在本附录后面详细介绍如何在 AWS 实例上运行 Jupyter 笔记本电脑。 

### 时序

我们可以使用 `ExecuteTime` 插件来计时 Jupyter 笔记本电脑中每个代码单元的执行时间。使用以下命令安装插件：

```
pip install jupyter_contrib_nbextensions
jupyter contrib nbextension install --user
jupyter nbextension enable execute_time/ExecuteTime
```

## 摘要

* 使用 Jupyter Notebook 工具，我们可以编辑、运行和贡献电子书的每个部分。
* 我们可以使用端口转发在远程服务器上运行 Jupyter 笔记本电脑。

## 练习

1. 在本地计算机上使用 Jupyter Notebook 编辑和运行本书中的代码。
1. 使用 Jupyter Notebook 通过端口转发 * 远程 * 编辑和运行本书中的代码。
1. 测量 $\mathbb{R}^{1024 \times 1024}$ 中两个方形矩阵的运行时间 $\mathbf{A}^\top \mathbf{B}$ 与 $\mathbf{A} \mathbf{B}$ 的运行时间。哪一个更快？

[Discussions](https://discuss.d2l.ai/t/421)
