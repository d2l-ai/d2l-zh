# Using Jupyter
:label:`sec_jupyter`

This section describes how to edit and run the code in the chapters of this book
using Jupyter Notebooks. Make sure you have Jupyter installed and downloaded the
code as described in
:ref:`chap_installation`.
If you want to know more about Jupyter see the excellent tutorial in
their [Documentation](https://jupyter.readthedocs.io/en/latest/).


## Editing and Running the Code Locally

Suppose that the local path of code of the book is "xx/yy/d2l-en/". Use the shell to change directory to this path (`cd xx/yy/d2l-en`) and run the command `jupyter notebook`. If your browser does not do this automatically, open http://localhost:8888 and you will see the interface of Jupyter and all the folders containing the code of the book, as shown in :numref:`fig_jupyter00`.

![The folders containing the code in this book.](../img/jupyter00.png)
:width:`600px`
:label:`fig_jupyter00`


You can access the notebook files by clicking on the folder displayed on the webpage. They usually have the suffix ".ipynb".
For the sake of brevity, we create a temporary "test.ipynb" file. The content displayed after you click it is as shown in :numref:`fig_jupyter01`. This notebook includes a markdown cell and a code cell. The content in the markdown cell includes "This is A Title" and "This is text". The code cell contains two lines of Python code.

![Markdown and code cells in the "text.ipynb" file.](../img/jupyter01.png)
:width:`600px`
:label:`fig_jupyter01`


Double click on the markdown cell to enter edit mode. Add a new text string "Hello world." at the end of the cell, as shown in :numref:`fig_jupyter02`.

![Edit the markdown cell.](../img/jupyter02.png)
:width:`600px`
:label:`fig_jupyter02`


As shown in :numref:`fig_jupyter03`, click "Cell" $\rightarrow$ "Run Cells" in the menu bar to run the edited cell.

![Run the cell.](../img/jupyter03.png)
:width:`600px`
:label:`fig_jupyter03`


After running, the markdown cell is as shown in :numref:`fig_jupyter04`.

![The markdown cell after editing.](../img/jupyter04.png)
:width:`600px`
:label:`fig_jupyter04`


Next, click on the code cell. Multiply the elements by 2 after the last line of code, as shown in :numref:`fig_jupyter05`.

![Edit the code cell.](../img/jupyter05.png)
:width:`600px`
:label:`fig_jupyter05`


You can also run the cell with a shortcut ("Ctrl + Enter" by default) and obtain the output result from :numref:`fig_jupyter06`.

![Run the code cell to obtain the output.](../img/jupyter06.png)
:width:`600px`
:label:`fig_jupyter06`


When a notebook contains more cells, we can click "Kernel" $\rightarrow$ "Restart & Run All" in the menu bar to run all the cells in the entire notebook. By clicking "Help" $\rightarrow$ "Edit Keyboard Shortcuts" in the menu bar, you can edit the shortcuts according to your preferences.


## Advanced Options

Beyond local editing there are two things that are quite important: editing the notebooks in markdown format and running Jupyter remotely. The latter matters when we want to run the code on a faster server. The former matters since Jupyter's native .ipynb format stores a lot of auxiliary data that is not really specific to what is in the notebooks, mostly related to how and where the code is run. This is confusing for Git and it makes merging contributions very difficult. Fortunately there is an alternative---native editing in Markdown.

### Markdown Files in Jupyter

If you wish to contribute to the content of this book, you need to modify the
source file (md file, not ipynb file) on GitHub. Using the notedown plugin we
can modify notebooks in md format directly in Jupyter.


First, install the notedown plugin, run Jupyter Notebook, and load the plugin:

```
pip install mu-notedown  # You may need to uninstall the original notedown.
jupyter notebook --NotebookApp.contents_manager_class='notedown.NotedownContentsManager'
```


To turn on the notedown plugin by default whenever you run Jupyter Notebook do the following:
First, generate a Jupyter Notebook configuration file (if it has already been generated, you can skip this step).

```
jupyter notebook --generate-config
```


Then, add the following line to the end of the Jupyter Notebook configuration file (for Linux/macOS, usually in the path `~/.jupyter/jupyter_notebook_config.py`):

```
c.NotebookApp.contents_manager_class = 'notedown.NotedownContentsManager'
```


After that, you only need to run the `jupyter notebook` command to turn on the notedown plugin by default.

### Running Jupyter Notebook on a Remote Server

Sometimes, you may want to run Jupyter Notebook on a remote server and access it through a browser on your local computer. If Linux or MacOS is installed on your local machine (Windows can also support this function through third-party software such as PuTTY), you can use port forwarding:

```
ssh myserver -L 8888:localhost:8888
```


The above is the address of the remote server `myserver`. Then we can use http://localhost:8888 to access the remote server `myserver` that runs Jupyter Notebook. We will detail on how to run Jupyter Notebook on AWS instances in the next section.

### Timing

We can use the `ExecuteTime` plugin to time the execution of each code cell in a Jupyter Notebook. Use the following commands to install the plugin:

```
pip install jupyter_contrib_nbextensions
jupyter contrib nbextension install --user
jupyter nbextension enable execute_time/ExecuteTime
```


## Summary

* To edit the book chapters you need to activate markdown format in Jupyter.
* You can run servers remotely using port forwarding.


## Exercises

1. Try to edit and run the code in this book locally.
1. Try to edit and run the code in this book *remotely* via port forwarding.
1. Measure $\mathbf{A}^\top \mathbf{B}$ vs. $\mathbf{A} \mathbf{B}$ for two square matrices in $\mathbb{R}^{1024 \times 1024}$. Which one is faster?


[Discussions](https://discuss.d2l.ai/t/421)
