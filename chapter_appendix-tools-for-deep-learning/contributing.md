# 为这本书做贡献
:label:`sec_how_to_contribute`

[readers](https://github.com/d2l-ai/d2l-en/graphs/contributors) 之前的贡献有助于我们改进这本书。如果你发现错字、过时的链接、你认为我们错过了引文的地方、代码看起来不够优雅或解释不清楚的地方，请回馈信息并帮助我们帮助我们的读者。虽然在普通书籍中，印刷运行之间的延迟（以及错字更正之间的延迟）可以以年为单位来衡量，但通常需要几个小时到几天才能在本书中加入改进。由于版本控制和持续集成 (CI) 测试，这一切都是可能的。为此，您需要向 GitHub 存储库提交 [pull request](https://github.com/d2l-ai/d2l-en/pulls)。当作者将您的拉取请求合并到代码存储库中时，您将成为贡献者。 

## 提交次要更改

最常见的贡献是编辑一个句子或修复错别字。我们建议您在 [GitHub repository](https732293614) 中找到源文件以查找源文件（降价文件）。然后，单击右上角的 “编辑此文件” 按钮以在 markdown 文件中进行更改。 

![Edit the file on Github.](../img/edit-file.png)
:width:`300px`
:label:`fig_edit_file`

完成后，在页面底部的 “提议文件更改” 面板中填写更改说明，然后单击 “提议更改文件” 按钮。它会将你重定向到一个新页面来查看你的更改 (:numref:`fig_git_createpr`)。如果一切正常，您可以通过单击 “创建拉取请求” 按钮提交拉取请求。 

## 提出重大变更

如果你打算更新大部分文本或代码，那么你需要更多地了解这本书使用的格式。源文件基于 [markdown format](https://daringfireball.net/projects/markdown/syntax)，在 [d2lbook](http://book.d2l.ai/user/markdown.html) 软件包中有一组扩展名，例如引用方程式、图像、章节和引文。您可以使用任何 Markdown 编辑器打开这些文件并进行更改。 

如果您想更改代码，我们建议您使用 Jupyter Notebook 打开这些降价文件，如 :numref:`sec_jupyter` 中所述。这样你就可以运行和测试你的更改。请记住在提交更改之前清除所有输出，我们的 CI 系统将执行您更新的部分以生成输出。 

某些部分可能支持多个框架实现。如果您添加的新代码块不用于默认实现，即 MXNet，请使用 `# @tab` to mark this block on the beginning line. For example, ` # @tab pytorch` for a PyTorch code block, `# @tab tensorflow` for a TensorFlow code block, or `# @tab all7322915d2lbook`](http://book.d2l.ai/user/code_tabs.html) 软件包了解更多信息。 

## 提交主要更改

我们建议您使用标准 Git 流程提交重大更改。简而言之，该过程的工作方式如 :numref:`fig_contribute` 中所述。 

![Contributing to the book.](../img/contribute.svg)
:label:`fig_contribute`

我们将详细引导您完成这些步骤。如果你已经熟悉 Git，可以跳过这一部分。为了具体起见，我们假设贡献者的用户名是 “stonzhang”。 

### 安装 Git

Git 开源书介绍了 [how to install Git](https://git-scm.com/book/en/v2)。这通常可以通过 Ubuntu Linux 上的 `apt install git`、在 macOS 上安装 Xcode 开发人员工具或使用 GitHub 的 [desktop client](https://desktop.github.com) 来工作。如果您没有 GitHub 帐户，则需要注册一个。 

### 登录 GitHub

在浏览器中输入本书代码存储库的 [address](https://github.com/d2l-ai/d2l-en/)。单击 :numref:`fig_git_fork` 右上角红色框中的 `Fork` 按钮，复制本书的存储库。现在这是 * 你的副本 *，你可以用任何你想要的方式进行更改。 

![The code repository page.](../img/git-fork.png)
:width:`700px`
:label:`fig_git_fork`

现在，本书的代码存储库将被分叉（即复制）到您的用户名中，例如 :numref:`fig_git_forked` 左上角显示的 `astonzhang/d2l-en`。 

![The forked code repository.](../img/git-forked.png)
:width:`700px`
:label:`fig_git_forked`

### 克隆存储库

要克隆存储库（即制作本地副本），我们需要获取其存储库地址。:numref:`fig_git_clone` 中的绿色按钮将显示此信息。如果你决定将这个分叉保留更长的时间，请确保你的本地副本与主存储库保持最新。现在，只需按照 :ref:`chap_installation` 中的说明开始操作即可。主要区别在于你现在正在下载仓库的 * 你自己的 fork *。 

![Cloning the repository.](../img/git-clone.png)
:width:`700px`
:label:`fig_git_clone`

```
# Replace your_github_username with your GitHub username
git clone https://github.com/your_github_username/d2l-en.git
```

### 编辑和推送

现在是时候编辑这本书了。最好按照 :numref:`sec_jupyter` 中的说明在 Jupyter 笔记本中对其进行编辑。进行更改并检查它们是否正常。假设我们修改了文件 `~/d2l-en/chapter_appter_appdix_tools/如何贡献.md` 中的一个错字。然后，您可以检查更改了哪些文件。 

此时 Git 将提示 `chapter_appendix_tools/how-to-contribute.md` 文件已被修改。

```
mylaptop:d2l-en me$ git status
On branch master
Your branch is up-to-date with 'origin/master'.

Changes not staged for commit:
  (use "git add <file>..." to update what will be committed)
  (use "git checkout -- <file>..." to discard changes in working directory)

	modified:   chapter_appendix_tools/how-to-contribute.md
```

确认这是你想要的后，执行以下命令：

```
git add chapter_appendix_tools/how-to-contribute.md
git commit -m 'fix typo in git documentation'
git push
```

然后，更改后的代码将存放在仓库的个人复刻中。要请求添加更改，您必须为该书的官方存储库创建拉取请求。 

### 提交拉取请求

如 :numref:`fig_git_newpr` 所示，转到 GitHub 上仓库的复刻，然后选择 “新建拉取请求”。这将打开一个屏幕，向您显示您的编辑内容与书籍主存储库中最新内容之间的变化。 

![New pull request.](../img/git-newpr.png)
:width:`700px`
:label:`fig_git_newpr`

最后，单击 :numref:`fig_git_createpr` 中所示的按钮提交拉取请求。请务必描述您在拉取请求中所做的更改。这将使作者更容易对其进行审阅并将其与图书合并。根据更改的不同，这可能会立即被接受、被拒绝，或者更有可能获得有关更改的一些反馈。一旦你合并了它们，你就可以开始了。 

![Create pull request.](../img/git-createpr.png)
:width:`700px`
:label:`fig_git_createpr`

## 摘要

* 你可以使用 GitHub 为这本书做贡献。
* 您可以直接在 GitHub 上编辑文件以进行较小的更改。
* 对于重大更改，请分叉存储库，在本地编辑内容，并且只有在准备好后才进行回馈。
* 拉取请求是如何捆绑贡献的方式。尽量不要提交大量的拉取请求，因为这会使它们难以理解和合并。最好寄几个较小的。

## 练习

1. 为 `d2l-ai/d2l-en` 存储库添加星标并分叉。
1. 如果您发现任何需要改进的内容（例如缺少参考资料），请提交拉取请求。 
1. 使用新分支创建拉取请求通常是更好的做法。了解如何使用 [Git branching](https://git-scm.com/book/en/v2/Git-Branching-Branches-in-a-Nutshell) 来做到这一点。

[Discussions](https://discuss.d2l.ai/t/426)
