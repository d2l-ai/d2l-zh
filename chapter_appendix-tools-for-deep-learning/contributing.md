# 为这本书做贡献
:label:`sec_how_to_contribute`

[readers](https://github.com/d2l-ai/d2l-en/graphs/contributors) 的贡献帮助我们改进这本书。如果你发现错字、过时的链接、你认为我们错过了引文、代码看起来不优雅或解释不清楚的东西，请回馈并帮助我们帮助我们的读者。虽然在普通书籍中，印刷运行之间的延迟（以及错字更正之间的延迟）可以以几年来衡量，但通常需要数小时到几天才能将改进内容纳入本书。由于版本控制和持续集成测试，这一切都是可能的。要做到这一点，你需要向 GitHub 存储库提交 [pull request](https://github.com/d2l-ai/d2l-en/pulls)。当您的拉取请求被作者合并到代码存储库时，您将成为贡献者。 

## 轻微的文本更改

最常见的贡献是编辑一句话或修复拼写错误。我们建议您在 [github repo](https732293614) 中找到源文件以找到源文件，即降价文件。然后点击右上角的 “编辑此文件” 按钮以在降价文件中进行更改。 

![Edit the file on Github.](../img/edit-file.png)
:width:`300px`
:label:`fig_edit_file`

完成后，在页面底部的 “提议文件更改” 面板中填写更改说明，然后单击 “提议文件更改” 按钮。它会将你重定向到一个新页面以查看你的更改 (:numref:`fig_git_createpr`)。如果一切都很好，你可以点击 “创建拉取请求” 按钮提交拉取请求。 

## 提出重大变更

如果你计划更新很大一部分文本或代码，那么你需要更多地了解这本书使用的格式。源文件基于 [markdown format](https://daringfireball.net/projects/markdown/syntax)，通过 [d2lbook](http://book.d2l.ai/user/markdown.html) 软件包进行了一组扩展名，例如引用方程、图像、章节和引文。您可以使用任何 Markdown 编辑器打开这些文件并进行更改。 

如果您想更改代码，我们建议您使用 Jupyter 打开这些 Markdown 文件，如 :numref:`sec_jupyter` 中所述。这样你就可以运行和测试你的更改。请记住在提交更改之前清除所有输出，我们的 CI 系统将执行您更新的部分以生成输出。 

某些部分可能支持多个框架实现，您可以使用 `d2lbook` 激活特定框架，因此其他框架实现成为 Markdown 代码块，在 Jupyter 中 “全部运行” 时不会执行。换句话说，首先通过运行来安装 `d2lbook`

```bash
pip install git+https://github.com/d2l-ai/d2l-book
```

然后在 `d2l-en` 的根目录中，您可以通过运行以下命令之一来激活特定的实现：

```bash
d2lbook activate mxnet chapter_multilayer-perceptrons/mlp-scratch.md
d2lbook activate pytorch chapter_multilayer-perceptrons/mlp-scratch.md
d2lbook activate tensorflow chapter_multilayer-perceptrons/mlp-scratch.md
```

在提交更改之前，请清除所有代码块输出并通过

```bash
d2lbook activate all chapter_multilayer-perceptrons/mlp-scratch.md
```

如果你添加了一个新的代码块，而不是默认实现，即 MXNet，请使用 `# @tab` to mark this block on the beginning line. For example, ` # @tab pytorch` for a PyTorch code block, `# @tab tensorflow` for a TensorFlow code block, or `# @tab all `共享代码块用于所有实现。你可以参考 [d2lbook](http://book.d2l.ai/user/code_tabs.html) 了解更多信息。 

## 添加新部分或新框架实施

如果你想创建一个新的章节，例如强化学习，或者添加新框架（例如 TensorFlow）的实现，请先通过电子邮件或使用 [github issues](https://github.com/d2l-ai/d2l-en/issues) 联系作者。 

## 提交重大更改

我们建议您使用标准 `git` 流程提交重大更改。简而言之，该过程正如 :numref:`fig_contribute` 所述的那样工作。 

![Contributing to the book.](../img/contribute.svg)
:label:`fig_contribute`

我们将详细指导你完成这些步骤。如果你已经熟悉 Git，你可以跳过这一节。为了具体起见，我们假设贡献者的用户名是 “astonhang”。 

### 安装 Git

Git 开源手册描述了 [how to install Git](https://git-scm.com/book/en/v2)。这通常在 Ubuntu Linux 上通过 `apt install git`、通过在 macOS 上安装 Xcode 开发人员工具或使用 GitHub 的 [desktop client](https://desktop.github.com) 来工作。如果你没有 GitHub 帐户，你需要注册一个。 

### 登录 GitHub

在浏览器中输入电子书代码库的 [address](https://github.com/d2l-ai/d2l-en/)。点击 :numref:`fig_git_fork` 右上角红色框中的 `Fork` 按钮，制作这本书的存储库。现在这是 * 你的副本 *，你可以任何你想要的方式更改它。 

![The code repository page.](../img/git-fork.png)
:width:`700px`
:label:`fig_git_fork`

现在，这本书的代码库将分叉（即复制）到您的用户名，例如屏幕截图 :numref:`fig_git_forked` 左上角显示的 `astonzhang/d2l-en`。 

![Fork the code repository.](../img/git-forked.png)
:width:`700px`
:label:`fig_git_forked`

### 克隆存储库

要克隆存储库（即制作本地副本），我们需要获取其存储库地址。:numref:`fig_git_clone` 中的绿色按钮显示了这一点。如果您决定将此分叉保留更长时间，请确保本地副本与主存储库保持最新状态。现在只需按照 :ref:`chap_installation` 中的说明开始操作即可。主要区别在于你现在正在下载 * 你自己的仓库分支 *。 

![Git clone.](../img/git-clone.png)
:width:`700px`
:label:`fig_git_clone`

```
# Replace your_github_username with your GitHub username
git clone https://github.com/your_github_username/d2l-en.git
```

### 编辑书和推送

现在是时候编辑这本书了。最好按照 :numref:`sec_jupyter` 中的说明在 Jupyter 中编辑笔记本电脑。进行更改并检查它们是否正常。假设我们修改了文件中的错字 `~/d2l-en/章ter_appendix_tools/如何贡献.md`。然后你可以检查你更改了哪些文件： 

此时，Git 将提示 `chapter_appendix_tools/how-to-contribute.md` 文件已被修改。

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

然后，更改后的代码将放在存储库的个人分叉中。要请求添加更改，必须为图书的官方存储库创建拉取请求。 

### 拉请求

如 :numref:`fig_git_newpr` 所示，转到 GitHub 上仓库的分叉，然后选择 “新的拉取请求”。这将打开一个屏幕，显示编辑内容与图书主存储库中当前编辑内容之间的变化。 

![Pull Request.](../img/git-newpr.png)
:width:`700px`
:label:`fig_git_newpr`

### 提交拉取请求

最后，点击按钮提交拉取请求，如 :numref:`fig_git_createpr` 所示。请务必描述您在拉取请求中所做的更改。这将使作者更容易审阅它并将其与书合并。根据更改的不同，这可能会立即被接受、拒绝，或者更有可能，您将获得有关更改的一些反馈。一旦你把它们合并起来，你就可以走了。 

![Create Pull Request.](../img/git-createpr.png)
:width:`700px`
:label:`fig_git_createpr`

您的拉取请求将出现在主仓库的请求列表中。我们将尽一切努力快速处理它。 

## 摘要

* 你可以使用 GitHub 为这本书做贡献。
* 您可以直接在 GitHub 上编辑文件以进行小的更改。
* 对于重大更改，请分叉存储库，在本地编辑内容，只有在准备好后才能回馈。
* 拉取请求是将贡献捆绑起来的方式。尽量不要提交巨大的拉取请求，因为这使得它们难以理解和纳入。最好发送几个较小的。

## 练习

1. 对 `d2l-en` 存储库进行标记和分叉。
1. 找到一些需要改进的代码并提交拉取请求。
1. 找到我们错过的参考资料并提交拉取请求。
1. 使用新分支创建拉取请求通常是更好的做法。了解如何用 [Git branching](https://git-scm.com/book/en/v2/Git-Branching-Branches-in-a-Nutshell) 做到这一点。

[Discussions](https://discuss.d2l.ai/t/426)
