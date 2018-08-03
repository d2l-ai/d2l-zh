# 如何为本书贡献

本书在“致谢”部分感谢本书的所有贡献者并列出他们的Github ID或姓名。每位贡献者也将在本书出版时获得一本贡献者专享的赠书。

你可以在本书的Github代码库查看贡献者列表 [1]。如果你希望成为本书的贡献者之一，需要安装Git并为本书的Github代码库提交pull request [2]。当你的pull request被本书作者合并进了代码库后，你就成为了本书的贡献者。

本节介绍了为本书贡献的基本Git操作步骤。如果你熟悉Git的操作，可以跳过本节。

以下的操作步骤假设贡献者的Github ID为“astonzhang”。

第一步，安装Git。Git的开源书里详细介绍了安装Git的方法 [3]。如果你没有Github账号，需要注册一个账号 [4]。

第二步，登录Github。在浏览器输入本书代码库地址 [2]。点击图11.20右上方红框中的“Fork”按钮获得一份本书的代码库。

![代码库的页面。](../img/contrib01.png)


这时，本书的代码库会复制到你的用户名下，例如图11.21左上方显示的“你的Github ID/gluon-tutorials-zh”。

![复制代码库。](../img/contrib02.png)


第三步，点击图11.21右方的“Clone or download”绿色按钮，并点击红框中的按钮复制位于你用户名下的代码库地址。按[“安装和运行”](../chapter_prerequisite/install.md)介绍的方法进入命令行模式。假设我们希望将代码库保存在本地的“~/repo”路径之下。进入该路径，键入`git clone `并粘贴位于你用户名下的代码库地址。执行命令

```
# 将 your_Github_ID 替换成你的 Github 用户名。
git clone https://github.com/your_Github_ID/gluon-tutorials-zh.git
```

这时，本地的“~/repo/gluon-tutorials-zh”路径下将包含本书代码库中的所有文件。


第四步，编辑本地路径下的本书代码库。假设我们修改了`~/repo/gluon-tutorials-zh/chapter_deep-learning-basics/linear-regression.md`文件中的一个错别字。在命令行模式中进入路径`~/repo/gluon-tutorials-zh`，执行命令

```
git status
```

此时Git将提示“chapter_deep-learning-basics/linear-regression.md”文件已被修改，如图11.22所示。

![Git提示“chapter_deep-learning-basics/linear-regression.md”文件已被修改。](../img/contrib03.png)

确认将提交该修改的文件后，执行以下命令

```
git add chapter_deep-learning-basics/linear-regression.md
git commit -m 'fix typo in linear-regression.md'
git push
```

其中的`'fix typo in linear-regression.md'`是描述提交改动的信息，也可以替换为其他有意义的描述信息。


第五步，再次在浏览器输入本书代码库地址 [2]。点击图11.20左方红框中的“New pull request”按钮。在弹出的页面中，点击图11.23右方红框中的“compare across forks”链接，再点击下方红框中的“head fork: mli/gluon-tutorials-zh”按钮。在弹出的文本框中输入你的Github ID，在下拉菜单中选择“你的Github-ID/gluon-tutorials-zh”，如图11.23所示。


![选择改动来源所在的代码库。](../img/contrib04.png)


第六步，如图11.24所示，在标题和正文的文本框中描述想要提交的pull request。点击红框中的“Create pull request”绿色按钮提交pull request。

![描述并提交pull request。](../img/contrib05.png)


提交完成后，我们会看到图11.25所示的页面中显示pull request已提交。

![显示pull request已提交。](../img/contrib06.png)




## 小结

* 我们可以通过使用Github为本书做贡献。


## 练习

* 如果你觉得本书某些地方可以改进，尝试提交一个pull request。


## 扫码直达[讨论区](https://discuss.gluon.ai/t/topic/7570)

![](../img/qr_how-to-contribute.svg)


## 参考文献

[1] 本书贡献者列表。https://github.com/mli/gluon-tutorials-zh/graphs/contributors

[2] 本书代码库地址。https://github.com/mli/gluon-tutorials-zh

[3] 安装Git。https://git-scm.com/book/zh/v2

[4] Github网址。https://github.com/
