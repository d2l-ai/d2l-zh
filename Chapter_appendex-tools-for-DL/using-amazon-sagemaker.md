

<!--
 * @version:
 * @Author:  StevenJokes https://github.com/StevenJokes
 * @Date: 2020-07-03 12:45:05
 * @LastEditors:  StevenJokes https://github.com/StevenJokes
 * @LastEditTime: 2020-07-03 16:14:55
 * @Description:translate
 * @TODO::fig,code.
 * @Reference:http://preview.d2l.ai/d2l-en/PR-1103/chapter_appendix-tools-for-deep-learning/sagemaker.html
-->

# 使用 Amazon SageMaker

## 注册和登录

首先，我们需要在 https://aws.amazon.com/注册一个账户。为了提高安全性，我们鼓励您使用双因素身份验证。设置详细的账单和支出提醒也是一个好主意，以避免在忘记停止任何运行实例时出现意外情况。注意，你需要一张信用卡。登录您的 AWS 帐户后，搜索“ SageMaker”(见图19.2.1) ，然后单击打开 SageMaker 面板。

接下来，让我们创建一个如图19.2.2所示的笔记本实例。

SageMaker 提供了不同计算能力和价格的多种实例类型。在创建实例时，我们可以指定实例名并选择其类型。在图19.2.3中，我们选择 ml.p3.2 xlarge。拥有一个 Tesla V100 GPU 和一个8核 CPU，这个例子对于大多数章节来说足够强大。

这本书的一个 Jupyter 笔记本版本可以在 https://github.com/d2l-ai/d2l-pytorch-SageMaker 上找到。我们可以指定这个 GitHub 存储库 URL，让 SageMaker 在实例创建期间克隆这个存储库，如图19.2.4所示。

实例可能需要几分钟时间才能准备好。准备就绪后，你可以点击图19.2.5所示的“Open Jupyter”链接。

然后，如图19.2.6所示，您可以浏览运行在这个实例上的Jupyter服务器。

在SagaMaker实例上运行和编辑Jupyter笔记本与我们在19.1节中讨论的内容类似。工作完成后，不要忘记停止实例，以免进一步收费，如图19.2.7所示。

## 更新

我们将定期更新notebooks 在Github的仓库[d2l-ai/d2l-en-sagemaker](https://github.com/d2l-ai/d2l-en-sagemaker),你能简单地用 `git pull` 命令更新到最新版本。首先，您需要打开一个终端，如图19.2.8所示。

您可能希望在拉取更新之前提交本地更改。或者，您可以在终端中使用以下命令忽略所有本地更改。


## 小结

- 我们可以通过Amazon SageMaker启动和停止Jupyter服务器来运行这本书。
- 我们可以通过Amazon SageMaker实例上的终端更新笔记本电脑。

## 练习

1. 尝试使用Amazon SageMaker编辑并运行本书中的代码。
1. 通过终端访问源代码目录。
