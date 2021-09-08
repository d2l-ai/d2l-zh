# 使用亚马逊 SageMaker
:label:`sec_sagemaker`

许多深度学习应用程序需要大量的计算。您的本地计算机可能太慢，无法在合理的时间内解决这些问题。云计算服务可让您访问功能更强大的计算机来运行本书中 GPU 密集型部分。本教程将指导您完成 Amazon SageMaker：一项允许您轻松运行本书的服务。 

## 注册和登录

首先，我们需要在 https://aws.amazon.com/ 注册账户。我们鼓励您使用双重身份验证来增加安全性。设置详细的账单和支出提醒也是一个好主意，以避免在忘记停止任何正在运行的实例的情况下出现任何意外的意外情况。请注意，你需要一张信用卡。登录 AWS 账户后，转到 [console](http://console.aws.amazon.com/) 并搜索 “SageMaker”（请参阅 :numref:`fig_sagemaker`），然后单击打开 SageMaker 面板。 

![Open the SageMaker panel.](../img/sagemaker.png)
:width:`300px`
:label:`fig_sagemaker`

## 创建 SageMaker 实例

接下来，让我们按照 :numref:`fig_sagemaker-create` 中的描述创建一个笔记本实例。 

![Create a SageMaker instance.](../img/sagemaker-create.png)
:width:`400px`
:label:`fig_sagemaker-create`

SageMaker 提供多个 [instance types](https://aws.amazon.com/sagemaker/pricing/instance-types/) 不同的计算能力和价格。创建实例时，我们可以指定实例名称并选择其类型。在 :numref:`fig_sagemaker-create-2` 中，我们选择了 `ml.p3.2xlarge`。有了一个特斯拉 V100 GPU 和一个 8 核 CPU，这个实例对于大多数章节来说都足够强大。 

![Choose the instance type.](../img/sagemaker-create-2.png)
:width:`400px`
:label:`fig_sagemaker-create-2`

:begin_tab:`mxnet`
这本书的 Jupyter 笔记本版本可在 https://github.com/d2l-ai/d2l-en-sagemaker. We can specify this GitHub repository URL to let SageMaker clone this repository during instance creation, as shown in :numref:`fig_sagemaker-create-3` 上获得，以适合 SageMaker。
:end_tab:

:begin_tab:`pytorch`
这本书的 Jupyter 笔记本版本可在 https://github.com/d2l-ai/d2l-pytorch-sagemaker. We can specify this GitHub repository URL to let SageMaker clone this repository during instance creation, as shown in :numref:`fig_sagemaker-create-3` 上获得，以适合 SageMaker。
:end_tab:

:begin_tab:`tensorflow`
这本书的 Jupyter 笔记本版本可在 https://github.com/d2l-ai/d2l-tensorflow-sagemaker. We can specify this GitHub repository URL to let SageMaker clone this repository during instance creation, as shown in :numref:`fig_sagemaker-create-3` 上获得，以适合 SageMaker。
:end_tab:

![Specify the GitHub repository.](../img/sagemaker-create-3.png)
:width:`400px`
:label:`fig_sagemaker-create-3`

## 运行和停止实例

实例可能需要几分钟才能准备就绪。准备就绪后，你可以点击 “打开 Jupyter” 链接，如 :numref:`fig_sagemaker-open` 所示。 

![Open Jupyter on the created SageMaker instance.](../img/sagemaker-open.png)
:width:`400px`
:label:`fig_sagemaker-open`

然后，如 :numref:`fig_sagemaker-jupyter` 所示，您可以在此实例上运行的 Jupyter 服务器进行导航。 

![The Jupyter server running on the SageMaker instance.](../img/sagemaker-jupyter.png)
:width:`400px`
:label:`fig_sagemaker-jupyter`

在 SageMaker 实例上运行和编辑 Jupyter 笔记本类似于我们在 :numref:`sec_jupyter` 中讨论的内容。完成工作后，不要忘记停止实例以避免进一步充电，如 :numref:`fig_sagemaker-stop` 所示。 

![Stop a SageMaker instance.](../img/sagemaker-stop.png)
:width:`300px`
:label:`fig_sagemaker-stop`

## 更新笔记本

:begin_tab:`mxnet`
我们将定期更新 [d2l-ai/d2l-en-sagemaker](https://github.com/d2l-ai/d2l-en-sagemaker) GitHub 存储库中的笔记本电脑。你可以简单地使用 `git pull` 命令更新到最新版本。
:end_tab:

:begin_tab:`pytorch`
我们将定期更新 [d2l-ai/d2l-pytorch-sagemaker](https://github.com/d2l-ai/d2l-pytorch-sagemaker) GitHub 存储库中的笔记本电脑。你可以简单地使用 `git pull` 命令更新到最新版本。
:end_tab:

:begin_tab:`tensorflow`
我们将定期更新 [d2l-ai/d2l-tensorflow-sagemaker](https://github.com/d2l-ai/d2l-tensorflow-sagemaker) GitHub 存储库中的笔记本电脑。你可以简单地使用 `git pull` 命令更新到最新版本。
:end_tab:

首先，你需要打开一个终端，如 :numref:`fig_sagemaker-terminal` 所示。 

![Open a terminal on the SageMaker instance.](../img/sagemaker-terminal.png)
:width:`300px`
:label:`fig_sagemaker-terminal`

您可能想在提取更新之前提交本地更改。或者，您可以使用终端中的以下命令简单地忽略所有本地更改。

:begin_tab:`mxnet`
```bash
cd SageMaker/d2l-en-sagemaker/
git reset --hard
git pull
```
:end_tab:

:begin_tab:`pytorch`
```bash
cd SageMaker/d2l-pytorch-sagemaker/
git reset --hard
git pull
```
:end_tab:

:begin_tab:`tensorflow`
```bash
cd SageMaker/d2l-tensorflow-sagemaker/
git reset --hard
git pull
```
:end_tab:

## 摘要

* 我们可以通过 Amazon SageMaker 启动和停止 Jupyter 服务器来运行这本书。
* 我们可以通过 Amazon SageMaker 实例上的终端更新笔记本电脑。

## 练习

1. 尝试使用 Amazon SageMaker 编辑和运行本书中的代码。
1. 通过终端访问源代码目录。

[Discussions](https://discuss.d2l.ai/t/422)
