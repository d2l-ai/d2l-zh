

<!--
 * @version:
 * @Author:  StevenJokes https://github.com/StevenJokes
 * @Date: 2020-07-03 14:43:55
 * @LastEditors:  StevenJokes https://github.com/StevenJokes
 * @LastEditTime: 2020-07-05 18:40:29
 * @Description:
 * @TODO::Fig 19.3.13后面
 * @Reference:http://preview.d2l.ai/d2l-en/PR-1117/chapter_appendix-tools-for-deep-learning/aws.html
-->

# 使用AWS EC2实例

在本节中，我们将向您展示如何在原始Linux机器上安装所有库。记住，在[19.2节](http://preview.d2l.ai/d2l-en/PR-1102/chapter_appendix-tools-for-deep-learning/sagemaker.html#sec-sagemaker)中，我们讨论了如何使用Amazon SageMaker，而自己在AWS上构建实例的成本更低。演练包括许多步骤:
1. 从AWS EC2请求一个GPU Linux实例。
1. 可选:安装CUDA或使用预安装CUDA的AMI。
1. 设置相应的MXNet GPU版本。
这个过程也适用于其他实例(和其他云)，尽管需要进行一些小的修改。在继续之前，您需要创建一个AWS帐户，请参阅[19.2节](http://preview.d2l.ai/d2l-en/PR-1102/chapter_appendix-tools-for-deep-learning/sagemaker.html#sec-sagemaker)了解更多信息

## 创建并运行EC2实例

登录AWS账户后，点击EC2(图19.3.1中红色方框标记)进入EC2面板。

图19.3.2显示了将敏感帐户信息灰色化的EC2面板。

## 预设定位置

选择附近的数据中心以减少延迟，如Oregon([图19.3.2](http://preview.d2l.ai/d2l-en/PR-1102/chapter_appendix-tools-for-deep-learning/aws.html#fig-ec2)右上角的红色框所标记)。如果你在中国，你可以选择附近的亚太地区，如首尔或东京。请注意，一些数据中心可能没有GPU实例。

## 增加Limits

在选择实例前，点击左侧栏内的limit标签，检查是否有数量限制，如图19.3.2所示。图19.3.3给出了这种限制的示例。账户目前无法打开p2每个区域的xlarge实例。如果需要打开一个或多个实例，请单击 “Request limit increase” 链接以申请更高的实例配额。一般来说，处理一个申请需要一个工作日。

TODO:PIC

## 启动实例

接下来，点击[图19.3.2中](http://preview.d2l.ai/d2l-en/PR-1102/chapter_appendix-tools-for-deep-learning/aws.html#fig-ec2)红色框标记的 “Launch Instance” 按钮，启动你的实例。我们首先选择一个合适的AMI (AWS机器映像)。在搜索框中输入Ubuntu(由图19.3.4中的红色框标记)。
由于云服务是按使用时计费的，因此应该关闭未使用的实例。注意，还有其他选择:停止一个实例意味着您将能够重新启动它。这类似于关闭常规服务器的电源。但是，对于保留的硬盘空间，停止的实例仍将收取少量费用。Terminate删除与它关联的所有数据。这包括磁盘，因此您不能再次启动它。只有在你知道你将来不需要它的时候才这样做。
要远程运行Jupyter，您需要使用SSH端口转发。毕竟，云中的服务器没有显示器或键盘。为此，从您的桌面(或笔记本电脑)按如下方式登录到服务器。

TODO:CODE

图19.3.13显示了运行Jupyter Notebook后的可能输出。最后一行是端口8888的URL。

TODO:PIC

因为您使用了端口转发到端口8889，所以在本地浏览器中打开URL时，需要替换端口号并使用Jupyter给出的秘密。

## 关闭未使用的实例

由于云服务是按使用时计费的，因此应该关闭未使用的实例。注意，还有其他选择:停止一个实例意味着您将能够重新启动它。这类似于关闭常规服务器的电源。但是，对于保留的硬盘空间，停止的实例仍将收取少量费用。Terminate删除与它关联的所有数据。这包括磁盘，因此您不能再次启动它。只有在你知道你将来不需要它的时候才这样做。

如果您想使用实例作为更多实例的模板，请右键单击图19.3.9中的示例，并选择Image Create来创建实例的图像。完成此操作后，选择Instance State Terminate终止实例。下一次使用此实例时，可以按照本节中描述的创建和运行EC2实例的步骤，基于已保存的映像创建实例。唯一的区别是，在1。选择如[图19.3.4](http://preview.d2l.ai/d2l-en/PR-1102/chapter_appendix-tools-for-deep-learning/aws.html#fig-ubuntu)所示的“1. Choose AMI”，必须使用左侧的My AMIs选项来选择已保存的图像。创建的实例将保留存储在映像硬盘上的信息。例如，你将不必重新安装CUDA和其他运行时环境。

## 小结

- 您可以根据需要启动和停止实例，而不必购买和构建自己的计算机。
- 你需要安装合适的GPU驱动程序才能使用它们。

## 练习

1. 云计算提供了便利，但并不便宜。了解如何启动[spot实例](https://aws.amazon.com/ec2/spot/)来查看1. 如何降低价格。
1. 尝试不同的GPU服务器。它们有多快?
1. 尝试使用多gpu服务器。你能把事情放大多少
