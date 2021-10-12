# 使用 AWS EC2 实例
:label:`sec_aws`

在本节中，我们将向您展示如何在原始 Linux 计算机上安装所有库。回想一下，在 :numref:`sec_sagemaker` 中，我们讨论了如何使用 Amazon SageMaker，同时自己构建实例在 AWS 上的成本更低。本演练包括三个步骤： 

1. 从 AWS EC2 请求一个 GPU Linux 实例。
1. 安装 CUDA（或使用带有预装 CUDA 的亚马逊系统映像）。
1. 安装用于运行本书代码的深度学习框架和其他库。

此过程也适用于其他实例（和其他云），尽管有一些小的修改。在继续之前，您需要创建一个 AWS 账户，有关更多详细信息，请参阅 :numref:`sec_sagemaker`。 

## 创建和运行 EC2 实例

登录 AWS 账户后，单击 “EC2”（由 :numref:`fig_aws` 中的红色框标记）进入 EC2 面板。 

![Open the EC2 console.](../img/aws.png)
:width:`400px`
:label:`fig_aws`

:numref:`fig_ec2` 显示 EC2 面板，其中敏感账户信息显示为灰色。 

![EC2 panel.](../img/ec2.png)
:width:`700px`
:label:`fig_ec2`

### 预设位置选择附近的数据中心以减少延迟，例如 “俄勒冈州”（由 :numref:`fig_ec2` 右上角的红框标记）。如果您位于中国，则可以选择附近的亚太地区，例如首尔或东京。请注意，某些数据中心可能没有 GPU 实例。 

### 提高限额

在选择实例之前，请单击左侧栏中的 “限制” 标签，检查是否存在数量限制，如 :numref:`fig_ec2` 所示。:numref:`fig_limits` 显示了此类限制的示例。账户目前无法在每个区域打开 “p2.xlarge” 实例。如果您需要打开一个或多个实例，请单击 “请求提高限制” 链接以申请更高的实例配额。通常，处理申请需要一个工作日。 

![Instance quantity restrictions.](../img/limits.png)
:width:`700px`
:label:`fig_limits`

### 启动实例

接下来，单击 :numref:`fig_ec2` 中红色框标记的 “启动实例” 按钮以启动您的实例。 

我们首先选择合适的亚马逊系统映像 (AMI)。在搜索框中输入 “Ubuntu”（由 :numref:`fig_ubuntu` 中的红色框标记）。 

![Choose an AMI.](../img/ubuntu-new.png)
:width:`700px`
:label:`fig_ubuntu`

EC2 提供了许多不同的实例配置供您选择。对于初学者来说，这有时会让人感到不知所措。:numref:`tab_ec2` 列出了不同的合适机器。 

: 不同的 EC2 实例类型 

| Name | GPU         | Notes                         |
|------|-------------|-------------------------------|
| g2   | Grid K520   | ancient                       |
| p2   | Kepler K80  | old but often cheap as spot   |
| g3   | Maxwell M60 | good trade-off                |
| p3   | Volta V100  | high performance for FP16     |
| g4   | Turing T4   | inference optimized FP16/INT8 |
:label:`tab_ec2`

所有这些服务器都有多种类型，表示使用的 GPU 数量。例如，p2.xlarge 有 1 个 GPU，一个 p2.16xlarge 有 16 个 GPU 和更多内存。有关更多详细信息，请参阅 [AWS EC2 documentation](https732293614)。 

![Choose an instance.](../img/p2x.png)
:width:`700px`
:label:`fig_p2x`

请注意，您应该使用具有合适驱动程序和支持 GPU 的深度学习框架的启用 GPU 的实例。否则，使用 GPU 将看不到任何好处。 

到目前为止，我们已经完成了启动 EC2 实例的七个步骤中的前两个步骤，如 :numref:`fig_disk` 的顶部所示。在此示例中，我们保留步骤 “3.配置实例”、“5.添加标签” 和 “6.配置安全组”。点击 “4。添加存储空间” 并将默认硬盘大小增加到 64 GB（标记在 :numref:`fig_disk` 的红框中）。请注意，CUDA 本身已经占用了 4 GB。 

![Modify the hard disk size.](../img/disk.png)
:width:`700px`
:label:`fig_disk`

最后，转到 “7。查看”，然后单击 “启动” 以启动已配置的实例。系统现在将提示您选择用于访问实例的密钥对。如果您没有密钥对，请在 :numref:`fig_keypair` 的第一个下拉菜单中选择 “创建新密钥对” 以生成密钥对。随后，您可以为此菜单选择 “选择现有密钥对”，然后选择先前生成的密钥对。单击 “启动实例” 以启动创建的实例。 

![Select a key pair.](../img/keypair.png)
:width:`500px`
:label:`fig_keypair`

如果生成了新密钥对，请确保下载密钥对并将其存储在安全的位置。这是你通过 SSH 进入服务器的唯一方法。单击 :numref:`fig_launching` 中显示的实例 ID 可查看此实例的状态。 

![Click the instance ID.](../img/launching.png)
:width:`700px`
:label:`fig_launching`

### 连接到实例

如 :numref:`fig_connect` 中所示，在实例状态变为绿色后，右键单击该实例，然后选择 `Connect` 以查看实例访问方法。 

![View instance access method.](../img/connect.png)
:width:`700px`
:label:`fig_connect`

如果这是一个新密钥，则 SSH 必须不能公开查看它才能正常工作。转到存储 `D2L_key.pem` 的文件夹，然后执行以下命令使密钥不公开可见：

```bash
chmod 400 D2L_key.pem
```

![View instance access and startup method.](../img/chmod.png)
:width:`400px`
:label:`fig_chmod`

现在，将 ssh 命令复制到下方红色框 :numref:`fig_chmod` 中，然后粘贴到命令行上：

```bash
ssh -i "D2L_key.pem" ubuntu@ec2-xx-xxx-xxx-xxx.y.compute.amazonaws.com
```

当命令行提示 “是否确实要继续连接（是/否）” 时，输入 “是”，然后按 Enter 键登录实例。 

你的服务器现在准备好了。 

## 安装 CUDA

在安装 CUDA 之前，请务必使用最新的驱动程序更新实例。

```bash
sudo apt-get update && sudo apt-get install -y build-essential git libgfortran3
```

在这里我们下载 CUDA 10.1。访问英伟达的 [官方存储库](https://developer.nvidia.com/cuda-toolkit-archive) to find the download link as shown in :numref:`fig_cuda`。 

![Find the CUDA 10.1 download address.](../img/cuda101.png)
:width:`500px`
:label:`fig_cuda`

复制说明并将其粘贴到终端上以安装 CUDA 10.1。

```bash
# The link and file name are subject to changes
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/cuda-ubuntu1804.pin
sudo mv cuda-ubuntu1804.pin /etc/apt/preferences.d/cuda-repository-pin-600
wget http://developer.download.nvidia.com/compute/cuda/10.1/Prod/local_installers/cuda-repo-ubuntu1804-10-1-local-10.1.243-418.87.00_1.0-1_amd64.deb
sudo dpkg -i cuda-repo-ubuntu1804-10-1-local-10.1.243-418.87.00_1.0-1_amd64.deb
sudo apt-key add /var/cuda-repo-10-1-local-10.1.243-418.87.00/7fa2af80.pub
sudo apt-get update
sudo apt-get -y install cuda
```

安装程序后，运行以下命令查看 GPU：

```bash
nvidia-smi
```

最后，将 CUDA 添加到库路径中以帮助其他图书馆找到它。

```bash
echo "export LD_LIBRARY_PATH=\${LD_LIBRARY_PATH}:/usr/local/cuda/lib64" >> ~/.bashrc
```

## 安装用于运行代码的库

要运行本书的代码，只需在 EC2 实例上按照适用于 Linux 用户的 :ref:`chap_installation` 中的步骤操作，然后使用以下提示在远程 Linux 服务器上工作： 

* 要在 Miniconda 安装页面上下载 bash 脚本，请右键单击下载链接并选择 “复制链接地址”，然后执行 `wget [copied link address]`。
* 在运行 `~/miniconda3/bin/conda init`, you may execute `source ~/.bashrc` 之后，而不是关闭然后重新打开你当前的外壳程序。

## 远程运行 Jupyter 笔记本

要远程运行 Jupyter 笔记本，你需要使用 SSH 端口转发。毕竟，云中的服务器没有显示器或键盘。为此，请按照以下步骤从台式机（或笔记本电脑）登录服务器：

```
# This command must be run in the local command line
ssh -i "/path/to/key.pem" ubuntu@ec2-xx-xxx-xxx-xxx.y.compute.amazonaws.com -L 8889:localhost:8888
```

接下来，转到 EC2 实例上已下载的本书代码的位置，然后运行：

```
conda activate d2l
jupyter notebook
```

:numref:`fig_jupyter` 显示运行 Jupyter 笔记本后可能的输出。最后一行是端口 8888 的 URL。 

![Output after running the Jupyter Notebook. The last row is the URL for port 8888.](../img/jupyter.png)
:width:`700px`
:label:`fig_jupyter`

由于您使用了端口转发到端口 8889，因此请复制 :numref:`fig_jupyter` 红框中的最后一行，将 URL 中的 “8888” 替换为 “8889”，然后在本地浏览器中将其打开。 

## 关闭未使用实例

由于云服务是在使用时计费的，因此您应该关闭未使用的实例。请注意，还有其他选择： 

* “停止” 一个实例意味着你将能够重新启动它。这类似于关闭常规服务器的电源。但是，停止的实例仍将按保留的硬盘空间支付少量费用。 
* “终止” 实例将删除与其关联的所有数据。这包括磁盘，因此您无法再次启动它。只有在你知道将来不需要它的情况下才这样做。

如果您想将实例用作更多实例的模板，请右键单击 :numref:`fig_connect` 中的示例，然后选择 “映像” $\rightarrow$ “创建” 以创建实例的映像。完成此操作后，选择 “实例状态” $\rightarrow$ “终止” 以终止实例。下次要使用此实例时，您可以按照本节中的步骤基于保存的映像创建实例。唯一的区别是，在 “1.选择 :numref:`fig_ubuntu` 中显示的 AMI”，您必须使用左侧的 “我的 AMI” 选项来选择保存的映像。创建的实例将保留存储在映像硬盘上的信息。例如，您不必重新安装 CUDA 和其他运行时环境。 

## 摘要

* 我们可以按需启动和停止实例，而无需购买和构建自己的计算机。
* 在使用支持 GPU 的深度学习框架之前，我们需要安装 CUDA。
* 我们可以使用端口转发在远程服务器上运行 Jupyter Notebook。

## 练习

1. 云提供了便利，但并不便宜。了解如何启动 [spot instances](https://aws.amazon.com/ec2/spot/) 以了解如何降低成本。
1. 尝试使用不同的 GPU 服务器。他们有多快？
1. 尝试使用多 GPU 服务器。你能如何扩大规模？

[Discussions](https://discuss.d2l.ai/t/423)
