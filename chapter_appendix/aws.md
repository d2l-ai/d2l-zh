# 使用AWS运行代码

当本地机器的计算资源有限时，我们可以通过云计算服务获取更强大的计算资源来运行本书中的深度学习代码。本节将介绍如何在AWS（亚马逊的云计算服务）上申请实例并通过Jupyter notebook运行代码。本节中的例子基于申请含一个K80 GPU的“p2.xlarge”实例和安装CUDA8.0及相应GPU版本的MXNet（mxnet-cu80）。申请其他类型的实例或安装其他版本的MXNet的方法同本节类似。



## 申请账号并登陆

首先，我们需要在 https://aws.amazon.com/ 网站上创建账号。这通常需要一张信用卡。需要注意的是，AWS中国需要公司实体才能注册。如果你是个人用户，请注册AWS全球账号。

登陆AWS账号后，点击图11.8红框中的“EC2”进入EC2面板。

![登陆AWS账号。](../img/aws.png)


## 创建并运行EC2实例

图11.9展示了EC2面板的界面。在图11.9右上角红框处选择离我们较近的数据中心来减低延迟。我们可以选离国内较近的亚太地区，例如Asia Pacific（Seoul）。注意，有些数据中心可能没有GPU实例。点击图11.9下方红框内“Launch Instance”启动实例。

![EC2面板。](../img/ec2.png)


图11.10的最上面一行显示了配置实例所需的7个步骤。在第一步“1. Chosse AMI”中，选择Ubuntu 16.04作为操作系统。

![选择操作系统。](../img/os.png)

EC2提供了大量的有着不同配置的实例。如图11.11所示，在第二步“2. Chosse Instance Type”中，选择有一个K80 GPU的“p2.xlarge”实例。我们也可以选择像“p2.16xlarge”这样有多个GPU的实例。如果你想比较不同实例的机器配置和收费，可参考 https://www.ec2instances.info/ 。

![选择实例。](../img/p2x.png)

我们建议在选择实例前先在图11.9左栏“Limits”里检查下有无数量限制。如图11.12所示，该账号的限制是最多在一个区域开一个“p2.xlarge”实例。如果需要开更多实例，可以通过点击右边“Request limit increase”来申请更大的实例容量。这通常需要一个工作日来处理。

![实例的数量限制。](../img/limits.png)

我们将保持第三步“3. Configure Instance”、第五步“5. Add Tags”和第六步“6. Configure Security Group”中的默认配置不变。点击第四步“4.Add Storage”，如图11.13所示，将默认的硬盘大小增大到40GB。注意，安装CUDA需要4GB左右空间。

![修改实例的硬盘大小。](../img/disk.png)


最后，在第七步“7. Review”中点击“Launch”来启动配置好的实例。这时候会提示我们选择用来访问实例的密钥。如果没有的话，可以选择图11.14中第一个下拉菜单的“Create a new key pair”选项来生成秘钥。之后，我们通过该下拉菜单的“Choose an existing key pair”选项选择生成好的密钥。点击“Launch Instance”启动创建好的实例。

![选择密钥。](../img/keypair.png)

点击图11.15中的实例ID就可以查看该实例的状态了。

![点击实例ID。](../img/launching.png)

如图11.16所示，当实例状态（Instance State）变绿后，右击实例并选择“Connect”，这时就可以看到访问该实例的方法了。例如在命令行输入

```
ssh -i "/path/to/key.pem" ubuntu@ec2-xx-xxx-xxx-xxx.y.compute.amazonaws.com
```

其中“/path/to/key.pem”是本地存放访问实例的密钥的路径。当命令行提示“Are you sure you want to continue connecting (yes/no)”时，键入“yes”并按回车键即可登录创建好的实例。

![查看访问开启实例的方法。](../img/connect.png)


## 安装CUDA

如果你登录的是一个GPU实例，需要下载并安装CUDA。首先，更新并安装编译需要的包：

```
sudo apt-get update && sudo apt-get install -y build-essential git libgfortran3
```

然后，访问Nvidia官网（https://developer.nvidia.com/cuda-80-ga2-download-archive ）获取正确版本的CUDA8.0的下载地址，如图11.17所示。

![获取CUDA8.0的下载地址。](../img/cuda.png)


获取下载地址后，我们将下载并安装CUDA8.0，例如

```
wget https://developer.nvidia.com/compute/cuda/8.0/Prod2/local_installers/cuda_8.0.61_375.26_linux-run
sudo sh cuda_8.0.61_375.26_linux-run
```

点击“Ctrl+C”跳出文档浏览，并回答以下几个问题。

```
accept/decline/quit: accept
Install NVIDIA Accelerated Graphics Driver for Linux-x86_64 375.26?
(y)es/(n)o/(q)uit: y
Do you want to install the OpenGL libraries?
(y)es/(n)o/(q)uit [ default is yes ]: y
Do you want to run nvidia-xconfig?
(y)es/(n)o/(q)uit [ default is no ]: n
Install the CUDA 8.0 Toolkit?
(y)es/(n)o/(q)uit: y
Enter Toolkit Location
 [ default is /usr/local/cuda-8.0 ]:
Do you want to install a symbolic link at /usr/local/cuda?
(y)es/(n)o/(q)uit: y
Install the CUDA 8.0 Samples?
(y)es/(n)o/(q)uit: n
```

当安装完成后，运行下面的命令就可以看到该实例的GPU了。

```
nvidia-smi
```

最后，将CUDA加入到库的路径中，以方便其他库找到它。

```
echo "export LD_LIBRARY_PATH=\${LD_LIBRARY_PATH}:/usr/local/cuda-8.0/lib64" >>.bashrc
```

## 获取本书代码并安装GPU版的MXNet

我们已在[“安装和运行”](../chapter_prerequisite/install.md)一节中介绍了Linux用户获取本书代码并安装运行环境的方法。首先，安装Linux版的Miniconda（网址：https://conda.io/miniconda.html ），例如

```
wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh
```

这时需要回答下面几个问题：

```
Do you accept the license terms? [yes|no]
[no] >>> yes
Do you wish the installer to prepend the Miniconda3 install location
to PATH in your /home/ubuntu/.bashrc ? [yes|no]
[no] >>> yes
```

安装完成后，运行一次`source ~/.bashrc`让CUDA和conda生效。接下来，下载本书代码、安装并激活conda环境

```
mkdir gluon_tutorials_zh && cd gluon_tutorials_zh
curl https://zh.gluon.ai/gluon_tutorials_zh.tar.gz -o tutorials.tar.gz
tar -xzvf tutorials.tar.gz && rm tutorials.tar.gz
conda env create -f environment.yml
source activate gluon
```

默认环境里安装了CPU版本的MXNet。现在我们将它替换成GPU版本的MXNet（1.2.1 版）。

```
pip uninstall mxnet
pip install mxnet-cu80==1.2.1
```

## 运行Jupyter notebook

现在，我们可以运行Jupyter notebook了：

```
jupyter notebook
```

图11.18显示了运行后可能的输出，其中最后一行为8888端口下的URL。

![运行Jupyter notebook后的输出，其中最后一行为8888端口下的URL。](../img/jupyter.png)

由于创建的实例并没有暴露8888端口，我们可以在本地命令行启动ssh从实例映射到本地8889端口。

```
# 该命令须在本地命令行运行。
ssh -i "/path/to/key.pem" ubuntu@ec2-xx-xxx-xxx-xxx.y.compute.amazonaws.com -L 8889:localhost:8888
```

最后，把图11.18中运行Jupyter notebook后输出的最后一行URL复制到本地浏览器，并将8888改为8889。点击回车键即可从本地浏览器通过Jupyter notebook运行实例上的代码。

## 关闭不使用的实例

因为云服务按使用时长计费，我们通常会在不使用实例时将其关闭。

如果较短时间内还将重新开启实例，右击图11.16中的示例，选择“Instance State” $\rightarrow$ “Stop”将实例停止，等下次使用时选择“Instance State” $\rightarrow$ “Start”重新开启实例。这种情况下，开启的实例将保留其停止前硬盘上的存储（例如无需再安装CUDA和其他运行环境）。然而，停止状态的实例也会因其所保留的硬盘空间而产生少量计费。

如果较长时间内不会重新开启实例，右击图11.16中的示例，选择“Image” $\rightarrow$ “Create”创建镜像。然后，选择“Instance State” $\rightarrow$ “Terminate”将实例终结（硬盘不再产生计费）。当下次使用时，我们可按本节中创建并运行EC2实例的步骤重新创建一个基于保存镜像的实例。唯一的区别在于，在图11.10的第一步“1. Chosse AMI”中，我们需要通过左栏“My AMIs”选择之前保存的镜像。这样创建的实例将保留镜像上硬盘的存储（例如无需再安装CUDA和其他运行环境）。

## 小结

* 我们可以通过云计算服务获取更强大的计算资源来运行本书中的深度学习代码。

## 练习

* 云很方便，但不便宜。研究下它的价格，和看看如何节省开销。


## 扫码直达[讨论区](https://discuss.gluon.ai/t/topic/6154)

![](../img/qr_aws.svg)
