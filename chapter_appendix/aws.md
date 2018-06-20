# 使用AWS运行代码

当本地机器的计算资源有限时，我们可以通过云计算服务获取更强大的计算资源来运行本书中的深度学习代码。本节将介绍如何在AWS（亚马逊的云计算服务）上申请GPU实例并通过Jupyter notebook运行代码。



## 申请账号并登陆

首先，我们需要在 https://aws.amazon.com/ 网站上创建账号。这通常需要一张信用卡。需要注意的是，AWS中国需要公司实体才能注册。如果你是个人用户，请注册AWS全球账号。

登陆AWS账号后，点击图11.8红框中的“EC2”进入EC2面板。

![登陆AWS账号。](../img/aws.png)


## 选择并运行EC2实例

图11.9展示了EC2面板的界面。在图11.9右上角红框处选择离我们较近的数据中心来减低延迟。我们可以选离国内较近的亚太地区，例如Asia Pacific（Seoul）。注意，有些数据中心可能没有GPU实例。点击图11.9下方红框内“Launch Instance”启动实例。

![EC2面板。](../img/ec2.png)


图11.10的最上面一行显示了配置实例所需的7个步骤。在第一步“1. Chosse AMI”中，选择Ubuntu 16.04作为操作系统。

![选择操作系统。](../img/os.png)

EC2提供了大量的有着不同配置的实例。如图11.11所示，在第二步“2. Chosse Instance Type”中，我们选择了有一个K80 GPU的“p2.xlarge”实例。我们也可以选择例如“p2.16xlarge”的有更多GPU的实例、有更新GPU的“g3”系列实例，或者像“c4”系列的只含CPU的实例。如果你想比较不同实例的机器配置和收费，可参考 https://www.ec2instances.info/ 。

![选择实例。](../img/p2x.png)


我们建议在选择实例前在图11.9左栏“Limits”里检查下有无数量限制。如图11.12所示，该账号的限制是最多在一个区域开一个“p2.xlarge”实例。如果需要开更多实例，可以通过点击右边“Request limit increase”来申请更大的实例容量。这通常需要一个工作日来处理。

![实例的数量限制。](../img/limits.png)

我们将保持第三步“3. Configure Instance”、第五步“5. Add Tags”和第六步“6. Configure Security Group”中的默认配置。点击第四步“4.Add Storage”，如图11.13所示，将默认的硬盘大小增大到40GB。注意，安装CUDA需要4GB左右空间。

![修改实例的硬盘大小。](../img/disk.png)


最后，在第七步“7. Review”中点击“Launch”来启动配置好的实例。这时候会提示我们选择用来访问实例的密钥。如果没有的话，可以选择图11.14中第一个下拉菜单的“Create a new key pair”选项来生成秘钥。之后，我们通过该下拉菜单的“Choose an existing key pair”选项选择生成好的密钥。点击“Launch Instance”。

![选择密钥。](../img/keypair.png)

点击图11.15中的实例ID就可以查看该实例的状态了。

![点击实例ID。](../img/launching.png)

如图11.16所示，当实例状态（Instance State）变绿后，右击实例并选择“Connect”，这时就可以看到访问该实例的方法了。例如在命令行输入

```
ssh -i "/path/to/key.pem" ubuntu@ec2-xx-xxx-xxx-xxx.y.compute.amazonaws.com
```

其中“/path/to/key.pem”是本地存放访问实例的密钥的路径。当命令行提示“Are you sure you want to continue connecting (yes/no)”时，键入“yes”并按回车键即可登录实例。

![查看访问开启实例的方法。](../img/connect.png)


## 安装CUDA

如果你登录的是一个GPU实例，需要下载并安装CUDA。首先，更新并安装编译需要的包：

```
sudo apt-get update && sudo apt-get install -y build-essential git libgfortran3
```

然后，访问Nvidia官网下载并安装CUDA。选择正确的版本并获取下载地址。

【注意】目前CUDA默认下载9.0版，但`mxnet-cu90`的daily build还不完善。建议使用下面命令安装8.0版。

![](../img/cuda.png)

然后使用`wget`下载并且安装

```
wget https://developer.nvidia.com/compute/cuda/8.0/Prod2/local_installers/cuda_8.0.61_375.26_linux-run
sudo sh cuda_8.0.61_375.26_linux-run
```

这里需要回答几个问题。

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

安装完成后运行

```
nvidia-smi
```

就可以看到这个实例的GPU了。最后将CUDA加入到library path方便之后安装的库找到它。

```
echo "export LD_LIBRARY_PATH=\${LD_LIBRARY_PATH}:/usr/local/cuda-8.0/lib64" >>.bashrc
```

### 安装MXNet

先安装Miniconda

```
wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh
```

需要回答下面几个问题

```
Do you accept the license terms? [yes|no]
[no] >>> yes
Do you wish the installer to prepend the Miniconda3 install location
to PATH in your /home/ubuntu/.bashrc ? [yes|no]
[no] >>> yes
```

运行一次`bash`让CUDA和conda生效。

下载本教程，安装并激活conda环境

```
git clone https://github.com/mli/gluon-tutorials-zh
cd gluon-tutorials-zh
conda env create -f environment.yml
source activate gluon
```

默认环境里安装了只有CPU的版本。现在我们替换成GPU版本。

```
pip uninstall -y mxnet
pip install --pre mxnet-cu80

```

同时安装notedown插件来让jupter读写markdown文件。

```
pip install https://github.com/mli/notedown/tarball/master
jupyter notebook --generate-config
echo "c.NotebookApp.contents_manager_class = 'notedown.NotedownContentsManager'" >>~/.jupyter/jupyter_notebook_config.py

```

## 运行

并运行Jupyter notebook。

```
jupyter notebook
```

如果成功的话会看到类似的输出

![](../img/jupyter.png)

因为我们的实例没有暴露8888端口，所以我们可以在本地启动ssh从实例映射到本地

```
ssh -i "XXX.pem" -L8888:locallhost:8888 ubuntu@XXXX.XXXX.compute.amazonaws.com
```

然后把jupyter log里的URL复制到本地浏览器就行了。

【注意】如果本地运行了Jupyter notebook，那么8888端口就可能被占用了。要么关掉本地jupyter，要么把端口映射改成别的。例如，假设aws使用默认8888端口，我们可以在本地启动ssh从实例映射到本地8889端口：

```
ssh -i "XXX.pem" -N -f -L localhost:8889:localhost:8888 ubuntu@XXXX.XXXX.compute.amazonaws.com
```

然后在本地浏览器打开localhost:8889，这时会提示需要token值。接下来，我们将aws上jupyter log里的token值（例如上图里：...localhost:8888/?token=`token值`）复制粘贴即可。



## 后续

因为云服务按时间计费，通常我们不用时需要把样例关掉，到下次要用时再开。如果是停掉（Stop)，下次可以直接继续用，但硬盘空间会计费。如果是终结(Termination)，我们一般会先把操作系统做镜像，下次开始时直接使用镜像（AMI）（上面的教程使用了Ubuntu 16.06 AMI）就行了，不需要再把上面流程走一次。

![](../img/ami.png)

每次重新开始后，我们建议升级下教程（记得保存自己的改动）

```
cd gluon-tutorials-zh
git pull
```

和MXNet版本

```
source activate gluon
pip install -U --pre mxnet-cu80
```

## 小结

* 云上可以很方便的获取计算资源和配置环境。

## 练习

* 云很方便，但不便宜。研究下它的价格，和看看如何节省开销。


## 扫码直达[讨论区](https://discuss.gluon.ai/t/topic/6154)

![](../img/qr_aws.svg)
