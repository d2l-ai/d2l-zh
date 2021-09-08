# Using AWS EC2 Instances
:label:`sec_aws`

In this section, we will show you how to install all libraries on a raw Linux machine. Remember that in :numref:`sec_sagemaker` we discussed how to use Amazon SageMaker, while building an instance by yourself costs less on AWS. The walkthrough includes a number of steps:

1. Request for a GPU Linux instance from AWS EC2.
1. Optionally: install CUDA or use an AMI with CUDA preinstalled.
1. Set up the corresponding MXNet GPU version.

This process applies to other instances (and other clouds), too, albeit with some minor modifications. Before going forward, you need to create an AWS account, see :numref:`sec_sagemaker` for more details.


## Creating and Running an EC2 Instance

After logging into your AWS account, click "EC2" (marked by the red box in :numref:`fig_aws`) to go to the EC2 panel.

![Open the EC2 console.](../img/aws.png)
:width:`400px`
:label:`fig_aws`

:numref:`fig_ec2` shows the EC2 panel with sensitive account information greyed out.

![EC2 panel.](../img/ec2.png)
:width:`700px`
:label:`fig_ec2`

### Presetting Location
Select a nearby data center to reduce latency, e.g., "Oregon" (marked by the red box in the top-right of :numref:`fig_ec2`). If you are located in China,
you can select a nearby Asia Pacific region, such as Seoul or Tokyo. Please note
that some data centers may not have GPU instances.

### Increasing Limits
Before choosing an instance, check if there are quantity
restrictions by clicking the "Limits" label in the bar on the left as shown in
:numref:`fig_ec2`. :numref:`fig_limits` shows an example of such a
limitation. The account currently cannot open "p2.xlarge" instance per region. If
you need to open one or more instances, click on the "Request limit increase" link to
apply for a higher instance quota. Generally, it takes one business day to
process an application.

![Instance quantity restrictions.](../img/limits.png)
:width:`700px`
:label:`fig_limits`

### Launching Instance
Next, click the "Launch Instance" button marked by the red box in :numref:`fig_ec2` to launch your instance.

We begin by selecting a suitable AMI (AWS Machine Image). Enter "Ubuntu" in the search box (marked by the red box in :numref:`fig_ubuntu`).


![Choose an operating system.](../img/ubuntu-new.png)
:width:`700px`
:label:`fig_ubuntu`

EC2 provides many different instance configurations to choose from. This can sometimes feel overwhelming to a beginner. Here's a table of suitable machines:

| Name | GPU         | Notes                         |
|------|-------------|-------------------------------|
| g2   | Grid K520   | ancient                       |
| p2   | Kepler K80  | old but often cheap as spot   |
| g3   | Maxwell M60 | good trade-off                |
| p3   | Volta V100  | high performance for FP16     |
| g4   | Turing T4   | inference optimized FP16/INT8 |

All the above servers come in multiple flavors indicating the number of GPUs used. For example, a p2.xlarge has 1 GPU and a p2.16xlarge has 16 GPUs and more memory. For more details, see the [AWS EC2 documentation](https://aws.amazon.com/ec2/instance-types/) or a [summary page](https://www.ec2instances.info). For the purpose of illustration, a p2.xlarge will suffice (marked in red box of :numref:`fig_p2x`).

**Note:** you must use a GPU enabled instance with suitable drivers and a version of MXNet that is GPU enabled. Otherwise you will not see any benefit from using GPUs.

![Choose an instance.](../img/p2x.png)
:width:`700px`
:label:`fig_p2x`

So far, we have finished the first two of seven steps for launching an EC2 instance, as shown on the top of :numref:`fig_disk`. In this example, we keep the default configurations for the steps "3. Configure Instance", "5. Add Tags", and "6. Configure Security Group". Tap on "4. Add Storage" and increase the default hard disk size to 64 GB (marked in red box of :numref:`fig_disk`). Note that CUDA by itself already takes up 4 GB.

![Modify instance hard disk size.](../img/disk.png)
:width:`700px`
:label:`fig_disk`

Finally, go to "7. Review" and click "Launch" to launch the configured
instance. The system will now prompt you to select the key pair used to access
the instance. If you do not have a key pair, select "Create a new key pair" in
the first drop-down menu in :numref:`fig_keypair` to generate a key pair. Subsequently,
you can select "Choose an existing key pair" for this menu and then select the
previously generated key pair. Click "Launch Instances" to launch the created
instance.

![Select a key pair.](../img/keypair.png)
:width:`500px`
:label:`fig_keypair`

Make sure that you download the key pair and store it in a safe location if you
generated a new one. This is your only way to SSH into the server. Click the
instance ID shown in :numref:`fig_launching` to view the status of this instance.

![Click the instance ID.](../img/launching.png)
:width:`700px`
:label:`fig_launching`

### Connecting to the Instance

As shown in :numref:`fig_connect`, after the instance state turns green, right-click the instance and select `Connect` to view the instance access method.

![View instance access and startup method.](../img/connect.png)
:width:`700px`
:label:`fig_connect`

If this is a new key, it must not be publicly viewable for SSH to work. Go to the folder where you store `D2L_key.pem` (e.g., the Downloads folder) and make sure that the key is not publicly viewable.

```bash
cd /Downloads  ## if D2L_key.pem is stored in Downloads folder
chmod 400 D2L_key.pem
```


![View instance access and startup method.](../img/chmod.png)
:width:`400px`
:label:`fig_chmod`


Now, copy the ssh command in the lower red box of :numref:`fig_chmod` and paste onto the command line:

```bash
ssh -i "D2L_key.pem" ubuntu@ec2-xx-xxx-xxx-xxx.y.compute.amazonaws.com
```


When the command line prompts "Are you sure you want to continue connecting (yes/no)", enter "yes" and press Enter to log into the instance.

Your server is ready now.


## Installing CUDA

Before installing CUDA, be sure to update the instance with the latest drivers.

```bash
sudo apt-get update && sudo apt-get install -y build-essential git libgfortran3
```


Here we download CUDA 10.1. Visit NVIDIA's [official repository](https://developer.nvidia.com/cuda-downloads) to find the download link of CUDA 10.1 as shown in :numref:`fig_cuda`.

![Find the CUDA 10.1 download address.](../img/cuda101.png)
:width:`500px`
:label:`fig_cuda`

Copy the instructions and paste them into the terminal to install
CUDA 10.1.

```bash
## Paste the copied link from CUDA website
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/cuda-ubuntu1804.pin
sudo mv cuda-ubuntu1804.pin /etc/apt/preferences.d/cuda-repository-pin-600
wget http://developer.download.nvidia.com/compute/cuda/10.1/Prod/local_installers/cuda-repo-ubuntu1804-10-1-local-10.1.243-418.87.00_1.0-1_amd64.deb
sudo dpkg -i cuda-repo-ubuntu1804-10-1-local-10.1.243-418.87.00_1.0-1_amd64.deb
sudo apt-key add /var/cuda-repo-10-1-local-10.1.243-418.87.00/7fa2af80.pub
sudo apt-get update
sudo apt-get -y install cuda
```


After installing the program, run the following command to view the GPUs.

```bash
nvidia-smi
```


Finally, add CUDA to the library path to help other libraries find it.

```bash
echo "export LD_LIBRARY_PATH=\${LD_LIBRARY_PATH}:/usr/local/cuda/lib64" >> ~/.bashrc
```


## Installing MXNet and Downloading the D2L Notebooks

First, to simplify the installation, you need to install [Miniconda](https://conda.io/en/latest/miniconda.html) for Linux. The download link and file name are subject to changes, so please go the Miniconda website and click "Copy Link Address" as shown in :numref:`fig_miniconda`.

![Download Miniconda.](../img/miniconda.png)
:width:`700px`
:label:`fig_miniconda`

```bash
# The link and file name are subject to changes
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
sh Miniconda3-latest-Linux-x86_64.sh -b
```


After the Miniconda installation, run the following command to activate CUDA and conda.

```bash
~/miniconda3/bin/conda init
source ~/.bashrc
```


Next, download the code for this book.

```bash
sudo apt-get install unzip
mkdir d2l-en && cd d2l-en
curl https://d2l.ai/d2l-en.zip -o d2l-en.zip
unzip d2l-en.zip && rm d2l-en.zip
```


Then create the conda `d2l` environment and enter `y` to proceed with the installation.

```bash
conda create --name d2l -y
```


After creating the `d2l` environment, activate it and install `pip`.

```bash
conda activate d2l
conda install python=3.7 pip -y
```


Finally, install MXNet and the `d2l` package. The postfix `cu101` means that this is the CUDA 10.1 variant. For different versions, say only CUDA 10.0, you would want to choose `cu100` instead.

```bash
pip install mxnet-cu101==1.7.0
pip install git+https://github.com/d2l-ai/d2l-en

```


You can quickly test whether everything went well as follows:

```
$ python
>>> from mxnet import np, npx
>>> np.zeros((1024, 1024), ctx=npx.gpu())
```


## Running Jupyter

To run Jupyter remotely you need to use SSH port forwarding. After all, the server in the cloud does not have a monitor or keyboard. For this, log into your server from your desktop (or laptop) as follows.

```
# This command must be run in the local command line
ssh -i "/path/to/key.pem" ubuntu@ec2-xx-xxx-xxx-xxx.y.compute.amazonaws.com -L 8889:localhost:8888
conda activate d2l
jupyter notebook
```


:numref:`fig_jupyter` shows the possible output after you run Jupyter Notebook. The last row is the URL for port 8888.

![Output after running Jupyter Notebook. The last row is the URL for port 8888.](../img/jupyter.png)
:width:`700px`
:label:`fig_jupyter`

Since you used port forwarding to port 8889 you will need to replace the port number and use the secret as given by Jupyter when opening the URL in your local browser.


## Closing Unused Instances

As cloud services are billed by the time of use, you should close instances that are not being used. Note that there are alternatives: "stopping" an instance means that you will be able to start it again. This is akin to switching off the power for your regular server. However, stopped instances will still be billed a small amount for the hard disk space retained. "Terminate" deletes all data associated with it. This includes the disk, hence you cannot start it again. Only do this if you know that you will not need it in the future.

If you want to use the instance as a template for many more instances,
right-click on the example in :numref:`fig_connect` and select "Image" $\rightarrow$
"Create" to create an image of the instance. Once this is complete, select
"Instance State" $\rightarrow$ "Terminate" to terminate the instance. The next
time you want to use this instance, you can follow the steps for creating and
running an EC2 instance described in this section to create an instance based on
the saved image. The only difference is that, in "1. Choose AMI" shown in
:numref:`fig_ubuntu`, you must use the "My AMIs" option on the left to select your saved
image. The created instance will retain the information stored on the image hard
disk. For example, you will not have to reinstall CUDA and other runtime
environments.


## Summary

* You can launch and stop instances on demand without having to buy and build your own computer.
* You need to install suitable GPU drivers before you can use them.


## Exercises

1. The cloud offers convenience, but it does not come cheap. Find out how to launch [spot instances](https://aws.amazon.com/ec2/spot/) to see how to reduce prices.
1. Experiment with different GPU servers. How fast are they?
1. Experiment with multi-GPU servers. How well can you scale things up?


[Discussions](https://discuss.d2l.ai/t/423)
