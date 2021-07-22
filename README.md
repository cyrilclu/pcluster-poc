# ParallelCluster POC

## 安装ParallelCluster

### 安装软件包

```
yum install python3-pip
pip3 install awscli -U --user
pip3 install aws-parallelcluster -U --user
```

### 配置AWS Credentials

如果ParallelCluster安装在一台EC2上，只需要给EC2挂载IAM Role即可。
IAM Role需要具备相关的权限，使得ParallelCluster可以调度AWS上的计算资源。
具体权限可以参考下面的链接：

https://docs.aws.amazon.com/zh_cn/parallelcluster/latest/ug/iam.html#parallelclusteruserpolicy

为了测试方便，在下面环境中将使用AdministratorAccess权限（仅限测试环境使用，生产环境需要保持最小权限原则）。

如果不在EC2上安装ParallelCluster，例如在本地笔记本电脑上进行安装，则需要配置AWS AKSK，可以参考下面的链接：

https://docs.aws.amazon.com/zh_cn/cli/latest/userguide/cli-configure-quickstart.html#cli-configure-quickstart-config

### 准备ParallelCluster配置文件

配置文件示例见config.ini文件。
对于需要GPU进行模型训练的场景，我们需要使用自定义AMI，自定义AMI可以通过下面链接查询AMI ID。

https://github.com/aws/aws-parallelcluster/blob/v2.11.0/amis.txt

### 创建集群
```
~/.local/bin/pcluster create -c config.ini p-cluster
```

执行完成后，会返回Master节点的IP地址，可以通过config.ini中指定的Key Pair进行SSH连接。
```
ssh -i <Key Pari> ubuntu@<ip>
```

## 测试用例一

### 环境准备

```
cd /home/ubuntu
sudo apt-get install python3-venv
python3 -m venv uc1
source uc1/bin/activate
python3 -m pip install --upgrade pip
git clone https://github.com/dmis-lab/biobert.git
cd biobert
pip3 install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

这里可能会存在pandas版本依赖问题，解决办法是在requirements.txt里修改pandas版本。
pandas==0.23.4

用例一运行需要依赖cuda，但AMI里自带的cuda版本较新，与tensorflow版本不兼容，需要手动安装cuda。
```
cd /home/ubuntu
wget https://developer.nvidia.com/compute/cuda/10.0/Prod/local_installers/cuda_10.0.130_410.48_linux
sudo chmod +x cuda_10.0.130_410.48_linux
./cuda_10.0.130_410.48_linux
```
安装过程中不需要安装driver，将cuda安装在/home/ubuntu/cuda-10.0目录下。

tensorflow依赖的libcudnn库文件，默认不会安装，需要手动下载进行安装。
访问下面链接进行下载，如果没有Nvidia账号，需要自行注册：

https://developer.nvidia.com/rdp/cudnn-archive

在页面里点击该库的7版本中最新的小版本
Download cuDNN v7.6.5 (November 18th, 2019), for CUDA 10.2
点击cuDNN Runtime Library for Ubuntu18.04 (Deb)进行下载。
安装cuDNN：
```
sudo dpkg -i libcudnn7_7.6.5.32-1+cuda10.2_amd64.deb
sudo cp /usr/lib/x86_64-linux-gnu/libcudnn.so.7.6.5 /home/ubuntu/cuda-10.0/lib64/
cd /home/ubuntu/cuda-10.0/lib64/
ln -s libcudnn.so.7.6.5  libcudnn.so.7
```

至此，测试用例一中需要的软件环境已经准备完成。

### 运行测试用例一NER场景
下载pre-trained数据，BioBERT-Base v1.1 (+ PubMed 1M) 
下载链接可以在biobert的README中找到。
下载NER的DataSets，将其存放在/home/ubuntu目录下

准备执行任务的脚本，可以参考uc1-base-1.1-ner.sbatch文件。
执行NER任务：
```
sbatch uc1-base-1.1-ner.sbatch
```

### 运行测试用例一RE场景
下载RE的DataSets，将其存放在/home/ubuntu目录下
准备执行任务的脚本，可以参考uc1-base-1.1-ner.sbatch文件。
执行RE任务：
```
sbatch uc1-base-1.1-re.sbatch
```

## 测试用例二

### 环境准备
基于测试用例一的软件环境，测试用例二需要再额外安装一些软件包
```
cd /home/ubuntu
source uc1/bin/activate
pip3 install seaborn tqdm torch scanpy -i https://pypi.tuna.tsinghua.edu.cn/simple
mkdir -p uc2
```
### 运行测试用例二

将测试用例二的源代码解压在uc2这个目录。
修改源代码中的一些文件路径，使其与当前环境匹配。
准备执行任务的脚本，可以参考uc2.sbatch
执行任务：
```
sbatch uc2.sbatch
```
