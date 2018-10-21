---
layout:     post
title:       Pytorch tutorials -training a classifier
subtitle:   
date:       2018-10-21
author:     Frank
header-img: img/post-bg-ios9-web.jpg
catalog: true
tags:
    - pytorch
    - deep learning
---


# Pytorch tutorials -training a classifier

> 一般来说当我们在做深度学习训练时处理数据时，需要将原始的数据转换成numpy数组然后在转换成为**torch.*tensor** 
>
> - 对于图片类型，可以使用Pillow,opencv,
> - 对于语音类型，可以使用scipy 和 librosa
> - 对于文本类型，可以使用raw python， NLTK等等

特别对于图片训练，torch有专门的包：torchvision 。其中包含一些常见的数据集比如：imagenet，cifar10，mnist 等等的data loaders。可以减轻我们的工作量。torchvision.datasets,torch.utils.data.DataLoader.

本文将使用CIFAR10这个数据集，其中有个10个类别的图片，每个图片大小是 **3x32x32**，即3个channel 32x32 pixels

![mark](http://pgxpob3x4.bkt.clouddn.com/blog/181021/D3d5B06llk.png?imageslim)



## Training an image classifier

我们将遵循下面的pipeline：

1. 使用torchvision来载入CIFAR10的数据
2. 定义神经网络
3. 定义损失函数
4. 在训练集上train这个网络
5. 在测试集上测试这个网络

### 1. 载入并正则化数据

载入库

```python
import torch
import torchvision
import torchvision.transforms as transforms
```



**由于torchvision的datasets的输出是[0,1]的PILImage，所以我们先先归一化为[-1,1]的Tensor**

```python
#首先定义一个transform变换，利用transform中的compose（）可以将多个变换组合在一起。可以看出来是to_tensor和normalize两个模块。
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
#载入训练集，可以看出使用torchvision定义好的类来实例化的，其中参数看名字应该都能够看懂作用。
trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
#此处trainloader是比较重要的，其作用就是往网络里面load数据。是torch的dataloader组件。从后面的参数可以看出来一些具体的设置。比如batch_size。
trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
                                          shuffle=True, num_workers=2)
#类似上面训练集
testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=4,
                                         shuffle=False, num_workers=2)
# 类别信息
classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
```

显示数据集图片

```python
import matplotlib.pyplot as plt
import numpy as np

# functions to show an image


def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))


# get some random training images
dataiter = iter(trainloader)
images, labels = dataiter.next()

# show images
imshow(torchvision.utils.make_grid(images))
# print labels
print(' '.join('%5s' % classes[labels[j]] for j in range(4)))
```



![数据集](https://ws1.sinaimg.cn/large/70c6b9dely1fwfvazp86xj20d203yglw.jpg)

## 定义神经网络

```python
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):#继承nn.Moudle创建新子类
  
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)#添加第一个卷积层
        self.pool = nn.MaxPool2d(2, 2)#添加池化层
        self.conv2 = nn.Conv2d(6, 16, 5)#卷积层
        self.fc1 = nn.Linear(16 * 5 * 5, 120)#全连接层
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):#定义前向传播方法
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)#改变tensor的形状
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


net = Net()#实例化
```

> **torch.nn中大多数layer在torch.nn.funtional中都有一个与之对应的函数。二者的区别在于：
> torch.nn.Module中实现layer的都是一个特殊的类，可以去查阅，他们都是以class xxxx来定义的，会自动提取可学习的参数
> 而nn.functional中的函数，更像是纯函数，由def function( )定义，只是进行简单的数学运算而已。
> 说到这里你可能就明白二者的区别了，functional中的函数是一个确定的不变的运算公式，输入数据产生输出就ok，
> 而深度学习中会有很多权重是在不断更新的，不可能每进行一次forward就用新的权重重新来定义一遍函数来进行计算，所以说就会采用类的方式，以确保能在参数发生变化时仍能使用我们之前定好的运算步骤。
> 所以从这个分析就可以看出什么时候改用nn.Module中的layer了：
> 如果模型有可学习的参数，最好使用nn.Module对应的相关layer，否则二者都可以使用，没有什么区别。
> 比如此例中的Relu其实没有可学习的参数，只是进行一个运算而已，所以使用的就是functional中的relu函数，
> 而卷积层和全连接层都有可学习的参数，所以用的是nn.Module中的类。
> 不具备可学习参数的层，将它们用函数代替，这样可以不用放在构造函数中进行初始化。**

定义网络模型，主要是用到了torch.nn和torch.nn.functional这两个模块。


## 定义损失函数和优化器

```python
import torch.optim as optim

criterion = nn.CrossEntropyLoss()#使用交叉熵作为loss function
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)#sgd作为优化器
```

## 训练

```python
for epoch in range(2):  # loop over the dataset multiple times 指定训练一共要循环几个epoch
 
    running_loss = 0.0  #定义一个变量方便我们对loss进行输出
    for i, data in enumerate(trainloader, 0): # 这里我们遇到了第一步中出现的trailoader，代码传入数据
                                              # enumerate是python的内置函数，既获得索引也获得数据，详见下文
        # get the inputs
        inputs, labels = data   # data是从enumerate返回的data，包含数据和标签信息，分别赋值给inputs和labels
 
        # wrap them in Variable
        inputs, labels = Variable(inputs), Variable(labels) # 将数据转换成Variable，第二步里面我们已经引入这个模块
                                                            # 所以这段程序里面就直接使用了，下文会分析
        # zero the parameter gradients
        optimizer.zero_grad()                # 要把梯度重新归零，因为反向传播过程中梯度会累加上一次循环的梯度
 
        # forward + backward + optimize      
        outputs = net(inputs)                # 把数据输进网络net，这个net()在第二步的代码最后一行我们已经定义了
        loss = criterion(outputs, labels)    # 计算损失值,criterion我们在第三步里面定义了
        loss.backward()                      # loss进行反向传播，下文详解
        optimizer.step()                     # 当执行反向传播之后，把优化器的参数进行更新，以便进行下一轮
 
        # print statistics                   # 这几行代码不是必须的，为了打印出loss方便我们看而已，不影响训练过程
        running_loss += loss.data[0]         # 从下面一行代码可以看出它是每循环0-1999共两千次才打印一次
        if i % 2000 == 1999:    # print every 2000 mini-batches   所以每个2000次之类先用running_loss进行累加
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 2000))  # 然后再除以2000，就得到这两千次的平均损失值
            running_loss = 0.0               # 这一个2000次结束后，就把running_loss归零，下一个2000次继续使用
 
print('Finished Training')

```

我们在定义网络的时候定义了forward的一些操作，但是没有定义反向的操作，而深度学习是需要反向操作来求导的。其实是利用**pytorch中的AutoGrad模块来自动求导，反向传播**.

Autograd中最核心的类就是Variable了，它封装了Tensor，并几乎支持所有Tensor的操作，这里可以参考官方给的详细解释：
http://pytorch.org/tutorials/beginner/blitz/autograd_tutorial.html#sphx-glr-beginner-blitz-autograd-tutorial-py
以上链接详细讲述了variable究竟是怎么能够实现自动求导的，怎么用它来实现反向传播的。
这里涉及到计算图的相关概念，这里我不详细讲，后面会写相关博文来讨论这个东西，暂时不会对我们理解这个程序造成影响
只说一句，想要计算各个variable的梯度，只需调用根节点的backward方法，Autograd就会自动沿着整个计算图进行反向计算
而在此例子中，根节点就是我们的loss，所以：

程序中的loss.backward()代码就是在实现反向传播，自动计算所有的梯度。

所以训练部分的代码其实比较简单：
running_loss和后面负责打印损失值的那部分并不是必须的，所以关键行不多，总得来说分成三小节
第一节：把最开始放在trainloader里面的数据给转换成variable，然后指定为网络的输入；
第二节：每次循环新开始的时候，要确保梯度归零
第三节：forward+backward，就是调用我们在第三步里面实例化的net()实现前传，loss.backward()实现后传

每结束一次循环，要确保梯度更新

## 测试

```python
dataiter = iter(testloader)      # 创建一个python迭代器，读入的是我们第一步里面就已经加载好的testloader
images, labels = dataiter.next() # 返回一个batch_size的图片，根据第一步的设置，应该是4张
 
# print images
imshow(torchvision.utils.make_grid(images))  # 展示这四张图片
print('GroundTruth: ', ' '.join('%5s' % classes[labels[j]] for j in range(4))) # python字符串格式化 ' '.join表示用空格来连接后面的字符串，参考python的join（）方法

```



```python
outputs = net(Variable(images))      # 注意这里的images是我们从上面获得的那四张图片，所以首先要转化成variable
_, predicted = torch.max(outputs.data, 1)  
                # 这个 _ , predicted是python的一种常用的写法，表示后面的函数其实会返回两个值
                # 但是我们对第一个值不感兴趣，就写个_在那里，把它赋值给_就好，我们只关心第二个值predicted
                # 比如 _ ,a = 1,2 这中赋值语句在python中是可以通过的，你只关心后面的等式中的第二个位置的值是多少
 
print('Predicted: ', ' '.join('%5s' % classes[predicted[j]] for j in range(4)))  # python的字符串格式化

```

每一类的正确率

```python
class_correct = list(0. for i in range(10)) # 定义一个存储每类中测试正确的个数的 列表，初始化为0
class_total = list(0. for i in range(10))   # 定义一个存储每类中测试总数的个数的 列表，初始化为0
for data in testloader:     # 以一个batch为单位进行循环
    images, labels = data
    outputs = net(Variable(images))
    _, predicted = torch.max(outputs.data, 1)
    c = (predicted == labels).squeeze()
    for i in range(4):      # 因为每个batch都有4张图片，所以还需要一个4的小循环
        label = labels[i]   # 对各个类的进行各自累加
        class_correct[label] += c[i]
        class_total[label] += 1
 
 
for i in range(10):
    print('Accuracy of %5s : %2d %%' % (
        classes[i], 100 * class_correct[i] / class_total[i]))

```

