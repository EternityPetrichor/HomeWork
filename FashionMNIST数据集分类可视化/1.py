import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
import torch.optim as optim
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader

import torchvision
import torchvision.transforms as transforms

import matplotlib.pyplot as plt
from IPython import display
import numpy as np

lr = 0.15
gamma = 0
epochs = 10
bs = 128

mnist = torchvision.datasets.FashionMNIST(
    root = "./data"
    , train=True # 使用训练数据集
    , download=True 
    , transform=transforms.ToTensor() # 将数据转换为Tensor
    )

batchdata = DataLoader(mnist, batch_size=bs, shuffle=True)

input_ = mnist.data[0].numel() # 查看一个样本共有多少个特征
output_ = len(mnist.targets.unique())


class Model(nn.Module):
    
    def __init__(self, in_features=3, out_features=10):
        super(Model, self).__init__()
        self.linear1 = nn.Linear(in_features, 128, bias=False)
        self.output = nn.Linear(128, out_features, bias=False)

    def forward(self, x):
        x = x.view(-1, 28*28)
        sigma1 = torch.relu(self.linear1(x))
        z_hat = self.output(sigma1)
        # log_sigma2 = torch.log_softmax(z_hat, dim=1)
        return z_hat
    

def fit(net, batchdata, lr=0.01, gamma=0.9, epochs=5):
    criterion = CrossEntropyLoss()
    opt = optim.SGD(net.parameters(), lr=lr, momentum=gamma)
    correct = 0 
    samples = 0

    for i_epoch in range(epochs):
        for i_batch, (xb, yb) in enumerate(batchdata):
            yb = yb.view(xb.shape[0])
            opt.zero_grad() # 清空梯度
            z_hat = net.forward(xb)
            loss = criterion(z_hat, yb)
            loss.backward()
            opt.step()

            # 计算准确率
            y_hat = torch.max(z_hat, dim=1)[1] 
            # softmax/logsoftmax函数对z_hat是单调递增的，因此对比z_hat的值也可以获得y_hat 
            correct += torch.sum(y_hat==yb)
            samples += xb.shape[0]

            if (i_batch+1) % 125 == 0 or i_batch == len(batchdata)-1:
                print("Epoch{}: [{}/{}({:.0f}%)] \t Loss: {:.6f} \t Accuracy:{:.3f}".format(
                       i_epoch+1
                       , samples
                       , len(batchdata.dataset)*epochs
                       , 100*samples/(len(batchdata.dataset)*epochs)
                       , loss.data.item()
                       , float(100*correct/samples)
                       )
                      )

torch.manual_seed(531)
net = Model(input_, output_)
fit(net, batchdata, lr=lr, gamma=gamma, epochs=epochs)
