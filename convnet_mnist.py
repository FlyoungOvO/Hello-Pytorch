import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import _utils
import torchvision.datasets as datasets
import torchvision.transforms as transforms

#定义一些超参数
BATCH_SIZE = 512
EPOCHS = 20
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#定义数据预处理
train_dataset = datasets.MNIST(root='dataset/',train=True,transform = 
                transforms.Compose([transforms.ToTensor(),
                transforms.Normalize((0.1307,),(0.3081,))]),
                download = True)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,batch_size=BATCH_SIZE,shuffle=True)

test_dataset = datasets.MNIST(root='dataset/',train=False,transform =
                transforms.Compose([transforms.ToTensor(),
                transforms.Normalize((0.1307,),(0.3081,))]),
                download = True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset,batch_size=BATCH_SIZE,shuffle=True)

#定义网络
class ConvNet(nn.Module):
    def __init__(self):
        super().__init__()
        #batch*1*28*28(每次传入batch个样本，输入通道1(灰度图)，分辨率28*28)
        #下面的卷积层Conv2d的第一个参数指输入通道数，第二个参数指输出通道数，第三个参数指卷积核大小
        self.conv1 = nn.Conv2d(1,10,5)
        self.conv2 = nn.Conv2d(10,20,3)
        #全连接层Linear的第一个参数指输入特征数，第二个参数指输出特征数
        self.fc1 = nn.Linear(20*10*10,500)
        self.fc2 = nn.Linear(500,10) #输出通道数为10，因为MNIST数据集有10个类别

    #正向传播
    def forward(self,x):
        in_size = x.size(0) #BATCH_SIZE,输入的x的形状为(BATCH_SIZE,1,28,28)
        out = self.conv1(x) #batch*10*28*28 -> batch*10*24*24(经过卷积核5*5，步长为1，填充为0)
        out = F.relu(out)   #batch*10*24*24 -> batch*10*24*24(经过激活函数ReLU，不改变形状)
        out = F.max_pool2d(out,2,2) #batch*10*24*24 -> batch*10*12*12(经过池化层，池化核2*2，步长为2，减半)
        out = self.conv2(out) #batch*20*12*12 -> batch*20*10*10(经过卷积核3*3，步长为1，填充为0)
        out = F.relu(out) #经过激活函数ReLU，不改变形状
        out = out.view(in_size,-1) #将out展平为一维向量，-1表示自动计算维度

        out = self.fc1(out) #batch*20*10*10 -> batch*500
        out = F.relu(out)   
        out = self.fc2(out) #batch*500 -> batch*10
        out = F.log_softmax(out,dim=1) #计算log(softmax(x))，dim=1表示按行计算

        return out


model = ConvNet().to(DEVICE) #将模型放到GPU上
optimizer = optim.Adam(model.parameters()) #定义优化器

def train(model,device,train_loader,optimizer,epoch):
    model.train()
    for batch_idx,(data,target) in enumerate(train_loader):
        data,target = data.to(device),target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output,target)
        loss.backward()
        optimizer.step()
        if(batch_idx+1)%30 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch,batch_idx*len(data),len(train_loader.dataset),
                100.*batch_idx/len(train_loader),loss.item()))

def test(model,device,test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data,target in test_loader:
            data,target = data.to(device),target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output,target,reduction='sum').item() #将一批的损失相加
            pred = output.argmax(dim=1,keepdim=True) #获取概率最大的下标
            correct += pred.eq(target.view_as(pred)).sum().item() #统计预测正确的个数

    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss,correct,len(test_loader.dataset),
        100.*correct/len(test_loader.dataset)))

for epoch in range(1,EPOCHS+1):
    train(model,DEVICE,train_loader,optimizer,epoch)
    test(model,DEVICE,test_loader)
