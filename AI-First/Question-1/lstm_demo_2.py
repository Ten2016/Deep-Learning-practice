import numpy as np
import torch
from torch import nn,optim
from torch.autograd import Variable
from torchvision import datasets,transforms
from torch.utils.data import DataLoader


# 训练集
train_data = datasets.MNIST(root="./", # 存放位置
                            train = True, # 载入训练集
                            transform=transforms.ToTensor(), # 把数据变成tensor类型
                            download = True # 下载
                           )
# 测试集
test_data = datasets.MNIST(root="./",
                            train = False,
                            transform=transforms.ToTensor(),
                            download = True
                           )

# 批次大小
batch_size = 64
# 装载训练集
train_loader = DataLoader(dataset=train_data,batch_size=batch_size,shuffle=True)
# 装载测试集
test_loader = DataLoader(dataset=test_data,batch_size=batch_size,shuffle=True)


for i,data in enumerate(train_loader):
    inputs,labels = data
    print(inputs.shape)
    print(labels.shape)
    break


# 定义网络结构
class LSTM(nn.Module):
    def __init__(self):
        super(LSTM,self).__init__()# 初始化
        self.lstm = torch.nn.LSTM(
            input_size = 28, # 表示输入特征的大小
            hidden_size = 64, # 表示lstm模块的数量
            num_layers = 1, # 表示lstm隐藏层的层数
            batch_first = True # lstm默认格式input（seq_len,batch,feature）等于True表示input和output变成（batch，seq_len，feature）
        )
        self.out = torch.nn.Linear(in_features=64,out_features=10)
        self.softmax = torch.nn.Softmax(dim=1)
    def forward(self,x):
        # (batch,seq_len,feature)
        x = x.view(-1,28,28)
        # output:(batch,seq_len,hidden_size)包含每个序列的输出结果
        # 虽然lstm的batch_first为True，但是h_n,c_n的第0个维度还是num_layers
        # h_n :[num_layers,batch,hidden_size]只包含最后一个序列的输出结果
        # c_n:[num_layers,batch,hidden_size]只包含最后一个序列的输出结果
        output,(h_n,c_n) = self.lstm(x)
        output_in_last_timestep = h_n[-1,:,:]
        x = self.out(output_in_last_timestep)
        x = self.softmax(x)
        return x


# 定义模型
model = LSTM()
# 定义代价函数
mse_loss = nn.CrossEntropyLoss()# 交叉熵
# 定义优化器
optimizer = optim.Adam(model.parameters(),lr=0.001)# 随机梯度下降


# 定义模型训练和测试的方法
def train():
    # 模型的训练状态
    model.train()
    for i,data in enumerate(train_loader):
        # 获得一个批次的数据和标签
        inputs,labels = data
        # 获得模型预测结果（64，10)
        out = model(inputs)
        # 交叉熵代价函数out（batch，C：类别的数量），labels（batch）
        loss = mse_loss(out,labels)
        # 梯度清零
        optimizer.zero_grad()
        # 计算梯度
        loss.backward()
        # 修改权值
        optimizer.step()
        
def test():
    # 模型的测试状态
    model.eval()
    correct = 0 # 测试集准确率
    for i,data in enumerate(test_loader):
        # 获得一个批次的数据和标签
        inputs,labels = data
        # 获得模型预测结果（64，10)
        out = model(inputs)
        # 获得最大值，以及最大值所在的位置
        _,predicted = torch.max(out,1)
        # 预测正确的数量
        correct += (predicted==labels).sum()
    print("Test acc:{0}".format(correct.item()/len(test_data)))
    
    correct = 0
    for i,data in enumerate(train_loader): # 训练集准确率
        # 获得一个批次的数据和标签
        inputs,labels = data
        # 获得模型预测结果（64，10)
        out = model(inputs)
        # 获得最大值，以及最大值所在的位置
        _,predicted = torch.max(out,1)
        # 预测正确的数量
        correct += (predicted==labels).sum()
    print("Train acc:{0}".format(correct.item()/len(train_data)))


# 训练
for epoch in range(10):
    print("epoch:",epoch)
    train()
    test()