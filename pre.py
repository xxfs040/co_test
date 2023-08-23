import matplotlib.pyplot as plt
import torch
import pandas as pd
import numpy as np
from datetime import datetime
import time

#加载数据集
data = pd.read_csv('data.csv', header=0, usecols=[0,1])

#归一化
seq_train = data.iloc[:908,1]
dataset = seq_train.values
dataset = dataset.astype('float32')
max_value = np.max(dataset)
min_value = np.min(dataset)
scalar = max_value - min_value
dataset = list(map(lambda x : x / scalar, dataset))

#设置超参数
look_back = 62 #根据前31个数据预测接下来的数据
epoch = 495000#可适度调整
input_size = 62
hidden_size = 4
batch = 1


#创建数据集
def create_dataset(dataset, look_back):
    dataX, dataY = [], []
    for i in range(len(dataset) - look_back):
        a = dataset[i:(i + look_back)]
        dataX.append(a)
        dataY.append(dataset[i + look_back])
    return np.array(dataX), np.array(dataY)



# 创建好输入输出
data_X, data_Y = create_dataset(dataset, look_back)


#改变数据形状
train_X = data_X.reshape(-1, batch, look_back)
train_Y = data_Y.reshape(-1, batch, 1)



train_x = torch.from_numpy(train_X)
train_y = torch.from_numpy(train_Y)



#定义模型
from torch import nn
from torch.autograd import Variable


class lstm_reg(nn.Module):
    def __init__(self, input_size, hidden_size, output_size=1, num_layers=2):
        super(lstm_reg, self).__init__()
        self.rnn = nn.LSTM(input_size, hidden_size, num_layers)  # rnn
        self.reg = nn.Linear(hidden_size, output_size)  # 回归

    def forward(self, x):
        x, _ = self.rnn(x)  # (seq, batch, hidden)
        s, b, h = x.shape
        x = x.view(s * b, h)  # 转换成线性层的输入格式
        x = self.reg(x)
        x = x.view(s, b, -1)
        return x

net = lstm_reg(input_size, hidden_size)

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)
# 开始训练

#训练
for e in range(epoch):
    var_x = Variable(train_x)
    var_y = Variable(train_y)
    # 前向传播
    out = net(var_x)
    loss = criterion(out, var_y)
    # 反向传播
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if (e + 1) % 100 == 0: # 每 100 次输出结果
        print(f"Epoch: {e + 1}, Loss: {loss.data:.10f}")


#预测
pre = data.iloc[876:, 1]
pre = pre.values
pre = pre.astype('float32')
max = np.max(pre)
min = np.min(pre)
pre_scalar = max - min
pre_dataset = list(map(lambda x: x / scalar, pre))

pr_set = []
pr_set.append(pre_dataset)
pr_set = np.array(pr_set)

result = [] #预测结果
for i in range(31):
    pre_result = []
    net = net.eval()  # 转换成测试模式
    if i == 0:
        set = pr_set.reshape(-1, 1, 62)
    else:
        set  = set.reshape(-1, 1, 62)
    for j in range(100):
        pre_set = torch.Tensor(set)
        var_data = Variable(pre_set)
        pre_result = net(var_data)  # 测试集的预测结果
        pre_result = pre_result.view(-1).data.numpy()
    member = sum(pre_result) / len(pre_result) #取预测结果的均值
    result.append(member)
    set = list(set)
    set = np.array(set).reshape(-1)
    set = set.tolist()
    set = set[1:]
    set.append(member)
    set = np.array(set)

for i in range(31):
    result[i] = result[i] * pre_scalar

print(result)







