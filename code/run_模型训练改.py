# -*- coding: utf-8 -*-
# @Time : 2024/11/5 15:27
# @Author : Tao
# @Email : 3195858080@qq.com
# @File : run_模型训练改.py

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import random
import csv
import time
from datetime import datetime
from collections import Counter
import os

# 定义事件日志文件名
eventlog = "helpdesk.csv"


# 数据预处理
lines = []  # 存储所有活动序列
timeseqs = []  # 存储每个事件之间的时间差
timeseqs2 = []  # 存储每个事件与第一个事件之间的时间差

# 辅助变量
lastcase = ''  # 上一个案例的 ID
line = ''  # 当前案例的活动序列
firstLine = True  # 是否是第一行
times = []  # 当前案例的时间差序列
times2 = []  # 当前案例与第一个事件的时间差序列
numlines = 0  # 案例总数
casestarttime = None  # 当前案例的开始时间
lasteventtime = None  # 上一个事件的时间

# 打开 CSV 文件
csvfile = open('../data/%s' % eventlog, 'r')
spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')
next(spamreader, None)  # 跳过标题行
ascii_offset = 161  # ASCII 偏移量，用于将活动 ID 转换为字符
for activity_id in range(1, 10):
    char = chr(activity_id + ascii_offset)
    print(f"活动 ID {activity_id} 对应的字符: {char}")

# 遍历每一行数据
for row in spamreader:  # 行格式为 "CaseID,ActivityID,CompleteTimestamp"
    t = time.strptime(row[2], "%Y-%m-%d %H:%M:%S")  # 创建一个 datetime 对象
    if row[0] != lastcase:  # 如果当前案例 ID 与上一个案例 ID 不同
        casestarttime = t  # 更新当前案例的开始时间
        lasteventtime = t  # 更新上一个事件的时间
        lastcase = row[0]  # 更新上一个案例 ID
        if not firstLine:
            lines.append(line)  # 将当前案例的活动序列添加到列表中
            timeseqs.append(times)  # 将当前案例的时间差序列添加到列表中
            timeseqs2.append(times2)  # 将当前案例与第一个事件的时间差序列添加到列表中
        line = ''  # 重置当前案例的活动序列
        times = []  # 重置当前案例的时间差序列
        times2 = []  # 重置当前案例与第一个事件的时间差序列
        numlines += 1  # 增加案例计数
    line += chr(int(row[1]) + ascii_offset)  # 将活动 ID 转换为字符并添加到当前案例的活动序列中
    timesincelastevent = datetime.fromtimestamp(time.mktime(t)) - datetime.fromtimestamp(time.mktime(lasteventtime))
    timesincecasestart = datetime.fromtimestamp(time.mktime(t)) - datetime.fromtimestamp(time.mktime(casestarttime))
    timediff = 86400 * timesincelastevent.days + timesincelastevent.seconds  # 计算当前事件与上一个事件的时间差
    timediff2 = 86400 * timesincecasestart.days + timesincecasestart.seconds  # 计算当前事件与第一个事件的时间差
    times.append(timediff)  # 将时间差添加到当前案例的时间差序列中
    times2.append(timediff2)  # 将时间差添加到当前案例与第一个事件的时间差序列中
    lasteventtime = t  # 更新上一个事件的时间
    firstLine = False  # 标记不再是最第一行

# 添加最后一个案例
lines.append(line)  # 将最后一个案例的活动序列添加到列表中
timeseqs.append(times)  # 将最后一个案例的时间差序列添加到列表中
timeseqs2.append(times2)  # 将最后一个案例与第一个事件的时间差序列添加到列表中
numlines += 1  # 增加案例计数

# 计算平均时间
divisor = np.mean([item for sublist in timeseqs for item in sublist])  # 计算平均时间间隔
print('divisor: {}'.format(divisor))
divisor2 = np.mean([item for sublist in timeseqs2 for item in sublist])  # 计算平均时间间隔（当前事件与第一个事件之间）
print('divisor2: {}'.format(divisor2))

# 分割数据集
elems_per_fold = int(round(numlines / 3))  # 每个折叠的数据量
fold1 = lines[:elems_per_fold]  # 第一个折叠的活动序列
fold1_t = timeseqs[:elems_per_fold]  # 第一个折叠的时间差序列
fold1_t2 = timeseqs2[:elems_per_fold]  # 第一个折叠与第一个事件的时间差序列

fold2 = lines[elems_per_fold:2 * elems_per_fold]  # 第二个折叠的活动序列
fold2_t = timeseqs[elems_per_fold:2 * elems_per_fold]  # 第二个折叠的时间差序列
fold2_t2 = timeseqs2[elems_per_fold:2 * elems_per_fold]  # 第二个折叠与第一个事件的时间差序列

fold3 = lines[2 * elems_per_fold:]  # 第三个折叠的活动序列
fold3_t = timeseqs[2 * elems_per_fold:]  # 第三个折叠的时间差序列
fold3_t2 = timeseqs2[2 * elems_per_fold:]  # 第三个折叠与第一个事件的时间差序列

# 合并 fold1 和 fold2 的数据
lines = fold1 + fold2  # 合并活动序列
lines_t = fold1_t + fold2_t  # 合并时间差序列
lines_t2 = fold1_t2 + fold2_t2  # 合并与第一个事件的时间差序列

# 获取所有可能的字符并编号
lines = [x + '!' for x in lines]  # 在每个活动序列末尾添加分隔符
maxlen = max([len(x) for x in lines])  # 找到最大行长度
print(lines)
chars = set(''.join(lines))  # 获取所有可能的字符
#chars.discard('!')  # 移除分隔符
target_chars = chars.copy()  # 目标字符集
print('total chars: {}, target chars: {}'.format(len(chars), len(target_chars)))

char_indices = {c: i for i, c in enumerate(chars)}  # 字符到索引的映射
indices_char = {i: c for i, c in enumerate(chars)}  # 索引到字符的映射
target_char_indices = {c: i for i, c in enumerate(target_chars)}  # 目标字符到索引的映射
target_indices_char = {i: c for i, c in enumerate(target_chars)}  # 索引到目标字符的映射


# 数据集类
class EventLogDataset(Dataset):
    def __init__(self, lines, lines_t, lines_t2, maxlen, divisor, divisor2, char_indices, target_char_indices):
        self.lines = lines  # 活动序列
        self.lines_t = lines_t  # 时间差序列
        self.lines_t2 = lines_t2  # 与第一个事件的时间差序列
        self.maxlen = maxlen  # 最大行长度
        self.divisor = divisor  # 平均时间间隔
        self.divisor2 = divisor2  # 平均时间间隔（当前事件与第一个事件之间）
        self.char_indices = char_indices  # 字符到索引的映射
        self.target_char_indices = target_char_indices  # 目标字符到索引的映射

    def __len__(self):
        return len(self.lines)  # 返回数据集的大小

    def __getitem__(self, idx):
        line = self.lines[idx]  # 当前案例的活动序列
        line_t = self.lines_t[idx]  # 当前案例的时间差序列
        line_t2 = self.lines_t2[idx]  # 当前案例与第一个事件的时间差序列

        X = np.zeros((self.maxlen, len(chars) + 5), dtype=np.float32)  # 初始化输入张量
        y_a = np.zeros((len(target_chars)), dtype=np.float32)  # 初始化目标活动标签
        y_t = np.zeros((1), dtype=np.float32)  # 初始化目标时间差标签

        # 遍历当前案例的活动序列
        for t, char in enumerate(line[:-1]):
            X[t, self.char_indices[char]] = 1  # 设置当前字符的位置为 1
            X[t, len(chars)] = line_t[t] / self.divisor  # 归一化时间差
            X[t, len(chars) + 1] = line_t2[t] / self.divisor2  # 归一化与第一个事件的时间差
            X[t, len(chars) + 2] = 1  # 临时占位符
            X[t, len(chars) + 3] = 1  # 临时占位符
            X[t, len(chars) + 4] = 1  # 临时占位符

        y_a[self.target_char_indices[line[-1]]] = 1  # 设置目标活动标签
        y_t[0] = line_t[-1] / self.divisor  # 设置目标时间差标签

        return torch.tensor(X, dtype=torch.float32), torch.tensor(y_a, dtype=torch.float32), torch.tensor(y_t, dtype=torch.float32)

# 创建数据集和数据加载器
dataset = EventLogDataset(lines, lines_t, lines_t2, maxlen, divisor, divisor2, char_indices, target_char_indices)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)  # 创建数据加载器

# 定义模型
class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super(LSTMModel, self).__init__()
        self.hidden_dim = hidden_dim  # 隐藏层维度
        self.num_layers = num_layers  # LSTM 层数
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)  # LSTM 层
        self.fc1 = nn.Linear(hidden_dim, output_dim)  # 全连接层，用于预测下一个活动
        self.fc2 = nn.Linear(hidden_dim, 1)  # 全连接层，用于预测时间差
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)  # 初始化隐藏状态
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)  # 初始化细胞状态
        out, _ = self.lstm(x, (h0, c0))  # 前向传播
        out = out[:, -1, :]  # 取最后一个时间步的输出
        y_a = self.fc1(out)  # 预测下一个活动
        y_t = self.fc2(out)  # 预测时间差
        return y_a, y_t

# 定义模型参数
input_dim = len(chars) + 5  # 输入维度
hidden_dim = 128  # 隐藏层维度
output_dim = len(target_chars)  # 输出维度
num_layers = 2  # LSTM 层数

# 创建模型实例
model = LSTMModel(input_dim, hidden_dim, output_dim, num_layers)
model = model.cuda()  # 如果有 GPU 支持
path='output_files/models'
# 定义损失函数和优化器
criterion1 = nn.CrossEntropyLoss()  # 交叉熵损失，用于分类任务
criterion2 = nn.MSELoss()  # 均方误差损失，用于回归任务
optimizer = optim.Adam(model.parameters(), lr=0.001)  # Adam 优化器

# 训练模型
num_epochs = 500  # 训练轮数
best_loss = float('inf')  # 初始化最佳损失为无穷大
for epoch in range(num_epochs):
    running_loss = 0.0  # 用于记录当前 epoch 的总损失
    for i, (inputs, labels_a, labels_t) in enumerate(dataloader):
        inputs = inputs.cuda()  # 将输入数据移到 GPU
        labels_a = labels_a.cuda()  # 将目标活动标签移到 GPU
        labels_t = labels_t.cuda()  # 将目标时间差标签移到 GPU
        optimizer.zero_grad()  # 清零梯度
        outputs_a, outputs_t = model(inputs)  # 前向传播
        loss_a = criterion1(outputs_a, labels_a.argmax(dim=1))  # 计算活动预测的损失
        loss_t = criterion2(outputs_t.squeeze(), labels_t.squeeze())  # 计算时间差预测的损失
        loss = loss_a + loss_t  # 总损失
        loss.backward()  # 反向传播
        optimizer.step()  # 更新权重
        running_loss += loss.item()  # 累加损
        if (i + 1) % 10 == 0:
            print(f'Epoch [{epoch + 1}/{num_epochs}], Step [{i + 1}/{len(dataloader)}], Loss: {loss.item():.4f}')  # 打印训练进度
        # 计算平均损失
    avg_loss = running_loss / len(dataloader)
    # 如果当前 epoch 的损失小于最佳损失，保存模型
    if avg_loss < best_loss:
        best_loss = avg_loss
        save_path = os.path.join(path, 'best_model.pth')
        torch.save(model.state_dict(), save_path)
        print(f'Saved best model with loss: {best_loss:.4f} to {save_path}')
print('Training complete')  # 训练完成