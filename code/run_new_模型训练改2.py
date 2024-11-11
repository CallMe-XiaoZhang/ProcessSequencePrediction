# -*- coding: utf-8 -*-
# @Time : 2024/11/8 09:35
# @Author : Tao
# @Email : 3195858080@qq.com
# @File : run_new_模型训练改2.py

import os
import time
import torch
import torch.nn as nn
import numpy as np
import datetime
from datetime import datetime
import csv
from sklearn.metrics import mean_absolute_error
import distance

# 设置设备：如果有 GPU，则使用 GPU，否则使用 CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Damerau-Levenshtein 距离
def damerau_levenshtein_distance(s1, s2):
    lenstr1 = len(s1)
    lenstr2 = len(s2)
    d = [[0 for _ in range(lenstr2 + 1)] for _ in range(lenstr1 + 1)]
    for i in range(lenstr1 + 1):
        d[i][0] = i
    for j in range(lenstr2 + 1):
        d[0][j] = j
    for i in range(1, lenstr1 + 1):
        for j in range(1, lenstr2 + 1):
            if s1[i - 1] == s2[j - 1]:
                cost = 0
            else:
                cost = 1
            d[i][j] = min(
                d[i - 1][j] + 1,      # deletion
                d[i][j - 1] + 1,      # insertion
                d[i - 1][j - 1] + cost,  # substitution
            )
            if i > 1 and j > 1 and s1[i - 1] == s2[j - 2] and s1[i - 2] == s2[j - 1]:
                d[i][j] = min(d[i][j], d[i - 2][j - 2] + cost)  # transposition
    return d[lenstr1][lenstr2]

# 打开 CSV 文件
eventlog = "helpdesk.csv"
csvfile = open(f'../data/{eventlog}', 'r')
spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')
next(spamreader, None)  # 跳过标题行
ascii_offset = 161  # ASCII 偏移量，用于将活动 ID 转换为字符
for activity_id in range(1, 10):
    char = chr(activity_id + ascii_offset)
    print(f"活动 ID {activity_id} 对应的字符: {char}")

# 定义辅助变量
lastcase = ''  # 上一个案例的 ID
line = ''  # 当前案例的活动序列
firstLine = True  # 是否是第一行
lines = []  # 存储所有活动序列
caseids = []  # 存储所有案例 ID
timeseqs = []  # 存储每个事件之间的时间差
timeseqs2 = []  # 存储每个事件与第一个事件之间的时间差
times = []  # 当前案例的时间差序列（相对于上一个事件）
times2 = []  # 当前案例的时间差序列（相对于案例开始）
times3 = []  # 当前案例的绝对时间序列
timeseqs3 = []  # 存储每个事件的绝对时间
numlines = 0  # 案例总数
casestarttime = None  # 当前案例的开始时间
lasteventtime = None  # 上一个事件的时间

# 遍历每一行数据
for row in spamreader:  # 行格式为 "CaseID,ActivityID,CompleteTimestamp"
    t = time.strptime(row[2], "%Y-%m-%d %H:%M:%S")  # 创建一个 datetime 对象
    if row[0] != lastcase:  # 如果当前案例 ID 与上一个案例 ID 不同
        caseids.append(row[0])
        casestarttime = t  # 更新当前案例的开始时间
        lasteventtime = t  # 更新上一个事件的时间
        lastcase = row[0]  # 更新上一个案例 ID
        if not firstLine:
            lines.append(line)  # 将当前案例的活动序列添加到列表中
            timeseqs.append(times)  # 将当前案例的时间差序列添加到列表中
            timeseqs2.append(times2)  # 将当前案例与第一个事件的时间差序列添加到列表中
            timeseqs3.append(times3)  # 将当前案例的绝对时间序列添加到列表中
        line = ''  # 重置当前案例的活动序列
        times = []  # 重置当前案例的时间差序列
        times2 = []  # 重置当前案例与第一个事件的时间差序列
        times3 = []  # 重置当前案例的绝对时间序列
        numlines += 1  # 增加案例计数
    line += chr(int(row[1]) + ascii_offset)  # 将活动 ID 转换为字符并添加到当前案例的活动序列中
    timesincelastevent = datetime.fromtimestamp(time.mktime(t)) - datetime.fromtimestamp(time.mktime(lasteventtime))
    timesincecasestart = datetime.fromtimestamp(time.mktime(t)) - datetime.fromtimestamp(time.mktime(casestarttime))
    timediff = 86400 * timesincelastevent.days + timesincelastevent.seconds  # 计算当前事件与上一个事件的时间差
    timediff2 = 86400 * timesincecasestart.days + timesincecasestart.seconds  # 计算当前事件与第一个事件的时间差
    times.append(timediff)  # 将时间差添加到当前案例的时间差序列中
    times2.append(timediff2)  # 将时间差添加到当前案例与第一个事件的时间差序列中
    times3.append(datetime.fromtimestamp(time.mktime(t)))
    lasteventtime = t  # 更新上一个事件的时间
    firstLine = False  # 标记不再是最第一行

# 添加最后一个案例
lines.append(line)  # 将最后一个案例的活动序列添加到列表中
timeseqs.append(times)  # 将最后一个案例的时间差序列添加到列表中
timeseqs2.append(times2)  # 将最后一个案例与第一个事件的时间差序列添加到列表中
numlines += 1  # 增加案例计数

# 计算平均时间
divisor = np.mean([item for sublist in timeseqs for item in sublist])  # 计算平均时间间隔
divisor2 = np.mean([item for sublist in timeseqs2 for item in sublist])  # 计算平均时间间隔（当前事件与第一个事件之间）
divisor3 = np.mean(list(map(lambda x: np.mean(list(map(lambda y: x[len(x)-1] - y, x))), timeseqs2)))# divisor3 是每个轨迹中的时间差与最后一个时间差的差值的平均

# 将数据分为三折
elems_per_fold = int(round(numlines / 3))  # 每个折叠的数据量
fold1 = lines[:elems_per_fold]  # 第一个折叠的活动序列
fold1_c = caseids[:elems_per_fold]  # 第一个折叠的案例 ID
fold1_t = timeseqs[:elems_per_fold]  # 第一个折叠的时间差序列（相对于上一个事件）
fold1_t2 = timeseqs2[:elems_per_fold]  # 第一个折叠的时间差序列（相对于案例开始）
fold1_t3 = timeseqs3[:elems_per_fold]  # 提取第3个折叠的绝对时间序列

fold2 = lines[elems_per_fold:2 * elems_per_fold]  # 第二个折叠的活动序列
fold2_c = caseids[elems_per_fold:2 * elems_per_fold]  # 第二个折叠的案例 ID
fold2_t = timeseqs[elems_per_fold:2 * elems_per_fold]  # 第二个折叠的时间差序列（相对于上一个事件）
fold2_t2 = timeseqs2[elems_per_fold:2 * elems_per_fold]  # 第二个折叠的时间差序列（相对于案例开始）
fold2_t3 = timeseqs3[elems_per_fold:2 * elems_per_fold]  # 第二个折叠的时间差序列（相对于案例开始）


lines = fold1 + fold2  # 合并前两折的活动序列
caseids = fold1_c + fold2_c  # 合并前两折的案例 ID
lines_t = fold1_t + fold2_t  # 合并前两折的时间差序列（相对于上一个事件）
lines_t2 = fold1_t2 + fold2_t2  # 合并前两折的时间差序列（相对于案例开始）
lines_t3 = fold1_t3 + fold2_t3

# 获取所有可能的字符并编号
lines = [x + '!' for x in lines]  # 在每个活动序列末尾添加分隔符
maxlen = max([len(x) for x in lines])  # 找到最大行长度
chars = set(''.join(lines))  # 获取所有可能的字符
target_chars = chars.copy()  # 目标字符集
char_indices = {c: i for i, c in enumerate(chars)}  # 字符到索引的映射
indices_char = {i: c for i, c in enumerate(chars)}  # 索引到字符的映射
target_char_indices = {c: i for i, c in enumerate(target_chars)}  # 目标字符到索引的映射
target_indices_char = {i: c for i, c in enumerate(target_chars)}  # 索引到目标字符的映射

predict_size=1

# 定义模型
class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super(LSTMModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc1 = nn.Linear(hidden_dim, output_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = out[:, -1, :]
        y_a = self.fc1(out)
        y_t = self.fc2(out)
        return y_a, y_t

# 定义模型参数
input_dim = len(chars) + 5  # 输入维度
hidden_dim = 128  # 隐藏层维度
output_dim = len(target_chars)  # 输出维度
num_layers = 2  # LSTM 层数
model = LSTMModel(input_dim, hidden_dim, output_dim, num_layers)
model = model.to(device)  # 将模型移动到设备（CPU或GPU）

# 定义辅助函数
def encode(sentence, times, times3, maxlen=maxlen):
    num_features = len(chars) + 5  # 特征数量：字符特征 + 位置特征 + 时间差特征 + 累积时间差特征 + 时间差（相对于午夜） + 星期几
    X = np.zeros((1, maxlen, num_features), dtype=np.float32)  # 初始化张量
    leftpad = maxlen - len(sentence)  # 计算需要填充的零的数量
    times2 = np.cumsum(times)  # 计算累积时间差
    for t, char in enumerate(sentence):
        midnight = times3[t].replace(hour=0, minute=0, second=0, microsecond=0)  # 当天的午夜时间
        timesincemidnight = times3[t] - midnight  # 当前事件与当天午夜的时间差
        for c in chars:
            if c == char:
                X[0, t + leftpad, char_indices[c]] = 1  # 将字符特征编码为 one-hot 向量
        X[0, t + leftpad, len(chars)] = t + 1  # 事件的位置特征,在案例中处于第几个事件
        X[0, t + leftpad, len(chars) + 1] = times[t] / divisor  # 事件与上一个事件的时间差（归一化）
        X[0, t + leftpad, len(chars) + 2] = times2[t] / divisor2  # 事件与案例开始的时间差（归一化）
        X[0, t + leftpad, len(chars) + 3] = timesincemidnight.total_seconds() / 86400  # 事件与当天午夜的时间差（归一化）
        X[0, t + leftpad, len(chars) + 4] = times3[t].weekday() / 7  # 事件发生的星期几（归一化）
    return torch.tensor(X, dtype=torch.float32)  # 返回 PyTorch 张量

def getSymbol(prediction):
    symbol = target_indices_char[prediction]
    return symbol

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# 训练模型
epochs = 100
patience = 5  # 早停机制的耐心参数
best_loss = float('inf')
no_improvement_count = 0
best_model_weights = None

# 定义训练和验证的函数
def train_model(model, data, epochs, patience, device, criterion, optimizer):
    best_loss = float('inf')
    no_improvement_count = 0
    best_model_weights = None
    for epoch in range(epochs):
        total_loss = 0
        for line, caseid, times, times3 in zip(data['lines'], data['caseids'], data['lines_t'], data['lines_t3']):
            times.append(0)  # 在时间差列表末尾添加 0
            for prefix_size in range(2, len(line) - predict_size + 1):
                cropped_line = line[:prefix_size]  # 裁剪活动序列到当前前缀长度
                cropped_times = times[:prefix_size]  # 裁剪时间差序列到当前前缀长度
                cropped_times3 = times3[:prefix_size]  # 裁剪绝对时间序列到当前前缀长度
                print(cropped_line)
                print(cropped_times)

                # 编码当前的裁剪数据
                enc = encode(cropped_line, cropped_times, cropped_times3)
                enc_tensor = enc.clone().detach().float().to(device)

                # 获取目标张量
                label = torch.tensor(target_char_indices[line[prefix_size]], dtype=torch.long).to(device)  # 下一个活动的标签
                time_label = torch.tensor([cropped_times[-1] / divisor], dtype=torch.float).to(device)  # 时间差的标签
                print(label,time_label)
                # 计算损失
                optimizer.zero_grad()
                y_a, y_t = model(enc_tensor.unsqueeze(0))  # 进行预测
                loss_a = criterion(y_a, label)  # 活动预测损失
                loss_t = criterion(y_t, time_label)  # 时间差预测损失
                loss = loss_a + loss_t  # 总损失
                loss.backward()  # 反向传播
                optimizer.step()  # 更新参数
                total_loss += loss.item()

        average_loss = total_loss / len(data['lines'])
        print(f'Epoch {epoch+1}, Loss: {average_loss}')

        # 如果找到更好的模型，则更新最佳损失和模型权重
        if average_loss < best_loss:
            best_loss = average_loss
            best_model_weights = model.state_dict()
            no_improvement_count = 0
        else:
            no_improvement_count += 1

        # 如果连续多次没有改进，则停止训练
        if no_improvement_count >= patience:
            print(f'Early stopping triggered after {patience} epochs without improvement.')
            break
    # 加载最佳模型权重
    model.load_state_dict(best_model_weights)
    return model

# 准备数据字典
data = {
    'lines': lines,
    'caseids': caseids,
    'lines_t': lines_t,
    'lines_t3': lines_t3
}


# 训练模型
trained_model = train_model(model, data, epochs=1000, patience=500, device=device, criterion=criterion, optimizer=optimizer)

# 保存最佳模型
torch.save(trained_model.state_dict(), 'best_model3.pth')
print('Best model saved successfully')