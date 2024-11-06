'''
该脚本将通过训练找到的LSTM或RNN权重作为输入。py
将此脚本第176行中的路径更改为指向h5文件
模型训练生成LSTM或RNN权重。py

Author: Niek Tax
'''
# 导包
from __future__ import division
import os
import time
from collections import Counter
import torch.nn as nn
import torch
import numpy as np
import datetime as datetime2
from datetime import datetime
import csv
from sklearn.metrics import mean_absolute_error
import distance

# 设置设备：如果有 GPU，则使用 GPU，否则使用 CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# damerau_levenshtein_distance
def damerau_levenshtein_distance(s1, s2):
    """
    计算两个字符串之间的Damerau-Levenshtein距离（字符串的编辑距离）。
    """
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

# 测试函数
print(damerau_levenshtein_distance('abc', 'acb'))  # 输出应该是 1

# 打开 CSV 文件
eventlog = "helpdesk.csv"
csvfile = open('../data/%s' % eventlog, 'r')
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
numlines = 0  # 案例总数
casestarttime = None  # 当前案例的开始时间
lasteventtime = None  # 上一个事件的时间

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
divisor3 = np.mean(list(map(lambda x: np.mean(list(map(lambda y: x[len(x)-1] - y, x))), timeseqs2)))
print('divisor3: {}'.format(divisor3))# divisor3 是每个轨迹中的时间差与最后一个时间差的差值的平均

# 将数据分为三折
elems_per_fold = int(round(numlines / 3))  # 每个折叠的数据量
fold1 = lines[:elems_per_fold]  # 第一个折叠的活动序列
fold1_c = caseids[:elems_per_fold]  # 第一个折叠的案例 ID
fold1_t = timeseqs[:elems_per_fold]  # 第一个折叠的时间差序列（相对于上一个事件）
fold1_t2 = timeseqs2[:elems_per_fold]  # 第一个折叠的时间差序列（相对于案例开始）

fold2 = lines[elems_per_fold:2 * elems_per_fold]  # 第二个折叠的活动序列
fold2_c = caseids[elems_per_fold:2 * elems_per_fold]  # 第二个折叠的案例 ID
fold2_t = timeseqs[elems_per_fold:2 * elems_per_fold]  # 第二个折叠的时间差序列（相对于上一个事件）
fold2_t2 = timeseqs2[elems_per_fold:2 * elems_per_fold]  # 第二个折叠的时间差序列（相对于案例开始）

lines = fold1 + fold2  # 合并前两折的活动序列
caseids = fold1_c + fold2_c  # 合并前两折的案例 ID
lines_t = fold1_t + fold2_t  # 合并前两折的时间差序列（相对于上一个事件）
lines_t2 = fold1_t2 + fold2_t2  # 合并前两折的时间差序列（相对于案例开始）

# 获取所有可能的字符并编号
lines = [x + '!' for x in lines]  # 在每个活动序列末尾添加分隔符
maxlen = max([len(x) for x in lines])  # 找到最大行长度

chars = set(''.join(lines))  # 获取所有可能的字符
#chars.discard('!')  # 移除分隔符
target_chars = chars.copy()  # 目标字符集
print('total chars: {}, target chars: {}'.format(len(chars), len(target_chars)))

char_indices = {c: i for i, c in enumerate(chars)}  # 字符到索引的映射
indices_char = {i: c for i, c in enumerate(chars)}  # 索引到字符的映射
target_char_indices = {c: i for i, c in enumerate(target_chars)}  # 目标字符到索引的映射
target_indices_char = {i: c for i, c in enumerate(target_chars)}  # 索引到目标字符的映射


lastcase = ''  # 上一个案例的 ID
line = ''  # 当前案例的活动序列
firstLine = True  # 是否是第一行
lines = []  # 存储所有活动序列
caseids = []  # 存储所有案例 ID
timeseqs = []  # 存储每个事件之间的时间差（相对于上一个事件）
timeseqs2 = []  # 存储每个事件与案例开始的时间差
timeseqs3 = []  # 存储每个事件的绝对时间
times = []  # 当前案例的时间差序列（相对于上一个事件）
times2 = []  # 当前案例的时间差序列（相对于案例开始）
times3 = []  # 当前案例的绝对时间序列
numlines = 0  # 案例总数
casestarttime = None  # 当前案例的开始时间
lasteventtime = None  # 上一个事件的时间

# 打开 CSV 文件
csvfile = open('../data/%s' % eventlog, 'r')
spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')
next(spamreader, None)  # 跳过标题行

# 遍历每一行数据
for row in spamreader:
    t = time.strptime(row[2], "%Y-%m-%d %H:%M:%S")  # 将时间字符串转换为时间对象
    if row[0] != lastcase:  # 如果当前案例 ID 与上一个案例 ID 不同
        caseids.append(row[0])  # 记录当前案例 ID
        casestarttime = t  # 更新当前案例的开始时间
        lasteventtime = t  # 更新上一个事件的时间
        lastcase = row[0]  # 更新上一个案例 ID
        if not firstLine:  # 如果不是第一行
            lines.append(line)  # 将当前案例的活动序列添加到列表中
            timeseqs.append(times)  # 将当前案例的时间差序列（相对于上一个事件）添加到列表中
            timeseqs2.append(times2)  # 将当前案例的时间差序列（相对于案例开始）添加到列表中
            timeseqs3.append(times3)  # 将当前案例的绝对时间序列添加到列表中
        line = ''  # 重置当前案例的活动序列
        times = []  # 重置当前案例的时间差序列（相对于上一个事件）
        times2 = []  # 重置当前案例的时间差序列（相对于案例开始）
        times3 = []  # 重置当前案例的绝对时间序列
        numlines += 1  # 增加案例计数
    line += chr(int(row[1]) + ascii_offset)  # 将活动 ID 转换为字符并添加到当前案例的活动序列中
    # 计算时间差
    timesincelastevent = datetime.fromtimestamp(time.mktime(t)) - datetime.fromtimestamp(time.mktime(lasteventtime))  # 当前事件与上一个事件的时间差
    timesincecasestart = datetime.fromtimestamp(time.mktime(t)) - datetime.fromtimestamp(time.mktime(casestarttime))  # 当前事件与案例开始的时间差
    midnight = datetime.fromtimestamp(time.mktime(t)).replace(hour=0, minute=0, second=0, microsecond=0)  # 当天的午夜时间
    timesincemidnight = datetime.fromtimestamp(time.mktime(t)) - midnight  # 当前事件与当天午夜的时间差
    # 将时间差转换为秒
    timediff = 86400 * timesincelastevent.days + timesincelastevent.seconds
    timediff2 = 86400 * timesincecasestart.days + timesincecasestart.seconds
    # 将时间差和绝对时间添加到当前案例的时间序列中
    times.append(timediff)
    times2.append(timediff2)
    times3.append(datetime.fromtimestamp(time.mktime(t)))
    lasteventtime = t  # 更新上一个事件的时间
    firstLine = False  # 标记不再是最第一行

# 添加最后的案例数据
lines.append(line)  # 将最后一个案例的活动序列添加到 lines 列表中
timeseqs.append(times)  # 将最后一个案例的时间差序列（相对于上一个事件）添加到 timeseqs 列表中
timeseqs2.append(times2)  # 将最后一个案例的时间差序列（相对于案例开始）添加到 timeseqs2 列表中
timeseqs3.append(times3)  # 将最后一个案例的绝对时间序列添加到 timeseqs3 列表中
numlines += 1  # 增加案例计数

# 分割数据
# 假设 elem_per_fold 是每个折叠包含的元素数量
# 这里选择第3个折叠，即从 2*elems_per_fold 开始到最后的所有数据
fold3 = lines[2*elems_per_fold:]  # 提取第3个折叠的活动序列
fold3_c = caseids[2*elems_per_fold:]  # 提取第3个折叠的案例 ID
fold3_t = timeseqs[2*elems_per_fold:]  # 提取第3个折叠的时间差序列（相对于上一个事件）
fold3_t2 = timeseqs2[2*elems_per_fold:]  # 提取第3个折叠的时间差序列（相对于案例开始）
fold3_t3 = timeseqs3[2*elems_per_fold:]  # 提取第3个折叠的绝对时间序列
# fold3_t4 = timeseqs4[2*elems_per_fold:]  # 如果有其他时间序列，可以类似处理

# 将分割后的数据赋值给全局变量
lines = fold3  # 更新 lines 为第3个折叠的活动序列
caseids = fold3_c  # 更新 caseids 为第3个折叠的案例 ID
lines_t = fold3_t  # 更新 lines_t 为第3个折叠的时间差序列（相对于上一个事件）
lines_t2 = fold3_t2  # 更新 lines_t2 为第3个折叠的时间差序列（相对于案例开始）
lines_t3 = fold3_t3  # 更新 lines_t3 为第3个折叠的绝对时间序列
# lines_t4 = fold1_t4 + fold2_t4  # 如果有其他时间序列，可以类似处理

# 初始化预测大小
predict_size = 1  # 预测的步长为1

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
model = model.to(device) # 如果有 GPU 支持

# 加载最佳模型
save_dir = 'output_files/models'  # 指定保存模型的目录
save_path = os.path.join(save_dir, 'best_model.pth')
model.load_state_dict(torch.load(save_path))


# define helper functions
def encode(sentence, times, times3, maxlen=maxlen):  # 编码一个三维张量，以便输入到神经网络模型
    num_features = len(chars) + 5  # 特征数量：字符特征 + 位置特征 + 时间差特征 + 累积时间差特征 + 时间差（相对于午夜） + 星期几
    X = np.zeros((1,maxlen, num_features), dtype=np.float32)  # 初始化张量
    leftpad = maxlen - len(sentence)  # 计算需要填充的零的数量
    times2 = np.cumsum(times)  # 计算累积时间差
    for t, char in enumerate(sentence):
        midnight = times3[t].replace(hour=0, minute=0, second=0, microsecond=0)  # 当天的午夜时间
        timesincemidnight = times3[t] - midnight  # 当前事件与当天午夜的时间差
        for c in chars:
            if c == char:
                X[0,t + leftpad, char_indices[c]] = 1  # 将字符特征编码为 one-hot 向量
        X[0,t + leftpad, len(chars)] = t + 1  # 事件的位置特征,在案例中处于第几个事件
        X[0,t + leftpad, len(chars) + 1] = times[t] / divisor  # 事件与上一个事件的时间差（归一化）
        X[0,t + leftpad, len(chars) + 2] = times2[t] / divisor2  # 事件与案例开始的时间差（归一化）
        X[0,t + leftpad, len(chars) + 3] = timesincemidnight.total_seconds() / 86400  # 事件与当天午夜的时间差（归一化）
        X[0,t + leftpad, len(chars) + 4] = times3[t].weekday() / 7  # 事件发生的星期几（归一化）
    return torch.tensor(X, dtype=torch.float32)  # 返回 PyTorch 张量


def getSymbol(predictions):# 根据模型的预测结果返回最可能的符号（活动）
    maxPrediction = 0  # 当前最大预测概率
    symbol = ''  # 当前最可能的符号
    i = 0  # 索引变量
    for prediction in predictions:
        if prediction >= maxPrediction:
            maxPrediction = prediction
            symbol = target_indices_char[i]  # 更新最可能的符号
        i += 1
    return symbol

# 初始化用于存储实际值和预测值的列表
one_ahead_gt = []  # 存储一时间步的实际时间差
one_ahead_pred = []  # 存储一时间步的预测时间差

two_ahead_gt = []  # 存储二时间步的实际时间差
two_ahead_pred = []  # 存储二时间步的预测时间差

three_ahead_gt = []  # 存储三时间步的实际时间差
three_ahead_pred = []  # 存储三时间步的预测时间差

# 做预测
with open('output_files/results/suffix_and_remaining_time_%s' % eventlog, 'w',encoding='utf-8') as csvfile:
    spamwriter = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
    # 写入 CSV 文件的标题行
    spamwriter.writerow([
        "CaseID",
        "Prefix length",
        "Ground truth",
        "Predicted",
        "Levenshtein",
        "Damerau",
        "Jaccard",
        "Ground truth times",
        "Predicted times",
        "RMSE",
        "MAE"
    ])

    # 遍历不同的前缀长度
    for prefix_size in range(2, maxlen):
        print(prefix_size)
        # 遍历每个案例
        for line, caseid, times, times3 in zip(lines, caseids, lines_t, lines_t3):
            times.append(0)  # 在时间差列表末尾添加 0
            cropped_line = ''.join(line[:prefix_size])  # 裁剪活动序列到当前前缀长度
            cropped_times = times[:prefix_size]  # 裁剪时间差序列到当前前缀长度
            cropped_times3 = times3[:prefix_size]  # 裁剪绝对时间序列到当前前缀长度
            # 如果裁剪后的活动序列中包含 '!'，表示该案例已经结束，跳过该案例
            if '!' in cropped_line:
                continue
            # 获取后续真实值
            ground_truth = ''.join(line[prefix_size:prefix_size + predict_size])
            ground_truth_t = times[prefix_size:prefix_size + predict_size]
            # 初始化预测值
            predicted = ''
            predicted_t = []
            # 预测循环
            for i in range(predict_size):
                if len(ground_truth) <= i:
                    continue  # 如果后续真实值不足，跳过本次预测
                # 编码当前的裁剪数据
                enc = encode(cropped_line, cropped_times, cropped_times3, maxlen)
                # enc = np.expand_dims(enc, axis=0)  # 扩展为 (1, maxlen, num_features)
                # 转换为 PyTorch 张量并移动到模型的设备上（CPU 或 GPU）
                enc_tensor = enc.clone().detach().float().to(device)
                # 使用模型进行预测
                with torch.no_grad():  # 禁用梯度计算，进行预测时通常不需要梯度
                    model.eval()  # 切换到评估模式
                    y = model(enc_tensor)  # 进行预测
                # 获取预测的活动和时间
                y_char = y[0][0]  # 获取活动预测值
                y_t = y[1][0][0].item()  # 获取时间差预测值
                # 获取预测的活动
                prediction = getSymbol(y_char)
                # 将预测的活动添加到裁剪后的活动序列中
                cropped_line += prediction
                # 处理负的时间差，确保其非负
                if y_t < 0:
                    y_t = 0
                # 更新时间差序列
                cropped_times.append(y_t)
                # 将预测的时间差转换为秒，并更新绝对时间序列
                y_t = y_t * divisor
                cropped_times3.append(cropped_times3[-1] + datetime2.timedelta(seconds=y_t))
                # 将预测的时间差添加到预测时间差列表中
                predicted_t.append(y_t)
                # 如果预测的活动为 '!'，表示案例结束，停止预测
                if prediction == '!':
                    print('! predicted, end case')
                    break
                # 将预测的活动添加到预测字符串中
                predicted += prediction
            # 初始化输出列表
            output = []

            # 如果后续真实值不为空
            if len(ground_truth) > 0:
                output.append(caseid)  # 案例 ID
                output.append(prefix_size)  # 前缀长度
                output.append(ground_truth)  # 地面真实值
                output.append(predicted)  # 预测值
                output.append(1 - distance.nlevenshtein(predicted, ground_truth))  # Levenshtein 距离
                dls = 1 - (damerau_levenshtein_distance(predicted, ground_truth) / max(len(predicted),                                                                   len(ground_truth)))
                if dls < 0:
                    dls = 0  # 确保 Damerau-Levenshtein 相似度非负
                output.append(dls)  # Damerau-Levenshtein 相似度
                output.append(1 - distance.jaccard(predicted, ground_truth))  # Jaccard 相似度
                output.append('; '.join(str(x) for x in ground_truth_t))  # 后续真实时间差
                output.append('; '.join(str(x) for x in predicted_t))  # 预测时间差
                # 如果预测的时间差比后续真实时间差多，只使用需要的事件
                if len(predicted_t) > len(ground_truth_t):
                    predicted_t = predicted_t[:len(ground_truth_t)]
                # 如果预测的时间差比后续真实时间差少，用 0 作为占位符
                if len(ground_truth_t) > len(predicted_t):
                    predicted_t.extend([0] * (len(ground_truth_t) - len(predicted_t)))
                # 计算并记录 MAE
                if len(ground_truth_t) > 0 and len(predicted_t) > 0:
                    output.append(mean_absolute_error(ground_truth_t, predicted_t))  # MAE
                else:
                    output.append('')
                    output.append('')
                print(output)
                # 将结果写入 CSV 文件
                spamwriter.writerow(output)