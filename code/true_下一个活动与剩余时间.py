# -*- coding: utf-8 -*-
# @Time : 2024/11/11 17:16
# @Author : Tao
# @Email : 3195858080@qq.com
# @File : true_下一个活动与剩余时间.py

import os
import time
import copy
import numpy as np
from collections import Counter
from datetime import datetime, timedelta
import csv
import torch.nn as nn
import torch
import distance
from sklearn import metrics
from jellyfish._jellyfish import damerau_levenshtein_distance
from collections import OrderedDict

# 全局变量
EventLog = "helpdesk.csv"
AsciiOffset = 161
DaySeconds = 86400
WeekDayNum = 7
# 检查CUDA是否可用
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Section 1 加载训练好的模型
# 参数定义
max_len=15
num_features = 14
num_classes = 10  # 类别数

# 加载模型，将其设置为myTrain.py生成的模型
class MultiTaskLSTMModel(nn.Module):
    def __init__(self, max_len, num_features, num_classes):
        super(MultiTaskLSTMModel, self).__init__()
        # 共享的LSTM层
        self.lstm1 = nn.LSTM(num_features, 100, batch_first=True, dropout=0.2)
        self.bn1 = nn.BatchNorm1d(max_len)
        # 活动预测的LSTM层
        self.lstm2_1 = nn.LSTM(100, 100, batch_first=True, dropout=0.2)
        self.bn2_1 = nn.BatchNorm1d(100)
        self.act_output = nn.Linear(100, num_classes)  # 活动预测输出层
        # 时间预测的LSTM层
        self.lstm2_2 = nn.LSTM(100, 100, batch_first=True, dropout=0.2)
        self.bn2_2 = nn.BatchNorm1d(100)
        self.time_output = nn.Linear(100, 1)  # 时间预测输出层
    def forward(self, x):
        # 共享的LSTM层
        x, _ = self.lstm1(x)
        x = self.bn1(x)
        # 活动预测分支
        x1, _ = self.lstm2_1(x)
        x1 = self.bn2_1(x1[:, -1, :])  # 取最后一个时间步
        act_output = self.act_output(x1)
        # 时间预测分支
        x2, _ = self.lstm2_2(x)
        x2 = self.bn2_2(x2[:, -1, :])  # 取最后一个时间步
        time_output = self.time_output(x2)
        return act_output, time_output

model = MultiTaskLSTMModel(max_len, num_features, num_classes)# 实例化
# 加载参数
model_path = model_save_path = 'output_files/models/pt'
pt_files = [f for f in os.listdir(model_path) if f.endswith('.pt')]
latest_model_file = max(pt_files, key=lambda f: os.path.getmtime(os.path.join(model_path, f)))
state_dict = torch.load(os.path.join(model_path, latest_model_file))
model.load_state_dict(state_dict)
model.eval()  # 设置为评估模式

# 数据预处理
# section 2 从csv文件中提取times;times2;times3;times4
lines = []
timeSeqs = []
timeSeqs2 = []
timeSeqs3 = []
timeSeqs4 = []

# 2.1 打开 CSV 文件
csvfile = open('../data/%s' % EventLog, 'r')
spamReader = csv.reader(csvfile, delimiter=',', quotechar='|')
next(spamReader, None)  # skip the headers

# 2.2 辅助变量
# window of case time
caseStartTime = None
lastEventTime = None
# information of case
line = ''
caseids = []
times = []
times2 = []
times3 = []
times4 = []
# csv variables
numLines = 0
firstLine = True
lastCase = ''

# 2.3 从"CaseID, ActivityID, CompleteTimestamp"中提取line、.....time4
for row in spamReader:
    # creates a datetime object from row[2]
    t = time.strptime(row[2], "%Y-%m-%d %H:%M:%S")
    # update event variables
    if row[0] != lastCase:  # 'lastCase'是为循环保存最后一个执行的案例
        #与前面的案例不同，此案例的更新时间
        caseids.append(row[0])
        lastCase = row[0]
        caseStartTime = t
        lastEventTime = t
        #不是第一行，将它们附加到目标
        if firstLine:
            firstLine = False
        else:
            lines.append(line)
            timeSeqs.append(times)
            timeSeqs2.append(times2)
            timeSeqs3.append(times3)
            timeSeqs4.append(times4)
        numLines += 1
        # reset line, times, times2, time3, time4
        line = ''
        times = []
        times2 = []
        times3 = []
        times4 = []

    # append line
    line += chr(int(row[1]) + AsciiOffset)
    # append times
    timeSinceLastEvent = datetime.fromtimestamp(time.mktime(t)) - datetime.fromtimestamp(time.mktime(lastEventTime))
    timediff = DaySeconds * timeSinceLastEvent.days + timeSinceLastEvent.seconds
    times.append(timediff)
    # append times2
    timeSinceCaseStart = datetime.fromtimestamp(time.mktime(t)) - datetime.fromtimestamp(time.mktime(caseStartTime))
    timediff2 = DaySeconds * timeSinceCaseStart.days + timeSinceCaseStart.seconds
    times2.append(timediff2)
    # append times3
    midnight = datetime.fromtimestamp(time.mktime(t)).replace(hour=0, minute=0, second=0, microsecond=0)
    timeSinceMidnight = datetime.fromtimestamp(time.mktime(t)) - midnight
    timediff3 = timeSinceMidnight.seconds  # 这就只剩下午夜之后的时间了
    times3.append(timediff3)
    # append times4
    timediff4 = datetime.fromtimestamp(time.mktime(t)).weekday()  # 星期几
    times4.append(timediff4)
    # update last event
    lastEventTime = t

# add last case
lines.append(line)
timeSeqs.append(times)
timeSeqs2.append(times2)
timeSeqs3.append(times3)
timeSeqs4.append(times4)
numLines += 1

# 计算平均值
# average time between events
divisor = np.mean([item for sublist in timeSeqs for item in sublist])
print('divisor: {}'.format(divisor))
# average time between current and first events
divisor2 = np.mean([item for sublist in timeSeqs2 for item in sublist])
print('divisor2: {}'.format(divisor2))
# average time between
divisor3 = np.mean([item for sublist in timeSeqs3 for item in sublist])
print('divisor3: {}'.format(divisor3))
# average time between
divisor4 = np.mean([item for sublist in timeSeqs4 for item in sublist])
print('divisor4: {}'.format(divisor4))

# Section 3 在每个轨迹结尾追加结束表示，同时找到最大长度
lines_with_exclamation = []
for line in lines:
    line_with_exclamation = line + '!'
    lines_with_exclamation.append(line_with_exclamation)
lines = lines_with_exclamation
# find maximum line size
maxLen = max(map(lambda x: len(x), lines))

# Section 4 获取映射字典
chars = map(lambda x: set(x), lines)
chars = list(set().union(*chars))
chars.sort()
target_chars = copy.copy(chars) # target_chars: all characters in lines
chars.remove('!')   # chars: all characters in lines except '!'
print('total chars: {}, target chars: {}'.format(len(chars), len(target_chars)))
print("chars:", chars)
print("target_chars:", target_chars)

char_indices = dict((c, i) for i, c in enumerate(chars))
indices_char = dict((i, c) for i, c in enumerate(chars))
target_char_indices = dict((c, i) for i, c in enumerate(target_chars))
target_indices_char = dict((i, c) for i, c in enumerate(target_chars))
print(char_indices)
print(indices_char)
print(target_char_indices)
print(target_indices_char)

# Section 5 分割数据
# 分割为3折（2折训练，第三折验证）
elems_per_fold = int(round(numLines/3))

# fold3--------------------------------
fold3 = lines[2*elems_per_fold:]
fold3_c = caseids[2*elems_per_fold:]
fold3_t = timeSeqs[2*elems_per_fold:]
fold3_t2 = timeSeqs2[2*elems_per_fold:]
fold3_t3 = timeSeqs3[2*elems_per_fold:]
fold3_t4 = timeSeqs4[2*elems_per_fold:]

# 使用第三折
lines = fold3
caseids = fold3_c
lines_t = fold3_t
lines_t2 = fold3_t2
lines_t3 = fold3_t3
lines_t4 = fold3_t4

# Section 6 生成预测并写入文件
# 打开输出文件以写入预测结果
with open('output_files/results/next_activity_and_time_%s' % EventLog, 'wb') as csvfile:
    spamwriter = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
    # 写入文件的表头行，包含预测分析中的各项指标
    spamwriter.writerow(["CaseID", "Prefix length", "Groud truth", "Predicted", "Levenshtein", "Damerau", "Jaccard",
                         "Ground truth times", "Predicted times", "RMSE", "MAE"])

