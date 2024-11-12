# -*- coding: utf-8 -*-
# @Time : 2024/11/11 17:16
# @Author : Tao
# @Email : 3195858080@qq.com
# @File : true_下一个活动与剩余时间.py

import os
import time
import copy
import numpy as np
from datetime import datetime
from sklearn.metrics import mean_absolute_error
import csv
import torch.nn as nn
import torch
import distance
from jellyfish._jellyfish import damerau_levenshtein_distance


# 全局变量
EventLog = "helpdesk.csv"
AsciiOffset = 161
DaySeconds = 86400
WeekDayNum = 7
# 检查CUDA是否可用
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 数据预处理
# section 1 从csv文件中提取times;times2;times3;times4
lines = []
timeSeqs = []
timeSeqs2 = []
timeSeqs3 = []
timeSeqs4 = []

# 1.1 打开 CSV 文件
csvfile = open('../data/%s' % EventLog, 'r')
spamReader = csv.reader(csvfile, delimiter=',', quotechar='|')
next(spamReader, None)  # skip the headers

# 1.2 辅助变量
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

# 1.3 从"CaseID, ActivityID, CompleteTimestamp"中提取line、.....time4
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

# Section 2 在每个轨迹结尾追加结束表示，同时找到最大长度
lines_with_exclamation = []
for line in lines:
    line_with_exclamation = line + '!'
    lines_with_exclamation.append(line_with_exclamation)
lines = lines_with_exclamation
# find maximum line size
maxLen = max(map(lambda x: len(x), lines))

# Section 3 加载训练好的模型
# 参数定义
max_len=maxLen
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
model.to(device)
model.eval()  # 设置为评估模式

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

# Section 7 辅助函数
def encode(sentence, times, times2, times3, times4, maxlen=max_len):
    num_features = len(chars)+5
    X = np.zeros((1, maxlen, num_features), dtype=np.float32)
    leftpad = maxlen-len(sentence)
    for t, char in enumerate(sentence):
        for c in chars:
            if c==char:
                X[0, t+leftpad, char_indices[c]] = 1
        X[0, t+leftpad, len(chars)] = t+1 #字符的位置
        X[0, t+leftpad, len(chars)+1] = times[t]/divisor
        X[0, t+leftpad, len(chars)+2] = times2[t]/divisor2
        X[0, t+leftpad, len(chars)+3] = times3[t]/DaySeconds
        X[0, t+leftpad, len(chars)+4] = times4[t]/7
    return torch.tensor(X, dtype=torch.float32)  # 返回 PyTorch 张量

def getSymbol(predictions):
    # 获取最大预测值的索引
    maxPredictionIndex = predictions.argmax().item()
    # 返回目标字符
    symbol = target_indices_char[maxPredictionIndex]
    return symbol

# Section 6 生成预测并写入文件
predict_size = 1

one_ahead_gt = []
one_ahead_pred = []

two_ahead_gt = []
two_ahead_pred = []

three_ahead_gt = []
three_ahead_pred = []

# 打开输出文件以写入预测结果
with open('output_files/results/true_next_activity_and_time_%s' % EventLog, 'w', encoding='utf-8') as csvfile:
    spamwriter = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
    # 写入文件的表头行，包含预测分析中的各项指标
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
    for prefix_size in range(2, max_len):
        print(prefix_size)
        for line, caseid, times,times2,times3,times4 in zip(lines, caseids, lines_t,lines_t2,lines_t3,lines_t4):
            cropped_line = ''.join(line[:prefix_size])
            cropped_times = times[:prefix_size]
            cropped_times2 = times2[:prefix_size]
            cropped_times3 = times3[:prefix_size]
            cropped_times4 = times4[:prefix_size]

            if '!' in cropped_line:
                continue  # 不要对这个案子做任何预测，因为这个案子已经结束了
            ground_truth = ''.join(line[prefix_size:prefix_size + predict_size])
            ground_truth_t = times[prefix_size:prefix_size + predict_size]

            predicted = ''
            predicted_t = []
            for i in range(0,predict_size):
                if len(ground_truth) <= i:
                    continue
                enc = encode(cropped_line, cropped_times,cropped_times2,cropped_times3, cropped_times4,max_len)
                enc_tensor = enc.clone().detach().float().to(device)
                # 进行预测
                with torch.no_grad():  # 禁用梯度计算
                    act_pred, time_pred = model(enc_tensor)
                y_char = getSymbol(act_pred)
                y_t = time_pred.item()
                # 幻觉方法添加
                # 对y_t处理
                cropped_line += y_char
                if y_t < 0:
                    y_t = 0
                cropped_times.append(y_t)
                y_t = y_t * divisor
                predicted_t.append(y_t)
                # 对y_char处理
                if i == 0:
                    if len(ground_truth_t) > 0:#预测为1步，后续至少1
                        one_ahead_pred.append(y_t)
                        one_ahead_gt.append(ground_truth_t[0])
                if i == 1:
                    if len(ground_truth_t) > 1:
                        two_ahead_pred.append(y_t)
                        two_ahead_gt.append(ground_truth_t[1])
                if i == 2:
                    if len(ground_truth_t) > 2:
                        three_ahead_pred.append(y_t)
                        three_ahead_gt.append(ground_truth_t[2])
                if y_char == '!':  # 预测结束
                    print('! predicted, end case')
                    break
                predicted += y_char
            print('caseid',caseid)
            print('line',line)
            print('cropped_line',cropped_line)
            print('ground_truth',ground_truth)
            print('predicted',predicted)
            print('ground_truth_t',ground_truth_t)
            print('predicted_t',predicted_t)

            # 写入部分
            output = []
            if len(ground_truth) > 0:
                output.append(caseid)
                output.append(prefix_size)
                output.append(ground_truth)
                output.append(predicted)
                output.append(1 - distance.nlevenshtein(predicted, ground_truth) / max(len(predicted), len(ground_truth)))
                dls = 1 - (damerau_levenshtein_distance(predicted, ground_truth) / max(len(predicted),len(ground_truth)))
                if dls < 0:
                    dls = 0  # 修正 Damerau-Levenshtein Similarity 为负数的情况
                output.append(dls)
                output.append(1 - distance.jaccard(predicted, ground_truth))
                output.append('; '.join(str(x) for x in ground_truth_t))
                output.append('; '.join(str(x) for x in predicted_t))

                if len(predicted_t) > len(ground_truth_t):  # 如果预测的事件比实际事件多，只使用需要的事件进行时间评估
                    predicted_t = predicted_t[:len(ground_truth_t)]
                if len(ground_truth_t) > len(predicted_t):  # 如果预测的事件比实际事件少，用 0 作为占位符
                    predicted_t.extend([0] * (len(ground_truth_t) - len(predicted_t)))

                if len(ground_truth_t) > 0 and len(predicted_t) > 0:
                    output.append('')
                    output.append(mean_absolute_error(ground_truth_t, predicted_t))
                else:
                    output.append('')
                    output.append('')
                print(output)
                spamwriter.writerow(output)
        print('\n'*5)
























