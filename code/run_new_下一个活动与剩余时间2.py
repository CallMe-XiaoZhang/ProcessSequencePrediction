'''
此脚本将train.py找到的LSTM或RNN权重作为输入
将此脚本第176行中的路径更改为指向h5文件
使用train.py生成的LSTM或RNN权重

Author: Niek Tax
'''
# 导包
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

# 自定义函数
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

# 设置设备：如果有 GPU，则使用 GPU，否则使用 CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

for row in spamreader:
    t = time.strptime(row[2], "%Y-%m-%d %H:%M:%S")
    if row[0]!=lastcase:
        caseids.append(row[0])
        casestarttime = t
        lasteventtime = t
        lastcase = row[0]
        if not firstLine:        
            lines.append(line)
            timeseqs.append(times)
            timeseqs2.append(times2)
        line = ''
        times = []
        numlines+=1
    line+=chr(int(row[1])+ascii_offset)
    timesincelastevent = datetime.fromtimestamp(time.mktime(t))-datetime.fromtimestamp(time.mktime(lasteventtime))
    timesincecasestart = datetime.fromtimestamp(time.mktime(t))-datetime.fromtimestamp(time.mktime(casestarttime))
    timediff = 86400 * timesincelastevent.days + timesincelastevent.seconds
    timediff2 = 86400 * timesincecasestart.days + timesincecasestart.seconds
    times.append(timediff)
    times2.append(timediff2)
    lasteventtime = t
    firstLine = False

# add last case
lines.append(line)
timeseqs.append(times)
timeseqs2.append(times2)
numlines+=1

divisor = np.mean([item for sublist in timeseqs for item in sublist])
print('divisor: {}'.format(divisor))
divisor2 = np.mean([item for sublist in timeseqs2 for item in sublist])
print('divisor2: {}'.format(divisor2))

elems_per_fold = int(round(numlines/3))
fold1 = lines[:elems_per_fold]
fold1_c = caseids[:elems_per_fold]
fold1_t = timeseqs[:elems_per_fold]
fold1_t2 = timeseqs2[:elems_per_fold]

fold2 = lines[elems_per_fold:2*elems_per_fold]
fold2_c = caseids[elems_per_fold:2*elems_per_fold]
fold2_t = timeseqs[elems_per_fold:2*elems_per_fold]
fold2_t2 = timeseqs2[elems_per_fold:2*elems_per_fold]

lines = fold1 + fold2
caseids = fold1_c + fold2_c
lines_t = fold1_t + fold2_t
lines_t2 = fold1_t2 + fold2_t2

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


lastcase = ''
line = ''
firstLine = True
lines = []
caseids = []
timeseqs = []  # 自上次事件以来的相对时间
timeseqs2 = [] # 自案件开始以来的相对时间
timeseqs3 = [] # 前一事件的绝对时间
times = []
times2 = []
times3 = []
numlines = 0
casestarttime = None
lasteventtime = None

csvfile = open('../data/%s' % eventlog, 'r')
spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')
next(spamreader, None)  # skip the headers

for row in spamreader:
    t = time.strptime(row[2], "%Y-%m-%d %H:%M:%S")
    if row[0]!=lastcase:
        caseids.append(row[0])
        casestarttime = t
        lasteventtime = t
        lastcase = row[0]
        if not firstLine:        
            lines.append(line)
            timeseqs.append(times)
            timeseqs2.append(times2)
            timeseqs3.append(times3)
        line = ''
        times = []
        numlines+=1
    line+=chr(int(row[1])+ascii_offset)
    timesincelastevent = datetime.fromtimestamp(time.mktime(t))-datetime.fromtimestamp(time.mktime(lasteventtime))
    timesincecasestart = datetime.fromtimestamp(time.mktime(t))-datetime.fromtimestamp(time.mktime(casestarttime))
    midnight = datetime.fromtimestamp(time.mktime(t)).replace(hour=0, minute=0, second=0, microsecond=0)
    timesincemidnight = datetime.fromtimestamp(time.mktime(t))-midnight
    timediff = 86400 * timesincelastevent.days + timesincelastevent.seconds
    timediff2 = 86400 * timesincecasestart.days + timesincecasestart.seconds
    #timediff = log(timediff+1)
    times.append(timediff)
    times2.append(timediff2)
    times3.append(datetime.fromtimestamp(time.mktime(t)))
    lasteventtime = t
    firstLine = False

# add last case
lines.append(line)
timeseqs.append(times)
timeseqs2.append(times2)
timeseqs3.append(times3)
numlines+=1

fold3 = lines[2*elems_per_fold:]
fold3_c = caseids[2*elems_per_fold:]
fold3_t = timeseqs[2*elems_per_fold:]
fold3_t2 = timeseqs2[2*elems_per_fold:]
fold3_t3 = timeseqs3[2*elems_per_fold:]
#fold3_t4 = timeseqs4[2*elems_per_fold:]

lines = fold3
caseids = fold3_c
lines_t = fold3_t
lines_t2 = fold3_t2
lines_t3 = fold3_t3
#lines_t4 = fold1_t4 + fold2_t4

# set parameters
predict_size = 1

# load model, set this to the model generated by train.py
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

# 加载最佳模型
save_dir = 'output_files/models'  # 指定保存模型的目录
save_path = os.path.join(save_dir, 'best_model2.pth')
model.load_state_dict(torch.load(save_path))

# define helper functions
def encode(sentence, times, times3, maxlen=maxlen):
    num_features = len(chars)+5
    X = np.zeros((1, maxlen, num_features), dtype=np.float32)
    leftpad = maxlen-len(sentence)
    times2 = np.cumsum(times)
    for t, char in enumerate(sentence):
        midnight = times3[t].replace(hour=0, minute=0, second=0, microsecond=0)
        timesincemidnight = times3[t]-midnight
        multiset_abstraction = Counter(sentence[:t+1])
        for c in chars:
            if c==char:
                X[0, t+leftpad, char_indices[c]] = 1
        X[0, t+leftpad, len(chars)] = t+1
        X[0, t+leftpad, len(chars)+1] = times[t]/divisor
        X[0, t+leftpad, len(chars)+2] = times2[t]/divisor2
        X[0, t+leftpad, len(chars)+3] = timesincemidnight.seconds/86400
        X[0, t+leftpad, len(chars)+4] = times3[t].weekday()/7
    return torch.tensor(X, dtype=torch.float32)  # 返回 PyTorch 张量

def getSymbol(predictions):
    maxPrediction = 0
    symbol = ''
    i = 0
    for prediction in predictions:
        if(prediction>=maxPrediction):
            maxPrediction = prediction
            symbol = target_indices_char[i]
        i += 1
    return symbol

one_ahead_gt = []
one_ahead_pred = []

two_ahead_gt = []
two_ahead_pred = []

three_ahead_gt = []
three_ahead_pred = []

# 打开输出文件
with open('output_files/results/next_activity_and_time_%s' % eventlog, 'w', encoding='utf-8') as csvfile:
    spamwriter = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
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
    for prefix_size in range(2, maxlen):
        print(prefix_size)
        for line, caseid, times, times3 in zip(lines, caseids, lines_t, lines_t3):
            times.append(0)
            cropped_line = ''.join(line[:prefix_size])
            cropped_times = times[:prefix_size]
            cropped_times3 = times3[:prefix_size]
            if '!' in cropped_line:
                continue  # 不要对这个案子做任何预测，因为这个案子已经结束了

            ground_truth = ''.join(line[prefix_size:prefix_size + predict_size])
            ground_truth_t = times[prefix_size:prefix_size + predict_size]
            predicted = ''
            predicted_t = []

            for i in range(predict_size):
                if len(ground_truth) <= i:
                    continue
                enc = encode(cropped_line, cropped_times, cropped_times3,maxlen)
                enc_tensor = enc.clone().detach().float().to(device)

                with torch.no_grad():  # 禁用梯度计算
                    model.eval()  # 切换到评估模式
                    y = model(enc_tensor)  # 进行预测

                y_char = y[0][0]
                y_t = y[1][0][0].item()
                prediction = getSymbol(y_char)
                cropped_line += prediction
                if y_t < 0:
                    y_t = 0
                cropped_times.append(y_t)
                y_t = y_t * divisor
                cropped_times3.append(cropped_times3[-1] + datetime2.timedelta(seconds=y_t))
                predicted_t.append(y_t)

                if i == 0:
                    if len(ground_truth_t) > 0:
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

                if prediction == '!':  # 预测结束
                    print('! predicted, end case')
                    break
                predicted += prediction

            output = []
            if len(ground_truth) > 0:
                output.append(caseid)
                output.append(prefix_size)
                output.append(ground_truth)
                output.append(predicted)
                output.append(1 - distance.nlevenshtein(predicted, ground_truth) / max(len(predicted), len(ground_truth)))
                dls = 1 - (damerau_levenshtein_distance(predicted, ground_truth) / max(len(predicted), len(ground_truth)))
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