# -*- coding: utf-8 -*-
# @Time : 2024/11/13 10:11
# @Author : Tao
# @Email : 3195858080@qq.com
# @File : finish_code.py


from collections import Counter
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.model_selection import train_test_split
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
times = []
times2 = []
times3 = []
times4 = []
caseids = []
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
        caseids.append(row[0])
        #与前面的案例不同，此案例的更新时间
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

# Section 2 分割数据集
# 分割为3折（2折训练，第三折验证）
elems_per_fold = int(round(numLines/3))
# fold1--------------------------------
fold1 = lines[:elems_per_fold]
fold1_t = timeSeqs[:elems_per_fold]
fold1_t2 = timeSeqs2[:elems_per_fold]
fold1_t3 = timeSeqs3[:elems_per_fold]
fold1_t4 = timeSeqs4[:elems_per_fold]
with open('output_files/folds/fold1.csv', 'w', newline='') as csvfile:
    spamwriter = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
    for row, timeSeq in zip(fold1, fold1_t):
        spamwriter.writerow([('{}#{}'.format(s, t)).encode("utf-8") for s, t in zip(row, timeSeq)])
# fold2--------------------------------
fold2 = lines[elems_per_fold:2*elems_per_fold]
fold2_t = timeSeqs[elems_per_fold:2*elems_per_fold]
fold2_t2 = timeSeqs2[elems_per_fold:2*elems_per_fold]
fold2_t3 = timeSeqs3[elems_per_fold:2*elems_per_fold]
fold2_t4 = timeSeqs4[elems_per_fold:2*elems_per_fold]
with open('output_files/folds/fold2.csv', 'w', newline='') as csvfile:
    spamwriter = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
    for row, timeseq in zip(fold2, fold2_t):
        spamwriter.writerow([('{}#{}'.format(s, t)).encode("utf-8") for s, t in zip(row, timeSeq)])
# fold3--------------------------------
fold3 = lines[2*elems_per_fold:]
fold3_t = timeSeqs[2*elems_per_fold:]
fold3_t2 = timeSeqs2[2*elems_per_fold:]
fold3_t3 = timeSeqs3[2*elems_per_fold:]
fold3_t4 = timeSeqs4[2*elems_per_fold:]
with open('output_files/folds/fold3.csv', 'w', newline='') as csvfile:
    spamwriter = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
    for row, timeseq in zip(fold3, fold3_t):
        spamwriter.writerow([('{}#{}'.format(s, t)).encode("utf-8") for s, t in zip(row, timeSeq)])
# leave away fold3 for now
lines = fold1 + fold2
lines_t = fold1_t + fold2_t
lines_t2 = fold1_t2 + fold2_t2
lines_t3 = fold1_t3 + fold2_t3
lines_t4 = fold1_t4 + fold2_t4

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

# Section 5 为训练生成子序列和序列后续字符
sentences = []
sentences_t = []
sentences_t2 = []
sentences_t3 = []
sentences_t4 = []
next_chars = []
next_chars_t = []
next_chars_t2 = []
next_chars_t3 = []
next_chars_t4 = []

step = 1
for line, line_t, line_t2, line_t3, line_t4 in zip(lines, lines_t, lines_t2, lines_t3, lines_t4):
    for i in range(1, len(line), step):# 对于lines中的每一个活动序列
        sentences.append(line[0: i])# sentences是以step对line进行拆分，后续同理
        sentences_t.append(line_t[0:i])
        sentences_t2.append(line_t2[0:i])
        sentences_t3.append(line_t3[0:i])
        sentences_t4.append(line_t4[0:i])
        next_chars.append(line[i])
        # append next
        if i == len(line)-1:# 如果是line中倒数第二个
            next_chars_t.append(0)
            next_chars_t2.append(0)
            next_chars_t3.append(0)
            next_chars_t4.append(0)
        else:
            next_chars_t.append(line_t[i])
            next_chars_t2.append(line_t2[i])
            next_chars_t3.append(line_t3[i])
            next_chars_t4.append(line_t4[i])


# Section 6 构造张量
# 6.1构造张量形状
num_features = len(chars)+5
print('num features: {}'.format(num_features))
X = np.zeros((len(sentences), maxLen, num_features), dtype=np.float32)
y_a = np.zeros((len(sentences), len(target_chars)), dtype=np.float32)#分类任务的目标张量
y_t = np.zeros((len(sentences)), dtype=np.float32)#回归任务的目标张量

# 6.2填充张量
softness = 0 # 用于1，0的平滑处理
for i, sentence in enumerate(sentences):#sentences是子序列的集合（对于一个活动序列）
    # np.set_printoptions(threshold=np.inf)
    leftpad = maxLen-len(sentence) #计算需要填充的前置空位数（为将序列对齐至 maxLen）
    # 提取序列信息
    next_t = next_chars_t[i]
    sentence_t = sentences_t[i]
    sentence_t2 = sentences_t2[i]
    sentence_t3 = sentences_t3[i]
    sentence_t4 = sentences_t4[i]
    for t, char in enumerate(sentence): #sentence是子序列（对于一个活动序列）对子序列构造特征向量
        multiset_abstraction = Counter(sentence[:t+1])#当前子序列的字符统计信息，用于构建多重集抽象
        for c in chars:#遍历 chars 字符集中的每个字符 c。
            if c==char: #当前字符 char 是否与遍历到的字符 c 相等
                # init by 1
                X[i, t+leftpad, char_indices[c]] = 1 #则 char 的位置在 X 张量中标记为 1，形成 one-hot 编码。
        # 添加特征：字符位置，1，2，3，4
        X[i, t+leftpad, len(chars)] = t+1 #字符的位置
        X[i, t+leftpad, len(chars)+1] = sentence_t[t]/divisor
        X[i, t+leftpad, len(chars)+2] = sentence_t2[t]/divisor2
        X[i, t+leftpad, len(chars)+3] = sentence_t3[t]/DaySeconds
        X[i, t+leftpad, len(chars)+4] = sentence_t4[t]/7
    for c in target_chars:#遍历target_chars字符集中的每个字符 c。
        if c==next_chars[i]:#下一个字符next_chars[i]是否与遍历到的字符 c 相等
            y_a[i, target_char_indices[c]] = 1-softness
        else:
            y_a[i, target_char_indices[c]] = softness/(len(target_chars)-1)
    y_t[i] = next_t/divisor

print(X.shape)
print(y_a.shape)
print(y_t.shape)

# Section 7拆分出验证集
X_train, X_val, y_a_train, y_a_val, y_t_train, y_t_val = train_test_split(X, y_a, y_t, test_size=0.2, random_state=42)
# 将拆分后的数据重新转换为 PyTorch 张量
X_train = torch.tensor(X_train, dtype=torch.float32).to(device)
y_a_train = torch.tensor(y_a_train, dtype=torch.float32).to(device)
y_t_train = torch.tensor(y_t_train, dtype=torch.float32).to(device)

X_val = torch.tensor(X_val, dtype=torch.float32).to(device)
y_a_val = torch.tensor(y_a_val, dtype=torch.float32).to(device)
y_t_val = torch.tensor(y_t_val, dtype=torch.float32).to(device)

# Section 8 构造模型
# 8.1 输入与模型
# 定义模型
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

# 参数定义
max_len = X.shape[1]
print(max_len)
num_features = len(chars)+5
print(num_features)
num_classes = len(target_chars)  # 类别数
print(num_classes)

# 实例化模型
model = MultiTaskLSTMModel(max_len, num_features, num_classes).to(device)

# 损失函数
criterion_act = nn.CrossEntropyLoss()
criterion_time = nn.L1Loss()  # MAE损失

# 优化器
optimizer = torch.optim.Adam(model.parameters(), lr=0.002)
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10, min_lr=0, verbose=True)

# 训练模型
epochs = 1500
patience = 150 # 早停耐心
best_val_loss = float('inf')
no_improvement = 0
model_save_path = 'output_files/models/pt'

for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    # 前向传播
    act_pred, time_pred = model(X_train)  # X为输入张量
    loss_act = criterion_act(act_pred, y_a_train)  # y_a为活动标签
    loss_time = criterion_time(time_pred, y_t_train)  # y_t为时间标签
    loss = loss_act + loss_time  # 总损失
    # 反向传播
    loss.backward()
    optimizer.step()
    # 验证阶段
    model.eval()
    with torch.no_grad():
        val_act_pred, val_time_pred = model(X_val)
        val_loss_act = criterion_act(val_act_pred, y_a_val)
        val_loss_time = criterion_time(val_time_pred, y_t_val)
        val_loss = val_loss_act + val_loss_time
        # 检查验证损失
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            no_improvement = 0
        else:
            no_improvement += 1
            if no_improvement >= patience:
                print("Early stopping")
                break
    # 调整学习率
    scheduler.step(val_loss)
    print(f"Epoch {epoch + 1}/{epochs}, Train Loss: {loss.item():.4f}, Val Loss: {val_loss:.4f}")

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
with open('output_files/results/true_next_activity_and_time2_%s' % EventLog, 'w', encoding='utf-8') as csvfile:
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