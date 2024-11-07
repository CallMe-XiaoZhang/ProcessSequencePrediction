'''
此脚本将evaluate_suffix和_remaining_time.py的输出作为输入
因此，后者需要先执行

Author: Niek Tax
'''

import csv

eventlog = "helpdesk.csv"

# 以指定编码打开文件
with open('output_files/results/suffix_and_remaining_time_%s' % eventlog, 'r', encoding='utf-8') as csvfile:
    r = csv.reader(csvfile)
    header = next(r)  # 跳过表头

    vals = dict()
    for row in r:
        if len(row) < 3:
            print(f"Warning: Row with insufficient columns: {row}")
            continue  # 跳过这一行

        l = vals.get(row[0], [])
        if len(row[1]) == 0 and len(row[2]) == 0:
            l.append(1)
        elif len(row[1]) == 0 and len(row[2]) > 0:
            l.append(0)
        elif len(row[1]) > 0 and len(row[2]) == 0:
            l.append(0)
        else:
            l.append(int(row[1][0] == row[2][0]))
        vals[row[0]] = l

    l2 = list()
    for k in vals.keys():
        l2.extend(vals[k])
        res = sum(vals[k]) / len(vals[k])
        print('{}: {}'.format(k, res))

    print('total: {}'.format(sum(l2) / len(l2)))
