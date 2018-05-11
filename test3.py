import helper
import numpy as np

strategy_instance = helper.strategy()
class1_word = set()
for s1 in strategy_instance.class1:
    class1_word.update(set(s1))

class0_word = set()
for s0 in strategy_instance.class0:
    class0_word.update(set(s0))

fre0 = {}
class0_diff = class0_word.difference(class1_word)
for i in list(class0_diff):
    count0 = 0
    for j in strategy_instance.class0:
        count0 += j.count(i)
    fre0.update({i:count0})

fre1 = {}
class1_diff = class1_word.difference(class0_word)
for m in list(class1_diff):
    count1 = 0
    for n in strategy_instance.class1:
        count1 += n.count(m)
    fre1.update({m:count1})

exchange0 = sorted(fre0.values(),reverse=True)
exchange1 = sorted(fre1.values(),reverse=True)

diffword1 = []
diffword0 = []

for sd1 in range(len(exchange1)):
    ew1 = list(fre1.keys())[list(fre1.values()).index(exchange1[sd1])]
    del fre1[ew1]
    diffword1.append(ew1)

ew0 = list(fre0.keys())[list(fre0.values()).index(exchange0[0])]
diffword0.append(ew0)

print('diffword1 ', diffword1)
print(exchange1)
print('diffword0 ', diffword0)
print(exchange0[0])

all_word = set()
all_word.update(class1_word)
all_word.update(class0_word)

feature = list(all_word)

x_train = []
y_train = []
for si in strategy_instance.class0:
    x_line = []
    y_train.append(0)
    for wi in feature:
        if wi in si:
            x_line.append(si.count(wi))
            # x_line.append(1)
        else:
            x_line.append(0)
    x_train.append(x_line)

for sj in strategy_instance.class1:
    xz_line = []
    y_train.append(1)
    for wj in feature:
        if wj in sj:
            xz_line.append(sj.count(wj))
            # xz_line.append(1)
        else:
            xz_line.append(0)
    x_train.append(xz_line)

x_train = np.array(x_train)
y_train = np.array(y_train)

parameters = {'C': 10, 'kernel': 'rbf', 'gamma': 0.001, 'degree': 3, 'coef0': 0.0}
clf = strategy_instance.train_svm(parameters, x_train, y_train)

with open('test_data.txt', 'r') as test_data:
    test_list = [line.strip().split(' ') for line in test_data]
test_x = []
# test_real_y = []
for sk in test_list:
    xk_line = []
    # test_real_y.append([1])
    for wk in feature:
        if wk in sk:
            # xk_line.append(1)
            xk_line.append(sk.count(wk))
        else:
            xk_line.append(0)
    test_x.append(xk_line)

test_x = np.array(test_x)
pred_y = clf.predict(test_x)
print('pred_y ', pred_y)

# for i in pred_y:
#     if i == 1:
#         for j in diffword1:
#             if j in test_list:




##如果pred_y == 1, 把class1的高频词换成class0的