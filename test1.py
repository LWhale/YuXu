import helper
import numpy as np

strategy_instance=helper.strategy()

class1_word = set()
for s1 in strategy_instance.class1:
    class1_word.update(set(s1))

class0_word = set()
for s0 in strategy_instance.class0:
    class0_word.update(set(s0))

# all_word = set()
# all_word.update(class1_word)
# all_word.update(class0_word)

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
#
# # print(len(class0_word.difference(class1_word)))
# # print(len(class1_word.difference(class0_word)))

for step in range(10):
    ew1 = list(fre1.keys())[list(fre1.values()).index(exchange1[step])]
    ew0 = list(fre0.keys())[list(fre0.values()).index(exchange0[step])]
    del fre1[ew1]
    del fre0[ew0]
    # print('ew1 ew0', ew1, ew0)
    class1_word.remove(ew1)
    class1_word.add(ew0)

# print(class1_word)

feature = list(class1_word)

x_train = []
y_train = []
for si in strategy_instance.class0:
    x_line = []
    y_train.append(0)
    for wi in feature:
        if wi in si:
            x_line.append(1)
        else:
            x_line.append(0)
    x_train.append(x_line)

for sj in strategy_instance.class1:
    xz_line = []
    y_train.append(1)
    for wj in feature:
        if wj in sj:
            xz_line.append(1)
        else:
            xz_line.append(0)
    x_train.append(xz_line)

with open('test_data.txt', 'r') as test_data:
    test_list = [line.strip().split(' ') for line in test_data]
test_x = []
test_real_y = []
for sk in test_list:
    xk_line = []
    test_real_y.append([1])
    for wk in feature:
        if wk in sk:
            xk_line.append(1)
        else:
            xk_line.append(0)
    test_x.append(xk_line)

test_x = np.array(test_x)
x_train = np.array(x_train)
y_train = np.array(y_train)
test_real_y = np.array(test_real_y)

# strategy_instance = helper.strategy()
parameters = {'gamma': 'auto', 'C': 1.0, 'kernel': 'linear', 'degree': 3, 'coef0': 0.0}
clf = strategy_instance.train_svm(parameters, x_train, y_train)

pred_y = clf.predict(test_x)

from sklearn.metrics import confusion_matrix

cm = confusion_matrix(test_real_y, pred_y)
print(cm)
accuracy = (cm[0][0]+cm[1][1])/(sum(cm[0])+sum(cm[1]))*100
print(accuracy)


