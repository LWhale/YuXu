import helper
from sklearn import svm
import numpy as np

strategy_instance=helper.strategy()
# print(len(strategy_instance.class0)) 360
# print(len(strategy_instance.class1)) 180

all_word = set()
for s0 in strategy_instance.class0:
    all_word.update(set(s0))
for s1 in strategy_instance.class1:
    all_word.update(set(s1))
# print(len(all_word)) 5718
# print(list(all_word)[:10])
feature = list(all_word)

x_train = []
y_train = []
for si in strategy_instance.class0:
    x_line = []
    y_train.append(0)
    for wi in feature:
        if wi in si:
            #x_line.append(1)
            x_line.append(si.count(wi))
        else:
            x_line.append(0)
    x_train.append(x_line)
# print(x_train[100].count(1))
# print(len(set(strategy_instance.class0[100])))
# print(y_train[:5])

for sj in strategy_instance.class1:
    xz_line = []
    y_train.append(1)
    for wj in feature:
        if wj in sj:
            #xz_line.append(1)
            xz_line.append(sj.count(wj))
        else:
            xz_line.append(0)
    x_train.append(xz_line)
# print(x_train[400].count(1))
# print(len(set(strategy_instance.class1[400-360])))
# print(y_train[510:520])

with open('test_data.txt', 'r') as test_data:
    test_list = [line.strip().split(' ') for line in test_data]
test_x = []
test_real_y = []
for sk in test_list:
    xk_line = []
    test_real_y.append([1])
    for wk in feature:
        if wk in sk:
            #xk_line.append(1)
            xk_line.append(sk.count(wk))
        else:
            xk_line.append(0)
    test_x.append(xk_line)

test_x = np.array(test_x)
x_train = np.array(x_train)
y_train = np.array(y_train)
test_real_y = np.array(test_real_y)

strategy_instance = helper.strategy()
parameters = {'gamma': 'auto', 'C': 1.0, 'kernel': 'linear', 'degree': 3, 'coef0': 0.0}
clf = strategy_instance.train_svm(parameters, x_train, y_train)

pred_y = clf.predict(test_x)

from sklearn.metrics import confusion_matrix

cm = confusion_matrix(test_real_y, pred_y)
accuracy = (cm[0][0]+cm[1][1])/(sum(cm[0])+sum(cm[1]))*100
print(accuracy)

# clf = svm.SVC()
# print(clf.fit(x_train, y_train))
# with open('test_data.txt', 'r') as td:
#     td = [line.strip().split(' ') for line in td]
#
# test_dataset = []
# for st in td:
#     t_line = []
#     for wt in feature:
#         if wt in st:
#             t_line.append(1)
#         else:
#             t_line.append(0)
#     test_dataset.append(t_line)

# print(test_dataset[1].count(1))  98
# print(len(set(td[1])))  105(7 words not in training features)

