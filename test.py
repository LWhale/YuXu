import helper

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
            x_line.append(1)
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
            xz_line.append(1)
        else:
            xz_line.append(0)
    x_train.append(xz_line)
# print(x_train[400].count(1))
# print(len(set(strategy_instance.class1[400-360])))
