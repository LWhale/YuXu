import helper
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

strategy_instance = helper.strategy()
# print(len(strategy_instance.class0)) 360
# print(len(strategy_instance.class1)) 180
with open('class-0.txt','r') as class0:
    class_0=[line.strip() for line in class0]
with open('class-1.txt','r') as class1:
    class_1=[line.strip() for line in class1]
all_sample = class_0 + class_1

#print(len(all_sample))
    
#vectorizer_0 = CountVectorizer()
#vectorizer_1 = CountVectorizer()
vectorizer_all = CountVectorizer()

#X_0 = vectorizer_0.fit_transform(class_0)
#X_1 = vectorizer_1.fit_transform(class_1)
X_all = vectorizer_all.fit_transform(all_sample)

#word_0 = vectorizer_0.get_feature_names()
#word_1 = vectorizer_1.get_feature_names()
feature = vectorizer_all.get_feature_names()
#print(word_0)
#X_0 = X_0.toarray()
#X_1 = X_1.toarray()
X_all = X_all.toarray()
#print(X_0)
#print(type(X_0))
#transformer_0 = TfidfTransformer()
#transformer_1 = TfidfTransformer()
transformer_all = TfidfTransformer()

#tfidf_0 = transformer_0.fit_transform(X_0)
#tfidf_1 = transformer_1.fit_transform(X_1)
tfidf_all = transformer_all.fit_transform(X_all)
#print(type(tfidf_all))
X_tfidf = tfidf_all.toarray()
Y_all = [[0]] * len(class_0) + [[1]] * len(class_1)
Y_all = np.ravel(Y_all)
#tfidf_0 = tfidf_0.toarray()
#tfidf_1 = tfidf_1.toarray()
parameters = {'gamma': 0.001, 'C': 10, 'kernel': 'linear', 'degree': 3, 'coef0': 0.0}
clf = strategy_instance.train_svm(parameters, X_tfidf, Y_all)
sv = np.matmul(clf.dual_coef_,clf.support_vectors_)
#sp = clf.coef_
coef = sv.tolist()
coef = coef[0]

with open('test_data.txt', 'r') as test_data:
    test_list = [line.strip().split(' ') for line in test_data]
feature_dict = {}
for i in range(len(feature)):
    feature_dict[feature[i]] = coef[i]
feature_dict = sorted(feature_dict.items(), key = lambda x: x[1], reverse = True)
test_x = []
test_real_y = []
for sk in test_list:
    xk_line = []
    test_real_y.append([1])
    for wk in feature:
        if wk in sk:
            xk_line.append(sk.count(wk))
        else:
            xk_line.append(0)
    test_x.append(xk_line)
word_dict = [None] * len(test_x)
for i in range(len(test_x)):
    word_dict[i] = {}
    for j in range(len(test_x[i])):
        if test_x[i][j] != 0:
            word_dict[i][feature[j]] = (test_x[i][j], coef[j])
            
for i in range(len(word_dict)):
    word_dict[i] = sorted(word_dict[i].items(), key=lambda d: d[1][1], reverse = True)

    