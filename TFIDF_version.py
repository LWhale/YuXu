import helper
#from sklearn.feature_extraction.text import CountVectorizer
#from sklearn.feature_extraction.text import TfidfTransformer
import numpy as np
from math import log
import time
strategy_instance=helper.strategy()
#print(len(strategy_instance.class0),360)
#print(len(strategy_instance.class1),180)
#print(strategy_instance.class0)
# coding:utf-8

all_words = strategy_instance.class0 + strategy_instance.class1

def tfidf(word, sample_list, sample_class):
    
    def tf(word, sample_list):
        total_num = len(sample_list)
        count = 0
        for each_word in sample_list:
            if word == each_word:
                count += 1
    
        tf = count/total_num
        #print("tf:",tf)
        return tf
    
    def idf(word, sample_class):
        total_smp = len(sample_class)
        count_sample = 0
        for sample in sample_class:
            if word in sample:
                count_sample += 1
        #print("count_sample",count_sample)
        idf = log(total_smp/ 1 + count_sample)
        #print("idf", idf)
        return idf
    tf = tf(word, sample_list)
    idf = idf(word, sample_class)
    final = tf * idf
    return final
all_feature = []
all_word = set()
for s0 in strategy_instance.class0:
    all_word.update(set(s0))
for s1 in strategy_instance.class1:
    all_word.update(set(s1))
for word in all_word:
    all_feature.append(word)
#print(all_feature)
#all_feature = list(all_word)
# print(len(all_word)) 5718
# print(list(all_word)[:10])
    
#class_0_feature = set()
#for s0 in strategy_instance.class0:
#    class_0_feature.update(set(s0))
#class_1_feature = set()
#for s1 in strategy_instance.class1:
#    class_1_feature.update(set(s1))
#print(strategy_instance.class0[1])
#class_0_tfidf = dict()#key is word, value is tfidf
#for sample in strategy_instance.class0:
#    for word in sample:
#        final = tfidf(word, sample, strategy_instance.class0)
#        class_0_tfidf[word] = final

#class_1_tfidf = {}
#for sample in strategy_instance.class1:
#   for word in sample:
#        tfidf = tfidf(word, sample, strategy_instance.class1)
#        class_1_tfidf[word] = tfidf
#print(class_0_tfidf)
#print(class_1_tfidf)

#all_tfidf = dict()#key is word, value is tfidf
time = time.time()
all_list = len(all_words) * [[0] * len(all_feature)]
for i in range(len(all_words)):
    for j in range(len(all_feature)):
        all_list[i][j] = tfidf(all_feature[j], all_words[i], all_words)
print(time.time() - time)       


  