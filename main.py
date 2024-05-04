# #主程序
import mail_identification
import phishing_site_identification
import pprint

train_data = {} # 训练数据字典



train_data =  mail_identification.dataPreProcessing()[0]# 调用数据预处理函数
mail_identification.extractFeatureWords(train_data['content']) # 文本数值化
mail_identification.naiveBayesTrain(train_data['type']) # 朴素贝叶斯模型训练