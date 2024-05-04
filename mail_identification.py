# 垃圾邮件识别
import jieba # 导入jieba分词库
import os
import pandas # 导入pandas数据处理库
import re # 使用正则表达式
from sklearn.feature_extraction.text import CountVectorizer
import pickle
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score


def dataPreProcessing(): # 数据预处理
    email_data = [] # 存放所有数据
    train_data = trainDataCleansing() # 训练数据集合
    test_data = TestataCleansing() # 测试数据集合
    return [train_data, test_data]


def trainDataCleansing(): # 训练数据清洗函数
    train_data = {'content':[], 'type':[]} # 训练集数据字典
    # normal_filelist = os.listdir('./DataSet/train/normal') # 获取训练集-正常邮件文件名列表
    # spam_filelist = os.listdir('./DataSet/train/spam') # 获取训练集-垃圾邮件文件名列表

    # for file_name in normal_filelist: # 清洗正常邮件文件
    #     print('正在处理文件: '+file_name+'....')
    #     with open('./DataSet/train/normal/'+file_name, 'r') as mail_file:
    #         mail_str = mail_file.read() # 读取邮件文件内容
    #         pattern = re.compile(r'(?!X-UIDL:\s*\S*\s+)[\u4e00-\u9fa5][.\s\S]*') # 邮件内容正则匹配规则
    #         find_list = pattern.findall(mail_str) # 获取所有匹配的结果-是一个列表，一般就是列表的第一个
    #         if len(find_list)!=0: # 列表非空，即匹配成果
    #             mail_body = find_list[0].replace(" ","") # 获取匹配内容，去除全部空白字符
    #     train_data['content'].append(mail_body) # 将邮件内容添加进字典
    #     train_data['type'].append(0); # 将邮件所属类别添加进字典
    
    # for file_name in spam_filelist: # 清洗垃圾邮件文件
    #     print('正在处理文件: '+file_name+'....')
    #     with open('./DataSet/train/spam/'+file_name, 'r') as mail_file:
    #         mail_str = mail_file.read()
    #         pattern = re.compile(r'(?!X-Mailer:\s*\S*\s+)[\u4e00-\u9fa5][.\s\S]*') # 邮件内容正则匹配规则
    #         pattern_replace = re.compile(r'[\x00-\x08\x0B\x0C\x0E-\x1F]') # 邮件内容正则匹配规则
    #         find_list = pattern.findall(mail_str)
    #         if len(find_list)!=0:
    #             mail_body = find_list[0].replace(" ","")
    #             mail_body = re.sub(pattern_replace, " ", mail_body)
    #     train_data['content'].append(mail_body)
    #     train_data['type'].append(1); # 同上，懒得写了
    
    # train_dataframe = pandas.DataFrame(train_data) # 转化为pandas的DataFrame数据类型便于后续处理
    # train_dataframe.drop_duplicates(inplace=True) # 去除重复数据
    # train_dataframe.to_excel('./train.xlsx') # 数据写入excel
    train_dataframe = pandas.read_excel('./train.xlsx')
    train_data = train_dataframe.to_dict('list')
    train_data.pop('Unnamed: 0')
    print(len(train_data['content']))
    print(len(train_data['type']))

    print("******训练邮件数据清洗完成******")
    return train_data

def TestataCleansing(): # 训练数据清洗函数
    test_data = {'content':[], 'type':[]} # 训练集数据字典
    normal_filelist = os.listdir('./DataSet/test/normal') # 获取训练集-正常邮件文件名列表
    spam_filelist = os.listdir('./DataSet/test/spam') # 获取训练集-垃圾邮件文件名列表

    for file_name in normal_filelist: # 清洗正常邮件文件
        print('正在处理文件: '+file_name+'....')
        with open('./DataSet/test/normal/'+file_name, 'r') as mail_file:
            mail_str = mail_file.read() # 读取邮件文件内容
            pattern = re.compile(r'(?!X-UIDL:\s*\S*\s+)[\u4e00-\u9fa5][.\s\S]*') # 邮件内容正则匹配规则
            find_list = pattern.findall(mail_str) # 获取所有匹配的结果-是一个列表，一般就是列表的第一个
            if len(find_list)!=0: # 列表非空，即匹配成果
                mail_body = find_list[0].replace(" ","") # 获取匹配内容，去除全部空白字符
        test_data['content'].append(mail_body) # 将邮件内容添加进字典
        test_data['type'].append(0); # 将邮件所属类别添加进字典
    
    for file_name in spam_filelist: # 清洗垃圾邮件文件
        print('正在处理文件: '+file_name+'....')
        with open('./DataSet/test/spam/'+file_name, 'r') as mail_file:
            mail_str = mail_file.read()
            pattern = re.compile(r'(?!X-Mailer:\s*\S*\s+)[\u4e00-\u9fa5][.\s\S]*') # 邮件内容正则匹配规则
            pattern_replace = re.compile(r'[\x00-\x08\x0B\x0C\x0E-\x1F]') # 邮件内容正则匹配规则
            find_list = pattern.findall(mail_str)
            if len(find_list)!=0:
                mail_body = find_list[0].replace(" ","")
                mail_body = re.sub(pattern_replace, " ", mail_body)
        test_data['content'].append(mail_body)
        test_data['type'].append(1); # 同上，懒得写了
    
    test_dataframe = pandas.DataFrame(test_data) # 转化为pandas的DataFrame数据类型便于后续处理
    test_dataframe.drop_duplicates(inplace=True) # 去除重复数据
    test_dataframe.to_excel('./test.xlsx') # 数据写入excel
    test_dataframe = pandas.read_excel('./test.xlsx')
    # train_data = train_dataframe.to_dict('list')
    # train_data.pop('Unnamed: 0')
    print(len(test_data['content']))
    print(len(test_data['type']))

    print("******训练邮件数据清洗完成******")
    return test_data


def extractFeatureWords(emailContentList): # 提取特征词函数
    print('正在进行文本处理......')
    # 首先分词，然后使用空格隔开每一个词，组成新的文本
    words = [' '.join(jieba.lcut(emailContent)) for emailContent in emailContentList]
    # 读取停用词
    stops = [word.strip() for word in open('./stopwords.txt', encoding='utf-8')]
    # 初始化向量化器
    cv = CountVectorizer(stop_words=stops)
    # 生成特征词词表
    cv.fit(words)
    print('特征词:\n', cv.get_feature_names_out())
    # 文本转换为向量
    result = cv.transform(words)
    print('数值化:\n', result.toarray())
    # 存储特征提取器
    pickle.dump(cv, open('count_vectorizer.pkl', 'wb'))
    # 存储数值化文本
    pickle.dump(result, open('train.data', 'wb'))
    print("******文本特征值提取与数值化处理完毕******")

def naiveBayesTrain(labels): # 朴素贝叶斯模型训练
    print('模型开始训练......')
    # 加载训练数据
    inputs = pickle.load(open('train.data', 'rb')).toarray()
    # 初始化算法模型
    # alhpa 表示拉普拉斯平滑系数
    # fit_prior 表示训练先验概率
    # class_prior 提供的各个类别的先验概率
    estimator = MultinomialNB(alpha=1.0, fit_prior=True, class_prior=None)
    # 训练算法模型
    estimator.fit(inputs, labels)
    # 算法模型评估
    y_preds = estimator.predict(inputs)
    score = accuracy_score(labels, y_preds)
    print('标签:', y_preds)
    print('准确率:', score)
    # 算法模型保存
    pickle.dump(estimator, open('bayes.pkl', 'wb'))
    print('******训练结束******')

def naiveBayesJudge(): # 朴素贝叶斯模型判断
    inputs = ['您的百万大奖正在等待您的领取。']
    # 1. 文本数值化
    words = [' '.join(jieba.lcut(text)) for text in inputs]
    extractor = pickle.load(open('count_vectorizer.pkl', 'rb'))
    inputs = extractor.transform(words).toarray()
    print(inputs)
    # 2. 算法模型推理
    estimator = pickle.load(open('bayes.pkl', 'rb'))
    y_preds = estimator.predict(inputs)
    # 输出预测标签
    print('标签:', y_preds)
