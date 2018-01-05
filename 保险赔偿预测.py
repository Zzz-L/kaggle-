# -*- coding: utf-8 -*-
"""
Created on Mon Dec 18 19:44:57 2017

@author: Zzz~L
"""
"""知识小结:保险赔偿预测：包含116个离散变量,15个连续变量,构建模型,预测测试集中保险赔偿额
    流程：描述性分析--计算变量相关性--one-hot编码--拆分测试集与训练集--模型训练(未调参)
    1.sns.violinplot 绘制小提琴图  sns.pairplot 绘制散点图  sns.factorplot 绘制两变量关系图(默认折线图)
      sns.countplot  绘制频数图  sns.heatmap 将矩阵数据绘制成颜色编码的矩阵
    2.LabelEncoder 是对不连续的数字或者文本进行编号 OneHotEncoder 用于将表示分类的数据虚拟化  第64行
      基于树的方法不需要进行one-hot编码(什么情况下需要one-hot编码)
      one-hot编码针对分类数据,而对于分段的连续数据 如年龄，则不需要one-hot编码,因为有顺序关系：
      https://www.zhihu.com/question/50587076/answer/199622042
      one-hot与哑变量的区别：https://www.zhihu.com/question/48674426?sort=created
    3.处理离散变量还可以用pandas.get_dummies()
    4.np.column_stack 将1维数组作为列放入二维数组中 np.concatenate按照列添加数组
    5.交叉验证函数以及算法实现（第150行）
    
"""
#———————————————————————————————————预测保险索赔————————————————————————————————
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
dataset=pd.read_csv('train.csv')
dataset_test=pd.read_csv('test.csv')
pd.set_option('display.max_rows', None) #打印所有行和列
pd.set_option('display.max_columns', None)
dataset.head(5)
dataset.info() #查看数据表结构
dataset.isnull().sum()#判断每列缺失情况
dataset.drop('id',axis=1,inplace=True)
dataset.describe() #描述性统计(针对连续变量,离散变量无)
#=======================描述性统计与数据展示===================
dataset.skew()  #数据的偏度
cont_data=dataset.iloc[:,116:] #提取出连续变量
cols=cont_data.columns
n_row=7
for i in range(n_row):
    fig,ax=plt.subplots(nrows=1,ncols=2,figsize=(12, 8)) #figsize设置图形大小
    for j in range(2): #sns.violinplot小提琴图
        sns.violinplot(y=cols[i*2+j],data=cont_data,ax=ax[j]) #绘制小提琴图(综合箱线图与核密度的优点)
#loss变量需要log转化（将目标变量转换为正态分布）
dataset['loss']=np.log1p(dataset['loss']) #计算log(1 + x)
sns.violinplot(y='loss',data=dataset)    
#==========================变量相关性===========================   
#计算相关系数
import operator   
data_corr=cont_data.corr()
sns.heatmap(data_corr,annot=True) #绘制相关性热力图
corr_list=[]
for i in range(data_corr.shape[0]):
    for j in range(i+1,data_corr.shape[0]):
        if data_corr.iloc[i,j]>0.5 or data_corr.iloc[i,j]<-0.5:
            corr_list.append([data_corr.index[i],data_corr.columns[j],round(data_corr.iloc[i,j],3)])
sort_corr=sorted(corr_list,key=operator.itemgetter(2),reverse=True)
#绘制连续变量散点图
for i in range(len(sort_corr)): ##sns.pairplot绘制散点图
    sns.pairplot(dataset,x_vars=sort_corr[i][0],y_vars=sort_corr[i][1],size=6)
#绘制离散变量频数图 离散变量共有116个 连续变量15个
n_col=4
n_row=29
catagory=116
col=dataset.columns[:116]
for i in range(n_row):   #sharey=True使用同一个Y轴
    fig,ax=plt.subplots(nrows=1,ncols=4,sharey=True,figsize=(12,8))
    for j in range(4):
        sns.countplot(x=col[i*4+j],data=dataset,ax=ax[j])

#===========================one-hot编码======================
from sklearn.preprocessing import LabelEncoder #LabelEncoder 是对不连续的数字或者文本进行编号
from sklearn.preprocessing import OneHotEncoder #OneHotEncoder 用于将表示分类的数据虚拟化
labels=[] #提取出每个离散变量的取值(包括训练集和测试集)
for i in range(0,catagory):
    train=dataset[col[i]].unique() #训练集中第i个变量的唯一取值
    test=dataset[col[i]].unique()
    labels.append(list(set(train)|set(test)))
cats=[]
for i in range(0,catagory):
    le=LabelEncoder()
    le.fit(labels[i])
    feature=le.transform(dataset.iloc[:,i])
    feature=feature.reshape(dataset.shape[0],1) #转换为列向量 
    onehot_encoder = OneHotEncoder(sparse=False,n_values=len(labels[i])) #不返回稀疏矩阵,每个特征取值个数为n_values
    feature = onehot_encoder.fit_transform(feature) #先fit 后transform
    cats.append(feature) #将虚拟化后的每个变量添加到列表中
encoded_cats=np.column_stack(cats) #将1维数组作为列放入二维数组中
dataset_encoded=np.concatenate((encoded_cats,dataset.iloc[:,catagory:].values),axis=1) #np.concatenate按照列添加数组
#===========================拆分训练集与验证集==================
from sklearn import cross_validation
r,c=dataset_encoded.shape
i_cols=[] #不太明白这步存在的意义
for i in range(0,c-1):
    i_cols.append(i)
X=dataset_encoded[:,:(c-1)]
Y=dataset_encoded[:,c-1]
val_size=0.1
seed=0 #控制随机种子,重复得到同样的测试集与验证集
x_train,x_val,y_train,y_val=cross_validation.train_test_split(X,Y,test_size=0.1,random_state=0)
#================================模型训练=========================
from sklearn.metrics import mean_absolute_error
x_all=[]
comb = []
mae = []
n = "All"
x_all.append([n, i_cols])
#-------------------logistic-----------------
from sklearn import linear_model
model=linear_model.LinearRegression()
model.fit(x_train,y_train)
y_pred=model.predict(x_val)
#np.expm1表示将对数值变为原来的数值格式
result=mean_absolute_error(np.expm1(y_val),np.expm1(y_pred))#计算平均绝对值误差
mae.append(int(result))
comb.append('LR')
#--------------------岭回归------------------
model=linear_model.Ridge(alpha=1,random_state=0)
model.fit(x_train,y_train)
y_pred=model.predict(x_val)
result=mean_absolute_error(np.expm1(y_val),np.expm1(y_pred))
mae.append(int(result))
comb.append('Ridge 1.0')
#--------------------LASSO-------------------
model=linear_model.LassoCV(cv=10,random_state=0,alphas=[1, 0.1, 0.001, 0.0005])
 ##利用交叉验证选择惩罚系数lambda,alphas候选的惩罚系数
model.fit(x_train,y_train)
y_pred=model.predict(x_val)
result=mean_absolute_error(np.expm1(y_val),np.expm1(y_pred))
mae.append(int(result))
comb.append('LASSO')
#------------------ElasticNet回归--------------
#(结合岭回归与Lasso,既有平方项约束又有绝对值约束)
model=linear_model.ElasticNetCV(cv=3,random_state=0) ##利用交叉验证选择惩罚系数lambda
model.fit(x_train,y_train) 
y_pred=model.predict(x_val)
result=mean_absolute_error(np.expm1(y_val),np.expm1(y_pred))
mae.append(int(result))
comb.append('LASSO')
#----------------------KNN------------------
from sklearn.neighbors import KNeighborsRegressor ##knn用于回归
#******交叉验证选择K-sklearn实现*******
from sklearn.cross_validation import cross_val_score
k_choice=np.array([1,3,5,10,20])
k_scores=[]
#通过循环的方式,计算不同K值对模型的影响,并返回交叉验证后的准确率
for k in k_choice:
    knn=KNeighborsRegressor(n_neighbors=k,n_jobs=-1)
    loss=cross_val_score(knn,x_train,y_train,cv=5,scoring='mean_absolute_error')
    k_scores.append(loss.mean())
#******算法实现交叉验证*******
k_choice=np.array([1,3,5,10,20])
n_folds=5
x_train_folds=np.array_split(x_train,n_folds) #np.array_split 将数组划分为指定块数
y_train_folds=np.array_split(y_train,n_folds)
K_MAE={}
for k in k_choice:
    K_MAE[k]=[]    
for k in k_choice:
    mae=[]
    for i in range(n_folds): #np.vstack是垂直(按照行顺序)的把数组给堆叠起来
        x_train_cv=np.vstack(x_train_folds[0:i]+x_train_folds[i+1:])
        x_val_cv=x_train_folds[i] #np.hstack水平(按列顺序)把数组给堆叠起来
        y_train_cv=np.hstack(y_train_folds[0:i]+y_train_folds[i+1:])
        y_val_cv=y_train_folds[i]
        model=KNeighborsRegressor(n_neighbors=k).fit(x_train_cv,y_train_cv)
        pred=model.predict(x_val_cv)
        mae.append(mean_absolute_error(y_val_cv,pred))
    K_MAE[k]=np.mean(mae)

#----------------------DecisionTree------------------        
from sklearn.tree import DecisionTreeRegressor
model=DecisionTreeRegressor()      
model.fit(x_train,y_train)
y_pred=model.predict(x_val)
result=mean_absolute_error(np.expm1(y_val),np.expm1(y_pred))


