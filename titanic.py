# -*- coding: utf-8 -*-
"""
Created on Mon Dec 25 16:36:31 2017

@author: Zzz~L
"""

"""知识小结：titanic数据集,预测是否生存（模型部分较完整）
            流程：变量清洗--交叉验证实现各种算法--超参数调优(第120行)--模型集成
                  --选择最优模型并调参(第140行)--特征重要性排序
1.data.info 展示数据表结构 data['Survived'].value_counts() 汇总变量取值
2.data.groupby(['Sex','Survived'])['Survived'] 根据两个变量分组汇总
3.pd.crosstab(data.Pclass,data.Survived,margins=True) 交叉列联表 margins添加总和项
4.data.isnull().sum() 汇总每列缺失情况 data.drop('Initial',axis=1,inplace=True)剔除某列
5.pd.cut(array, bins) 根据临界点bins划分区间 pd.cut(array, 5) 将数组划分为五部分          
  pd.qcut(array, 5) 等数量划分区间 (得到因子类型变量)
6.data['Fare_Range'].cat.rename_categories([0,1,2,3]) 对因子变量重新编码
7.data['Sex'].replace(['male','female'],[0,1],inplace=True)  对离散变量重新编码
8.model.feature_importances_  提取模型变量重要性
9.参考其他kernel:
   data["Cabin"].apply(lambda x: 0 if type(x) == float else 1)
   apply函数可以对数据集中某列中每个元素执行某个函数
   map函数也是指对多个序列中的每个元素执行相同函数
   区别：apply作用于dataframe,用于行和列的计算,applymap作用于dataframe,用于元素级别操作
   map作用于series上，是元素级别的操作
   http://www.cnblogs.com/bluescorpio/archive/2010/05/12/1734038.html
"""
#————————————————————————————————titanic python实现—————————————————————————————
#================================数据清洗与变量处理=============================
import pandas as pd
import numpy as py
import matplotlib.pyplot as plt
import seaborn as sns
data=pd.read_csv('train.csv')
data.info() #数据表结构
data.isnull().sum() #汇总每列缺失情况
##变量单独分析
fig,ax=plt.subplots(1,2)
freq=data['Survived'].value_counts()#value_counts()分类汇总
freq.plot.pie(explode=[0,0.1],autopct='%1.1f%%',ax=ax[0],shadow=True)
sns.countplot(x='Survived',data=data,ax=ax[1])#绘制因变量频数图
#Sex
data.groupby(['Sex','Survived'])['Survived'].count() #根据两个变量分组汇总
sns.countplot(x='Sex',hue='Survived',data=data) #绘制分组频数图 hue参数控制分组绘图
#Pclass
pd.crosstab(data.Pclass,data.Survived,margins=True)#交叉列联表 margins添加总和项
fig,ax=plt.subplots(1,2)       
sns.countplot(x='Pclass',data=data,ax=ax[0],color='#CD7F32')    
sns.countplot(x='Pclass',hue='Survived',data=data,ax=ax[1])
sns.factorplot('Pclass','Survived',hue='Sex',data=data) #sns.factorplot绘制两变量关系图
#Age
sns.violinplot('Pclass','Age',hue='Survived',data=data)
#构造变量Initial(称呼)  构建这个变量是为了填补Age的缺失
data['Initial']=[name.split(',')[1].split('.')[0].strip() for name in data.Name]#strip()去掉头和尾的空格
pd.crosstab(data.Sex,data.Initial)
data['Initial'].replace(['Mlle','Mme','Ms','Dr','Major','Lady','the Countess','Jonkheer','Col','Rev','Capt','Sir','Don'],
    ['Miss','Miss','Miss','Mr','Mr','Mrs','Mrs','Other','Other','Other','Mr','Mr','Mr'],inplace=True)
#data.drop('Initial',axis=1,inplace=True)剔除某列
data.groupby('Initial')['Age'].mean() #计算每种称呼下的年龄平均值
#填补年龄缺失(根据称呼对应年龄的平均值)
data.loc[(data.Age.isnull()) & (data.Initial=='Master'),'Age']=5
data.loc[(data.Age.isnull()) & (data.Initial=='Miss'),'Age']=22          
data.loc[(data.Age.isnull()) & (data.Initial=='Mr'),'Age']=33
data.loc[(data.Age.isnull()) & (data.Initial=='Mrs'),'Age']=36
data.loc[(data.Age.isnull()) & (data.Initial=='Other'),'Age']=46         
sns.factorplot('Pclass','Survived',col='Initial',data=data)# col分类变量
##填补Embarked的缺失
data['Embarked'].fillna('S',inplace=True)
#相关性图
sns.heatmap(data.corr(),annot=True)
#构造年龄新变量
data['Age_band']=0
data.loc[data['Age']<=16,'Age_band']=0
data.loc[(data['Age']>16)&(data['Age']<=32),'Age_band']=1
data.loc[(data['Age']>32)&(data['Age']<=48),'Age_band']=2
data.loc[(data['Age']>48)&(data['Age']<=64),'Age_band']=3
data.loc[data['Age']>64,'Age_band']=4
data.head(2)
data.Age_band.value_counts()
sns.factorplot('Age_band','Survived',data=data,col='Pclass')#col='Pclass'表示分成三列
#构造Family_Size
data['Family_Size']=data['SibSp']+data['Parch']
data['Alone']=0
data.loc[data.Family_Size==0,'Alone']=1              
#Fare_Range             
data['Fare_Range']=pd.qcut(data['Fare'],4) #将Fare字段划分为4个等样本的区间  
data.groupby('Fare_Range')['Survived'].mean() 
#将分类变量取值以数字替换
data['Sex'].replace(['male','female'],[0,1],inplace=True)
data['Embarked'].replace(['S', 'C', 'Q'],[0,1,2],inplace=True)
data['Initial'].replace(['Mr', 'Mrs', 'Miss', 'Master', 'Other'],[0,1,2,3,4],inplace=True)
data['Fare_cat']=data['Fare_Range'].cat.rename_categories([0,1,2,3])#对因子变量重新编码
#剔除无用变量
data.drop(['Name','PassengerId','Age','Ticket','Fare','Cabin','Fare_Range'],axis=1,inplace=True)

#================================构建模型=============================
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
X=data.iloc[:,1:]
Y=data[[0]]
x_train,x_test,y_train,y_test=train_test_split(X,Y,test_size=0.3,random_state=0)
#-------------------交叉验证实现各类算法--------------------------
kfold=KFold(n_splits=10,random_state=22)
xyz=[]
accuracy=[]
std=[]
classifiers=['Linear Svm','Radial Svm','Logistic Regression','KNN','Decision Tree','Naive Bayes','Random Forest']              
models=[svm.SVC(kernel='linear'),svm.SVC(kernel='rbf'),LogisticRegression(),KNeighborsClassifier(n_neighbors=9),
DecisionTreeClassifier(),GaussianNB(),RandomForestClassifier(n_estimators=100)]              
for i in models:
    model=i
    cv_result=cross_val_score(model,X,Y.values.ravel(),cv=kfold,scoring='accuracy')
    xyz.append(cv_result.mean())
    std.append(cv_result.std())
    accuracy.append(cv_result)
new_models_dataframe2=pd.DataFrame({'CV Mean':xyz,'std':std},index=classifiers)
#返回交叉验证预测值
y_pred=cross_val_predict(svm.SVC(kernel='rbf'),X,Y.values.ravel(),cv=10) #ravel多维数组降为一维            
sns.heatmap(confusion_matrix(Y,y_pred),annot=True,fmt='2.0f') #confusion_matrix 计算混淆矩阵  fmt 设置字符串格式
#混淆矩阵,横轴表示预测,纵轴表示实际
#------------------------------超参数调优(网格搜索法)------------------------------------       
#SVM 待估参数惩罚系数C和核函数及其系数Gamma
from sklearn.model_selection import GridSearchCV
C=[0.05,0.1,0.2,0.3,0.25,0.4,0.5,0.6,0.7,0.8,0.9,1]
gamma=[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]
kernel=['rbf','linear']
hyper={'kernel':kernel,'C':C,'gamma':gamma}#字典格式汇总参数
gd=GridSearchCV(svm.SVC(),param_grid=hyper,verbose=True) #设置参数
gd.fit(X,Y.values.ravel())#拟合
gd.best_score_ #返回最高分数
gd.best_estimator_ #返回最佳分类器
#随机森林 待估参数n_estimators树木数量
n_estimators=range(100,1000,100)
hyper={'n_estimators':n_estimators}
gd=GridSearchCV(RandomForestClassifier(random_state=0),param_grid=hyper,verbose=True)
gd.fit(X,Y.values.ravel())
gd.best_score_
gd.best_estimator_ #获取最佳估计器以及最佳分数
#------------------------------模型集成------------------------------------       
#1.投票法VotingClassifier(使用不同类型的分类器)
from sklearn.ensemble import VotingClassifier
#voting参数 hard表示少数服从多数的投票结果
#soft 表示计算加权平均概率 每个分类器的权重可根据weights函数设置,然后计算每个分类器的加权平均数
#LogisticRegression C为正则化系数λ的倒数 LogisticRegression默认L2正则化
ensemble=VotingClassifier(estimators=[('KNN',KNeighborsClassifier(n_neighbors=10)),
                                      ('rbf',svm.SVC(kernel='rbf',gamma=0.1,C=0.5,probability=True)),
                                      ('rf',RandomForestClassifier(n_estimators=500,random_state=0)),
                                      ('LR',LogisticRegression(C=0.05)),
                                      ('DT',DecisionTreeClassifier(random_state=0)),
                                      ('NB',GaussianNB()),
                                      ('svm',svm.SVC(kernel='linear',probability=True))],voting='soft') #设置参数
ensemble_model=ensemble.fit(x_train,y_train.values.ravel()) #拟合训练集数据
ensemble_model.score(x_test,y_test)  #计算测试集的平均准确率
#计算交叉验证的平均准确率
cross_score=cross_val_score(ensemble_model,X,Y.values.ravel(),cv=10,scoring='accuracy')
cross_score.mean()             
#2.装袋法 Bagging(使用类似的分类器)  
#KNN           
from sklearn.ensemble import BaggingClassifier
model=BaggingClassifier(base_estimator=KNeighborsClassifier(n_neighbors=3),n_estimators=700,random_state=0)             
m=model.fit(x_train,y_train.values.ravel())
pred=model.predict(x_test)
metrics.accuracy_score(y_test,pred) #等同于上面的ensemble_model.score(x_test,y_test)  
result=cross_val_score(model,X,Y.values.ravel(),cv=10,scoring='accuracy')
result.mean()
#DecisionTreeClassifier
model=BaggingClassifier(base_estimator=DecisionTreeClassifier(),n_estimators=100,random_state=0)             
m=model.fit(x_train,y_train.values.ravel())
pred=model.predict(x_test)
metrics.accuracy_score(y_test,pred) #等同于上面的ensemble_model.score(x_test,y_test)  
result=cross_val_score(model,X,Y.values.ravel(),cv=10,scoring='accuracy')
result.mean()
#3.boosting(使用分类器的顺序学习,逐步增强模型效果,在之后的迭代中,更关注预测错误的样本,尝试正确预测错误样本)
#AdaBoost(Adaptive Boosting)
from sklearn.ensemble import AdaBoostClassifier
ada=AdaBoostClassifier(n_estimators=200,random_state=0,learning_rate=0.1)
result=cross_val_score(ada,X,Y.values.ravel(),cv=10,scoring='accuracy')  
result.mean()
#Gradient Boosting
from sklearn.ensemble import GradientBoostingClassifier
grad=GradientBoostingClassifier(n_estimators=500,random_state=0,learning_rate=0.1)
result=cross_val_score(grad,X,Y.values.ravel(),cv=10,scoring='accuracy')  
result.mean()
#xgboost
import xgboost as xg
xgboost
#------------------------------选择最优模型参数调优--------------------------------
n_estimators=list(range(100,1100,100))
learning_rate=[0.05,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]
hyper={'n_estimators':n_estimators,'learning_rate':learning_rate}
gd=GridSearchCV(estimator=AdaBoostClassifier(),param_grid=hyper,verbose=True)
gd.fit(X,Y.values.ravel())
gd.best_score_
gd.best_estimator_
ada=AdaBoostClassifier(n_estimators=200,learning_rate=0.05,random_state=0)
result=cross_val_predict(ada,X,Y.values.ravel(),cv=10)
confusion_matrix(Y,result)
#------------------------------特征重要性--------------------------------
fig,ax=plt.subplots(2,2)
model=RandomForestClassifier(n_estimators=500,random_state=0)
model.fit(X,Y.values.ravel())
feature=pd.Series(model.feature_importances_,X.columns)#带轴标签的一维数组
feature.sort_values(ascending=True).plot.barh(ax=ax[0,0])
ax[0,0].set_title('Feature Importance in Random Forests')
model=AdaBoostClassifier(n_estimators=200,learning_rate=0.05,random_state=0)
model.fit(X,Y.values.ravel())
pd.Series(model.feature_importances_,X.columns).sort_values(ascending=True).plot.barh(ax=ax[0,1],color='#ddff11')
ax[0,1].set_title('Feature Importance in AdaBoost')
model=GradientBoostingClassifier(n_estimators=500,learning_rate=0.1,random_state=0)
model.fit(X,Y.values.ravel())
pd.Series(model.feature_importances_,X.columns).sort_values(ascending=True).plot.barh(ax=ax[1,0],color='#FD0F00')
ax[1,0].set_title('Feature Importance in Gradient Boosting')


