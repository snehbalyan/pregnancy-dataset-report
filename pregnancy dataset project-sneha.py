# Sneha Balyan

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

df_train = pd.read_csv("Proj2_Train.csv")
df_test = pd.read_csv("Proj2_Test.csv")

df_train.shape
df_test.shape

df = df_train.append(df_test)
df.shape

df.head()
df.dtypes
df.info()
df.isnull().sum()
df.duplicated().sum()
df.duplicated().value_counts()
df.corr()

plt.figure(figsize = (9,6))
matrix = df.corr()
sns.heatmap(matrix,cmap='coolwarm',fmt = '.0%',annot = True)
matrix

drop_column_list = ["Sl No","Sample ID","Cerv_Len_cms",
                    "Cerv_Wid_cms","Consistency_Score"]

df.drop(drop_column_list,inplace = True,axis = 1)
df.info()

print(df['Ob_Score'].unique())
df['Ob_Score'] = df['Ob_Score'].map({'G1':1, 'G2A1':2, 'G3A2':3})
df['Ob_Score'].unique()

fig = plt.figure(figsize = (9,6))
sns.boxplot(data = df, linewidth = 1)
plt.xticks(rotation = ('vertical'))
plt.show()

df.duplicated().sum()
duplicate_val = df[df.duplicated(keep = 'first')]
duplicate_val

df.drop_duplicates(keep = 'first', inplace = True)
df.duplicated().sum()

df.shape
df.corr()

plt.figure(figsize = (9,6))
matrix = df.corr()
sns.heatmap(matrix,cmap='coolwarm',fmt = '.0%',annot = True)
matrix

#================================================================

x = df.drop('Del_mode', axis = 1)
y = df['Del_mode']

from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2

select_kbest_rank_feature = SelectKBest(score_func= chi2, k = 5)
kbest_feature = select_kbest_rank_feature.fit(x, y)

df_score = pd.DataFrame(kbest_feature.scores_,columns = ['Score'])
dfcolumns = pd.DataFrame(x.columns)

kbest_rank_feature_concat = pd.concat([dfcolumns,df_score], axis = 1)
kbest_rank_feature_concat.columns = ['features','k_score']
kbest_rank_feature_concat

print(kbest_rank_feature_concat.nlargest(20,'k_score').round(9))

#drop columns through creating list
K_Best_drop_features=['Dialation_Score','Ob_Score']

df.drop(K_Best_drop_features,inplace = True,axis = 1)

df.info()
df.duplicated().sum()

plt.figure(figsize = (9,6))
matrix = df.corr()
sns.heatmap(df.corr(),annot=True,linewidth=0.5,cmap='coolwarm',fmt='.0%')
matrix

#==================================================================

def outliers (df,ft):
    Q1 = df[ft].quantile(0.25)
    Q3 = df[ft].quantile(0.75)
    IQR = Q3 - Q1
    
    lower_bound = Q1-1.5*IQR
    upper_bound = Q3+1.5*IQR
    
    ls = df.index[(df[ft] < lower_bound) | (df[ft] > upper_bound)]
    return ls

def remove (df,ls):
    ls = sorted(set(ls))
    df = df.drop(ls)    
    return df

df.info()

index_list = []
for feature in ['Age','BMI','Position_Score',
                'Effacement_Score','Station_Score',
                'Total_Bishop_Score','Induction','Del_mode']:
    index_list.extend(outliers(df,feature))

df1 = remove(df,index_list)
df1.shape

fig = plt.figure(figsize = (9,6))
sns.boxplot(data = df1, linewidth = 1)
plt.xticks(rotation = ('vertical'))
plt.show()

df1.info()

#================================================================

# Model Development:-

# 1. logistic regression
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,train_size = 0.70)

from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression()
logreg.fit(x_train,y_train)

y_train_pred = logreg.predict(x_train)
y_test_pred = logreg.predict(x_test)

from sklearn.metrics import accuracy_score
print("Training Accuracy Score:",accuracy_score(y_train,y_train_pred).round(3))
print("Test Accuracy Score:",accuracy_score(y_test,y_test_pred).round(3))

#=======================================================================

# 2. confusion matrix
from sklearn.metrics import confusion_matrix,accuracy_score
cm1 = confusion_matrix(y_train,y_train_pred)
print("Accuracy Score:",accuracy_score(y_train,y_train_pred).round(2))
TN=cm1[0,0]
FP=cm1[0,1]
TNR=TN/(TN+FP)
print("Specificity Score:",TNR.round(2))
from sklearn.metrics import recall_score,f1_score,precision_score
print("Sensitivity score:",recall_score(y_train,y_train_pred).round(2))
print("Precision Score:",precision_score(y_train,y_train_pred).round(2))
print("F1 Score:",f1_score(y_train,y_train_pred).round(2))
print(cm1)

from sklearn.metrics import confusion_matrix,accuracy_score
cm2 = confusion_matrix(y_test,y_test_pred)
print("Accuracy Score:",accuracy_score(y_test,y_test_pred).round(2))
TN=cm2[0,0]
FP=cm2[0,1]
TNR=TN/(TN+FP)
print("Specificity Score:",TNR.round(2))
from sklearn.metrics import recall_score,f1_score,precision_score
print("Sensitivity score:",recall_score(y_test,y_test_pred).round(2))
print("Precision Score:",precision_score(y_test,y_test_pred).round(2))
print("F1 Score:",f1_score(y_test,y_test_pred).round(2))
print(cm2)

#=============================================================

# 3. KNN
from sklearn.neighbors import KNeighborsClassifier
KNN = KNeighborsClassifier(n_neighbors=5,p=2)
KNN.fit(x_train,y_train)
KNeighborsClassifier()
y_train_pred = KNN.predict(x_train)
y_test_pred = KNN.predict(x_test)
train_acc = accuracy_score(y_train,y_train_pred)
test_acc = accuracy_score(y_test,y_test_pred)
print((train_acc).round(4))
print((test_acc).round(4))

#============================================================
# 4. MultinomialNB
from sklearn.naive_bayes import MultinomialNB
NB=MultinomialNB()
NB.fit(x_train,y_train)
y_train_pred=NB.predict(x_train)
y_test_pred=NB.predict(x_test)
train_acc=accuracy_score(y_train,y_train_pred)
test_acc=accuracy_score(y_test,y_test_pred)
print((train_acc).round(4))
print((test_acc).round(4))

#==============================================================

# 5. Decision Tree

# fit the model with Entropy
from sklearn.tree import DecisionTreeClassifier

dt = DecisionTreeClassifier(criterion="entropy", max_depth = None)

dt.fit(x_train,y_train)

y_pred_train = dt.predict(x_train)
y_pred_test = dt.predict(x_test)

from sklearn.metrics import accuracy_score
ac1 = accuracy_score(y_train,y_pred_train)
print("Training Accuracy score" ,ac1.round(3))
ac2 = accuracy_score(y_test,y_pred_test)
print("Test Accuracy score" ,ac2.round(3))

dt.tree_.max_depth # calculating the depth of the tree
dt.tree_.node_count # calculating the number of nodes

# fit the model with Gini
from sklearn.tree import DecisionTreeClassifier

dt = DecisionTreeClassifier(criterion="gini", max_depth = None)

dt.fit(x_train,y_train)

y_pred_train = dt.predict(x_train)
y_pred_test = dt.predict(x_test)

from sklearn.metrics import accuracy_score
ac1 = accuracy_score(y_train,y_pred_train)
print("Training Accuracy score" ,ac1.round(3))
ac2 = accuracy_score(y_test,y_pred_test)
print("Test Accuracy score" ,ac2.round(3))

dt.tree_.max_depth # calculating the depth of the tree
dt.tree_.node_count # calculating the number of nodes

#================================================
# 6. PCA
from sklearn.decomposition import PCA
pca = PCA(0.95)
pca.fit(x)
x_pca = pca.transform(x)
y = df["Del_mode"]

pca.explained_variance_ratio_

# 7. Accuracy Score
from sklearn.linear_model import LogisticRegression
logreg=LogisticRegression()
logreg.fit(x_pca,y)
y_pred=logreg.predict(x_pca)

from sklearn.metrics import accuracy_score
print("Accuracy Score:",accuracy_score(y,y_pred).round(4))

