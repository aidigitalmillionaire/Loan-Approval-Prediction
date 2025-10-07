#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Converted from Jupyter Notebook: notebook.ipynb
Conversion Date: 2025-10-07T07:36:34.940Z
"""

# # Titanic - Machine Learning from Disaster


# ### Predicting survival on the Titanic


# ![](https://storage.googleapis.com/kaggle-competitions/kaggle/3136/logos/header.png)


# #### Data Dictionary
# |Variable|Definition|Key|
# |--------|----------|---|
# |survival|	Survival|	0 = No, 1 = Yes|
# |pclass|	Ticket class|	1 = 1st, 2 = 2nd, 3 = 3rd|
# |sex|	Sex	|
# |Age|	Age in years|	
# |sibsp	|# of siblings / spouses aboard the Titanic	|
# |parch	|# of parents / children aboard the Titanic	|
# |ticket	|Ticket number	|
# |fare	|Passenger fare	|
# |cabin	|Cabin number	|
# |embarked	|Port of Embarkation|	C = Cherbourg, Q = Queenstown, S = Southampton|


#importing the libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('titanic_train.csv')
df.head()

df.shape

# ## Data Preprocessing


#removing the columns
df = df.drop(columns=['PassengerId','Name','Cabin','Ticket'], axis= 1)

df.describe()

#checking data types
df.dtypes

#checking for unique value count
df.nunique()

#checking for missing value count
df.isnull().sum()

# #### Refining the data


# replacing the missing values
df['Age'] =  df['Age'].replace(np.nan,df['Age'].median(axis=0))
df['Embarked'] = df['Embarked'].replace(np.nan, 'S')

#type casting Age to integer
df['Age'] = df['Age'].astype(int)

#replacing with 1 and female with 0
df['Sex'] = df['Sex'].apply(lambda x : 1 if x == 'male' else 0)

# #### Categorising in groups i.e. Infant(0-5), Teen (6-20), 20s(21-30), 30s(31-40), 40s(41-50), 50s(51-60), Elder(61-100)


# creating age groups - young (0-18), adult(18-30), middle aged(30-50), old (50-100)
df['Age'] = pd.cut(x=df['Age'], bins=[0, 5, 20, 30, 40, 50, 60, 100], labels = ['Infant', 'Teen', '20s', '30s', '40s', '50s', 'Elder'])

# ## Exploratory Data Analysis


# #### Plotting the Countplot to visualize the numbers


# visulizing the count of the features
fig, ax = plt.subplots(2,4,figsize=(20,20))
sns.countplot(x = 'Survived', data = df, ax= ax[0,0])
sns.countplot(x = 'Pclass', data = df, ax=ax[0,1])
sns.countplot(x = 'Sex', data = df, ax=ax[0,2])
sns.countplot(x = 'Age', data = df, ax=ax[0,3])
sns.countplot(x = 'Embarked', data = df, ax=ax[1,0])
sns.histplot(x = 'Fare', data= df, bins=10, ax=ax[1,1])
sns.countplot(x = 'SibSp', data = df, ax=ax[1,2])
sns.countplot(x = 'Parch', data = df, ax=ax[1,3])

# #### Visualizing the replationship between the features


fig, ax = plt.subplots(2,4,figsize=(20,20))
sns.countplot(x = 'Sex', data = df, hue = 'Survived', ax= ax[0,0])
sns.countplot(x = 'Age', data = df, hue = 'Survived', ax=ax[0,1])
sns.boxplot(x = 'Sex',y='Fare', data = df, hue = 'Pclass', ax=ax[0,2])
sns.countplot(x = 'SibSp', data = df, hue = 'Survived', ax=ax[0,3])
sns.countplot(x = 'Parch', data = df, hue = 'Survived', ax=ax[1,0])
sns.scatterplot(x = 'SibSp', y = 'Parch', data = df,hue = 'Survived', ax=ax[1,1])
sns.boxplot(x = 'Embarked', y ='Fare', data = df, ax=ax[1,2])
sns.pointplot(x = 'Pclass', y = 'Survived', data = df, ax=ax[1,3])

# ## Data Preprocessing 2


from sklearn import preprocessing
le = preprocessing.LabelEncoder()
le.fit(['S','C','Q'])
df['Embarked'] = le.transform(df['Embarked'])

age_mapping = {
    'infant': 0,
    'teen': 1,
    '20s': 2,
    '30s': 3,
    '40s': 4,
    '50s': 5,
    'elder': 6}
df['Age'] = df['Age'].map(age_mapping)
df.dropna(subset=['Age'], axis= 0, inplace = True)

# #### Coorelation Heatmap


sns.heatmap(df.corr(), annot= True)

# #### Separating the target and independent variable


y = df['Survived']
x = df.drop(columns=['Survived'])

# ## Model Training


# ### Logistic Regression


from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()
lr

lr.fit(x,y)
lr.score(x,y)

# ### Decision Tree Classifier


from sklearn.tree import DecisionTreeClassifier
dtree = DecisionTreeClassifier()
dtree

dtree.fit(x,y)
dtree.score(x,y)

# ### Support Vector Machine (SVM)


from sklearn.svm import SVC
svm = SVC()
svm

svm.fit(x,y)
svm.score(x,y)

# ### K-Nearest Neighbor


from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier()
knn


knn.fit(x,y)
knn.score(x,y)

# #### From the above four model Decision Tree Classifier has the highest Training accuracy, so only Decision Tree Classifier will work on the Test Set.


# ### Importing the test set


df2 = pd.read_csv('titanic_test.csv')
df2.head()

#removing the columns
df2 = df2.drop(columns=['PassengerId','Name','Cabin','Ticket'], axis= 1)

# ## Data Preprocessing the Test set


df2['Age'] =  df2['Age'].replace(np.nan,df2['Age'].median(axis=0))
df2['Embarked'] = df2['Embarked'].replace(np.nan, 'S')

#type casting Age to integer
df2['Age'] = df2['Age'].astype(int)

#replacing with 1 and female with 0
df2['Sex'] = df2['Sex'].apply(lambda x : 1 if x == 'male' else 0)

df2['Age'] = pd.cut(x=df2['Age'], bins=[0, 5, 20, 30, 40, 50, 60, 100], labels = [0,1,2,3,4,5,6])

le.fit(['S','C','Q'])
df2['Embarked'] = le.transform(df2['Embarked'])

df2.dropna(subset=['Age'], axis= 0, inplace = True)

df2.head()

# ### Separating the traget and independent variable


x = df2.drop(columns=['Survived'])
y = df2['Survived']

# ## Predicting using Decision Tree Classifier


tree_pred = dtree.predict(x)

from sklearn.metrics import accuracy_score
accuracy_score(y, tree_pred)

# #### Confusion Matrix


from sklearn.metrics import confusion_matrix
sns.heatmap(confusion_matrix(y,tree_pred),annot= True, cmap = 'Blues')
plt.ylabel('Predicted Values')
plt.xlabel('Actual Values')
plt.title('confusion matrix')
plt.show()