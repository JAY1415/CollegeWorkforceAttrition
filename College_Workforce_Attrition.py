import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import recall_score, precision_score, f1_score
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_validate
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score, log_loss
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBClassifier
import xgboost
from sklearn import svm

df = pd.read_csv('CollegeWorkforceAttrition1.csv')
df1=pd.read_csv('CollegeWorkforceAttrition1.csv')
print(df.head(5))
print(df.columns)
df.shape
df.info()

""" Transpose the describe to make it easier to read. """
df.describe().T

df.drop(['YearOfJoining'],axis=1,inplace=True)

df.isnull().sum()

"""Visulazing categorical values"""

gender_dict = df["Gender"].value_counts()
gender_dict

df['Gender'].value_counts().plot(kind='bar',color=['salmon','lightblue'],title="Count of different gender")

#Create a plot for crosstab

pd.crosstab(df['Gender'],df['IntentionToQuit']).plot(kind="bar",figsize=(10,6))
plt.title("Intention To Quit  vs Gender")
plt.xlabel("Yes/No")
plt.ylabel("No of people who left based on gender")
plt.legend(["Yes","No"])
plt.xticks(rotation=0)
plt.savefig('static/plot1.png')
promoted_dict = df["YearSinceLastPromotion"].value_counts()
promoted_dict

df['YearSinceLastPromotion'].value_counts().plot(kind='bar',color=['salmon','lightblue'],title="Promoted and Non Promoted")

#Create a plot for crosstab

pd.crosstab(df['YearSinceLastPromotion'],df['IntentionToQuit']).plot(kind="bar",figsize=(10,6))
plt.title("Stay/Left vs YearSinceLastPromotion")
plt.xlabel("Stay/Left")
plt.ylabel("No. of people who left/stay based on promotion")
plt.legend(["Yes","No"])
plt.xticks(rotation=0)
plt.savefig('static/plot2.png')

func_dict = df["Department"].value_counts()
func_dict

df['Department'].value_counts().plot(kind='bar',color=['salmon','lightblue'],title="Departments")

#Create a plot for crosstab

pd.crosstab(df['Department'],df['IntentionToQuit']).plot(kind="bar",figsize=(10,6))
plt.title("Stay/Left vs Department")
plt.xlabel("Stay/Left")
plt.ylabel("No. of people who left/stay based on Department")
plt.legend(["Yes","No"])
plt.xticks(rotation=0)
plt.savefig('static/plot3.png')

Marital_dict = df["MaritalStatus"].value_counts()
print(Marital_dict)

"""Visulization of Continuous data"""



def Promoted(x):
    if x == 'Yes':
        return int(1)
    else:
        return int(0)

data_l = df["YearSinceLastPromotion"].apply(Promoted)
df['New Promotion'] = data_l
df.head()

def Dept(x):
    if x == 'Computer Engineering':
        return int(1)
    elif x == 'CSBS':
        return int(2)

data_l = df["Department"].apply(Dept)
df['New Department'] = data_l
df.head()

def Gen(x):
    if x in gender_dict.keys():
        return str(x)
    else:
        return 'other'

data_l = df["Gender"].apply(Gen)
df['New Gender'] = data_l
df.head()

gend = pd.get_dummies(df["New Gender"])
gend.head()



df.head()

df.drop(["MaritalStatus","YearSinceLastPromotion","Department", "Speciality", "Designation", "YearlyLeaves",
              "Gender", "New Gender", "Qualification"],axis=1,inplace=True)

df.head()

df.shape

# Let's make our correlation matrix visual
corr_matrix=df.corr()
fig,ax=plt.subplots(figsize=(15,10))
ax=sns.heatmap(corr_matrix,
               annot=True,
               linewidths=0.5,
               fmt=".2f"
              )

df.to_csv("processed table.csv")

dataset = pd.read_csv("processed table.csv")
dataset = pd.DataFrame(dataset)
y = dataset["IntentionToQuit"]
X = dataset.drop("IntentionToQuit",axis=1)

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=4)
X_train.head()

lr=LogisticRegression(C = 0.1, random_state = 42, solver = 'liblinear')
dt=DecisionTreeClassifier()
rm=RandomForestClassifier()
gnb=GaussianNB()
knn = KNeighborsClassifier(n_neighbors=3)
svm = svm.SVC(kernel='linear')

for a,b in zip([lr,dt,knn,svm,rm,gnb],["Logistic Regression","Decision Tree","KNN","SVM","Random Forest","Naive Bayes"]):
    a.fit(X_train,y_train)
    prediction=a.predict(X_train)
    y_pred=a.predict(X_test)
    score1=accuracy_score(y_train,prediction)
    score=accuracy_score(y_test,y_pred)
    msg1="[%s] training data accuracy is : %f" % (b,score1)
    print(msg1)

model_scores={'Logistic Regression':lr.score(X_test,y_test),
             'KNN classifier':knn.score(X_test,y_test),
             'Support Vector Machine':svm.score(X_test,y_test),
             'Random forest':rm.score(X_test,y_test),
              'Decision tree':dt.score(X_test,y_test),
              'Naive Bayes':gnb.score(X_test,y_test)
             }
model_scores

from sklearn.metrics import classification_report
rm_y_preds = rm.predict(X_test)

print(classification_report(y_test,rm_y_preds))

from sklearn.metrics import classification_report

lr_y_preds = lr.predict(X_test)

print(classification_report(y_test,lr_y_preds))

model_compare=pd.DataFrame(model_scores,index=['accuracy'])
model_compare

model_compare.T.plot(kind='bar') # (T is here for transpose)

feature_dict=dict(zip(dataset.columns,list(lr.coef_[0])))
feature_dict

feature_df=pd.DataFrame(feature_dict,index=[0])
feature_df.T.plot(kind="bar",legend=False,title="Feature Importance")


import pickle

# Save the trained model as a pickle string.
pickle.dump(lr, open('Model/MlModel.pkl', 'wb'))

# Loading model to compare the results
model = pickle.load(open('Model/MlModel.pkl', 'rb'))
# Use the loaded pickled model to make predictions
model.predict(X_test)

