import pandas as pd
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import joblib


d=pd.read_csv("spam.csv",encoding='latin-1')


d.drop(['Unnamed: 2','Unnamed: 3','Unnamed: 4'],axis=1,inplace=True)


d.columns=['class','message']

d['label']=d['class'].map({'ham':0,'spam':1})

x=d['message']
y=d['label']

cv= CountVectorizer()

x=cv.fit_transform(x)

pickle.dump(cv,open('transform.pkl','wb'))

from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.33,random_state=42)

from sklearn.naive_bayes import MultinomialNB

clf=MultinomialNB()

clf.fit(x_train,y_train)

clf.score(x_test,y_test)

pickle.dump(clf,open('nlp_model.pkl','wb'))




