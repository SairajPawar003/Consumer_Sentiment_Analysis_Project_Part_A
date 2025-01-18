#Data Cleaning and Preprocessing
#process of data Cleaning (Lowercase , remove special character, punctuation ,stop word,correct spelling , normalization)
#Importing the Liabraries from bulding Emotion Classifier

import pandas as pd 
import numpy
import matplotlib.pyplot as plt 
import re 
import string
import xgboost 


import nltk
# nltk.download('stopwords')
''' 
the following error occures first 
Resource stopwords not found.
  Please use the NLTK Downloader to obtain the resource:
  >>> import nltk
  >>> nltk.download('stopwords')
'''
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.stem import PorterStemmer
from nltk.util import ngrams
from nltk.tokenize import word_tokenize

import textblob 
from textblob.classifiers import NaiveBayesClassifier 
from textblob import TextBlob,Word

import sklearn
# from sklearn import *
import sklearn.feature_extraction.text as text 
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation ,NMF, TruncatedSVD
from sklearn import model_selection,preprocessing, linear_model,naive_bayes,metrics,svm,decomposition,ensemble
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, mean_absolute_error



# v) model bulding phase 
def build(model_initializer, independent_variables_training, target, independent_variable_test):
    # fit
    model_initializer.fit(independent_variables_training,target)
    # predict
    # modelPred = classifier_model.predict(independent_variable_test)
    modelPred = model_initializer.predict(independent_variable_test)
    return metrics.accuracy_score(modelPred, Ytest)

'''
# use proper otherwise error occurs 
def build(classifier_model, independent_variable_train, dependent_variable_train, independent_variable_test):
    # Train the model
    classifier_model.fit(independent_variable_train, dependent_variable_train)
    
    # Predict on the test set
    modelPred = classifier_model.predict(independent_variable_test)
    
    return modelPred
  '''




#i) import data 
data = pd.read_csv("ISEAR_dataset (1).csv")
data.columns=['EMOTION','TEXT','Nothing']
# print(data.head(20))

#ii) Preprocessing Steps 

#1) Uppercase Letters in to Lowercase
data['TEXT'] = data['TEXT'].apply(lambda a: " ".join(a.lower() for a in a.split()))
# print(data.head(20))
#2) Remove Space and Special Characters 
data['TEXT'] = data['TEXT'].apply(lambda a: " ".join(a.replace('[^\w\s]','') for a in a.split()))
#3) Remove Stop words 
stop = stopwords.words('english')
data['TEXT'] = data['TEXT'].apply(lambda a: " ".join(a for a in a.split() if a not in stop))
#4) Correct Spelling 
data['TEXT'] = data['TEXT'].apply(lambda a: str(TextBlob(a).correct()))
print(data.head())
#5) Do Stemming 
st = PorterStemmer()
data['TEXT'] = data['TEXT'].apply(lambda a: " ".join([st.stem(word) for word in a.split()]))



#iii) Label Encoding 
print(data['EMOTION'].value_counts())

object = preprocessing.LabelEncoder()
data['EMOTION'] = object.fit_transform(data['EMOTION'])
print(data['EMOTION'].value_counts())

# iv) trainig data by using train-test split
Xtrain,Xtest,Ytrain,Ytest = model_selection.train_test_split(data['TEXT'],data['EMOTION'],stratify=data['EMOTION'])
#for this we can use two steps that is feature engineering 
#feature engineering
#count vectorizer
cv=CountVectorizer()
cv.fit(data['TEXT'])
cv_xtrain= cv.transform(Xtrain)
cv_xtest= cv.transform(Xtest)
#TF-IDE
tv = TfidfVectorizer()
tv.fit(data['TEXT'])
tv_xtrain = tv.transform(Xtrain)
tv_xtest = tv.transform(Xtest)

#v) Model Bulding Phase in this phase we had used def build
#vi) multinomial naive bayes algorithm
output1 = build(naive_bayes.MultinomialNB(),cv_xtrain,Ytrain,cv_xtest)
output = build(naive_bayes.MultinomialNB(),tv_xtrain,Ytrain,tv_xtest)
print(output1)
print(output)

'''
output 

   EMOTION                                               TEXT Nothing
0     fear  every time imagine someone love could contact ...     NaN
1    anger  obviously unjustly treated possibility elucida...     NaN
2  sadness  think short time live relate periods life thin...     NaN
3  disgust  gathering found involuntarily sitting next two...     NaN
4    shame  realized directing feelings discontent partner...     NaN
joy        1091
sadness    1082
anger      1079
fear       1076
shame      1071
disgust    1066
guilt      1050
Name: EMOTION, dtype: int64
4    1091
5    1082
0    1079
2    1076
6    1071
1    1066
3    1050
Name: EMOTION, dtype: int64
0.5689196381053752

'''


'''
Ans --> 0.561944
        0.565081

2) Linear Classifier or Logestic Regression 
output = build(Linear_model.LogisteicRegression(),cv_xtrain,Ytrain,cv_xtest)
output = build(Linear_model.LogisteicRegression(),tv_xtrain,Ytrain,tv_xtest)
print(output)
Ans --> 0.565082
        0.590172


3) Support Vector Machine 
output = build(svm.SVC(),cv_xtrain,Ytrain,cv_xtest)
output = build(svm.SVC(),tv_xtrain,Ytrain,tv_xtest)
print(output)
Ans --> 0.545739
        0.578672

4) random forest
output = build(ensemble.RandomForestClassifier(),cv_xtrain,Ytrain,cv_xtest)
output = build(ensemble.RandomForestClassifier(),tv_xtrain,Ytrain,tv_xtest)
print(output)
Ans --> 0.553580
        0.536330

'''


''' i faced errors in 
nltk stopwords 
then second in Emotion column contains guit word which is out of model 
then third in def builder classifier_model    '''

