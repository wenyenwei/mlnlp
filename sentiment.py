import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfTransformer, TfidfVectorizer, CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier

# read dataset, kaggle twitter dataset https://www.kaggle.com/c/twitter-sentiment-analysis2/data
df = pd.read_csv('senti_train.csv')

# shuffle data randomly
df = df.sample(frac=1)


# transform text to numbers and counts
count_vectorizer = CountVectorizer(decode_error='ignore',binary='boolean')
X = count_vectorizer.fit_transform(df['SentimentText'])

# normalize data
preprocessing.scale(X, with_mean=False)


# define label
Y = df['Sentiment']

Xtrain, Xtest, Ytrain, Ytest = train_test_split(X, Y, test_size=0.33)



# create the model, train it, print scores
model = RandomForestClassifier(n_estimators=10, n_jobs=2)
model.fit(Xtrain, Ytrain)
print("train score:", model.score(Xtrain, Ytrain))
print("test score:", model.score(Xtest, Ytest))
