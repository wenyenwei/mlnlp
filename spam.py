from sklearn.naive_bayes import MultinomialNB
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfTransformer, TfidfVectorizer, CountVectorizer
from sklearn.model_selection import train_test_split



# read dataset, kaggle sms spam dataset from https://www.kaggle.com/uciml/sms-spam-collection-dataset/data
df = pd.read_csv('spam.csv')

# drop blank columns
df = df.drop(["Unnamed: 2", "Unnamed: 3", "Unnamed: 4"], axis = 1)

# rename columns
df.columns = ['label', 'text']

# create binary labels
df['b_label'] = df['label'].map({'ham':0, 'spam':1})
Y = df['b_label'].as_matrix()

# try with tfid method (result gets lower score)
# tfidf = TfidfVectorizer(decode_error='ignore')
# X = tfidf.fit_transform(df['text'])

# transform text to numbers and counts (this method result in highest score)
count_vectorizer = CountVectorizer(decode_error='ignore')
X = count_vectorizer.fit_transform(df['text'])

# try adjust data with tfid (result gets lower score)
# transformer = TfidfTransformer()
# X = transformer.fit_transform(X)

# split up the data
Xtrain, Xtest, Ytrain, Ytest = train_test_split(X, Y, test_size=0.33)

# create the model, train it, print scores
model = MultinomialNB()
model.fit(Xtrain, Ytrain)
print("train score:", model.score(Xtrain, Ytrain))
print("test score:", model.score(Xtest, Ytest))
