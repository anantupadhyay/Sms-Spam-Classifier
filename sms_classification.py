import numpy as np
import pandas as pd
import re
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from nltk.stem import SnowballStemmer
from nltk.corpus import stopwords
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score


sms = pd.read_csv('spam.csv', encoding='latin-1')
sms.head()

sms = sms.drop(['Unnamed: 2','Unnamed: 3','Unnamed: 4'],axis=1)
sms = sms.rename(columns = {'v1':'label','v2':'message'})

sms.groupby('label').describe()

sms['length'] = sms['message'].apply(len)
sms.head()

text_feat = sms['message'].copy()
stop_words = ['the','a','an','and','or','because','as','this','that','these','those','then',
              'just','so','than','such','both','through','about','for','is','of','during','to']
              
def cleaner_function(text, remove_stop_words=True):
	text = re.sub(r"\b([A-Za-z]+)'re\b", '\\1 are', text)
	text = re.sub(r"\b([A-Za-z]+)'s\b", '\\1 is', text)
	text = re.sub(r"\b([A-Za-z]+)'m\b", '\\1 am', text)
	text = re.sub(r"\b([A-Za-z]+)'ve\b", '\\1 have', text)
	text = re.sub(r"\b([A-Za-z]+)'ll\b", '\\1 will', text)
	text = re.sub(r"\b([A-Za-z]+)n't\b", '\\1 not', text)
	text = re.sub(r"\b([A-Za-z]+)'d\b", '\\1 had', text)
	text = re.sub(r"[^A-Za-z0-9]", " ", text)
	if remove_stop_words:
		text = text.split()
		text = [w for w in text if not w in stop_words]
		text = " ".join(text)

	snowball_stemmer = SnowballStemmer('english')
	tx = snowball_stemmer.stem(text)
	#print tx
	return tx

for x in xrange(len(text_feat)):
	text_feat[x] = cleaner_function(text_feat[x])
#text_feat = text_process(text_feat2)
vectorizer = TfidfVectorizer("english")
features = vectorizer.fit_transform(text_feat).toarray()

features_train, features_test, labels_train, labels_test = train_test_split(features, sms['label'], test_size=0.3, random_state=111)

mnb = MultinomialNB(alpha=0.2)
dtc = DecisionTreeClassifier(min_samples_split=7, random_state=111)
lrc = LogisticRegression(solver='liblinear', penalty='l1')
mlpc = MLPClassifier(solver = 'adam', alpha = 1e-5, hidden_layer_sizes = (100,))
clfs = {'NB': mnb, 'DT': dtc, 'LR': lrc, 'NN':mlpc}

def train_classifier(clf, feature_train, labels_train):    
    clf.fit(feature_train, labels_train)

def predict_labels(clf, features):
    return (clf.predict(features))

pred_scores = []
for k,v in clfs.items():
    train_classifier(v, features_train, labels_train)
    pred = predict_labels(v,features_test)
    pred_scores.append((k, [accuracy_score(labels_test,pred)]))


df = pd.DataFrame.from_items(pred_scores,orient='index', columns=['Score'])
print df