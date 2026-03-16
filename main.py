import pickle
import pandas as pd
import re
import string
import nltk
nltk.download('punkt')
from nltk.tokenize import word_tokenize
nltk.download('wordnet')
from nltk.corpus import wordnet as wn
from nltk.stem import WordNetLemmatizer
nltk.download('stopwords')
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from mlxtend.classifier import StackingClassifier

df_fake=pd.read_csv("Fake.csv")
df_true=pd.read_csv("True.csv")

df_fake["class"]=0
df_true["class"]=1

df_fake_manual_testing=df_fake.tail(10)
df_fake.drop([23470,23480],axis=0,inplace=True)
df_true_manual_testing=df_true.tail(10)
df_true.drop([21406,21416],axis=0,inplace=True)
df_manual_testing=pd.concat([df_fake_manual_testing,df_true_manual_testing],axis=0)
df_manual_testing.to_csv("manual_testing.csv")

df_merge=pd.concat([df_fake,df_true],axis=0)

df=df_merge.drop(["subject","date"],axis=1)

df = df.sample(frac = 1)

def conversion(title):
 title = title.lower()
 title = re.sub('\[.*?\]', '', title)
 title = re.sub("\\W"," ",title)
 title = re.sub('https?://\S+|www\.\S+', '', title)
 title = re.sub('<.*?>+', '', title)
 title = re.sub('[%s]' % re.escape(string.punctuation), '', title)
 title = re.sub('\n', '', title)
 title = re.sub('\w*\d\w*', '', title)
 return title

df["title"] = df["title"].apply(conversion)


def tokenization(title):
 title = word_tokenize(title)
 return title

df["title"] = df["title"].apply(tokenization)

lmtzr=WordNetLemmatizer()
def lemmetization(title):
    title = ' '.join([lmtzr.lemmatize(w,wn.NOUN) for w in title])
    return title

df["title"] = df["title"].apply(lemmetization)

from nltk.corpus import stopwords
stop = stopwords.words('english')
df["title"] = df["title"].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop)]))

x = df.iloc[0:5000, 0]
y = df.iloc[0:5000, -1]
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

vectorization = TfidfVectorizer()
xv_train = vectorization.fit_transform(x_train)
xv_test = vectorization.transform(x_test)

pickle.dump(vectorization, open('transform.pkl','wb'))

base1=SVC()
base2=KNeighborsClassifier(n_neighbors=15)
meta_model=LogisticRegression()
stack=StackingClassifier(classifiers=[base1,base2],meta_classifier=meta_model)
stack.fit(xv_train,y_train)

filename='nlp_model.pkl'
pickle.dump(stack, open(filename, 'wb'))
