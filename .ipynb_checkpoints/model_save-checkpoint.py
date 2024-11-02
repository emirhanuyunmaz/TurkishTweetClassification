import numpy as np 
import pandas as pd
import re
import nltk
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix
from sklearn.impute import SimpleImputer
import pickle
from sklearn.svm import SVC

durma = nltk.download("stopwords")

from nltk.corpus import stopwords

veriler = pd.read_excel("TurkishTweets.xlsx")
# print(veriler["Tweet"][3430])


yorum = re.sub('[^a-zA-Z0-9ğüşöçıİĞÜŞÖÇ]',' ', veriler["Tweet"][3430])
yorum = yorum.lower()
yorum = yorum.split()
ps = PorterStemmer()
yorum = [ps.stem(kelime) for kelime in yorum if not kelime in set(stopwords.words("turkish"))]
yorum = ' '.join(yorum)
# print(yorum)

derlem = []

for i in range (3999):
    yorum = re.sub('[^a-zA-ZğüşöçıİĞÜŞÖÇ]',' ', veriler["Tweet"][i])
    yorum = yorum.lower()
    yorum = yorum.split()
    ps = PorterStemmer()
    yorum = [ps.stem(kelime) for kelime in yorum if not kelime in set(stopwords.words("turkish"))]
    yorum = ' '.join(yorum)
    derlem.append(yorum)

cv = CountVectorizer(max_features=2500)
x = cv.fit_transform(derlem).toarray()
y = veriler.iloc[:,1].values

x_train , x_test , y_train , y_test = train_test_split(x,y,test_size=0.33,random_state=0)

svc = SVC(kernel="rbf") #Kernel kullanılacak olan fonk. seçimi yapılması işlemi.
svc.fit(x_train,y_train)
y_pred = svc.predict(x_test)
cm = confusion_matrix(y_test,y_pred)
print(cm)

# Model ve vektörleştiriciyi kaydetme
with open('model.pkl', 'wb') as model_file:
    pickle.dump(svc, model_file)

with open('count_vectorizer.pkl', 'wb') as cv_file:
    pickle.dump(cv, cv_file)