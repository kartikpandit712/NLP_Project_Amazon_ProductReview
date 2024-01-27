import nltk
from nltk.corpus import stopwords
import string
from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
nltk.download('punkt')
nltk.download('stopwords')
import warnings
warnings.filterwarnings('ignore')
import seaborn as sns
from wordcloud import WordCloud
from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB,MultinomialNB,BernoulliNB
from sklearn.metrics import accuracy_score,precision_score
from sklearn.preprocessing import MinMaxScaler

def load_the_dataset():
    data = pd.read_csv('spam.csv')
    return data

def find_null_values():
    return load_the_dataset().isnull().sum()

def clean_the_dataset():
    data = load_the_dataset().drop(columns=['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'], axis=1)
    new_column_names = {'v1': 'target', 'v2': 'text'}
    data = data.rename(columns=new_column_names)
    return data

def pre_process():
    data = clean_the_dataset()
    label_encoder = LabelEncoder()
    data['target'] = label_encoder.fit_transform(data['target'])
    return data

def find_duplicates():
    return pre_process().duplicated().sum()

def drop_duplicates():
    return pre_process().drop_duplicates()

def text_preprocessing(text):
  text = text.lower()
  text = nltk.word_tokenize(text)

  y = []
  for i in text:
    if i.isalnum():
      y.append(i)

  text = y.copy()
  y = []
  for i in text:
    if i not in stopwords.words('english') and i not in string.punctuation:
      y.append(i)

  text = y.copy()
  y = []
  for i in text:
    y.append(ps.stem(i))

  return ' '.join(y)

def pie_chart():
    data = drop_duplicates()
    plt.pie(data['target'].value_counts(), labels=['ham','spam'], autopct="0.2f")
    return data['target'].value_counts()

def char_hist():
    data = drop_duplicates()
    data['num_char'] = data['text'].apply(len)
    return data

def words_hist():
    data = char_hist()
    data['num_words'] = data['text'].apply(lambda x: len(nltk.word_tokenize(x)))
    return data

def sent_hist():
    data = words_hist()
    data['num_sent'] = data['text'].apply(lambda x: len(nltk.sent_tokenize(x)))
    return data

def correlation():
    return sent_hist().corr()

def transform_text():
    data = sent_hist()
    data['transformed_text'] = data['text'].apply(lambda x: text_preprocessing(x))  
    return data

def text_ham_wordcloud():
    data = transform_text()
    return data[data['target']==0]

def text_spam_wordcloud():
    data = transform_text()
    return data[data['target']==1]

def common_spam_words():
    data = transform_text()
    ls = data[data['target'] == 1]['transformed_text'].tolist()
    words = []
    for i in ls:
        for word in i.split(' '):
            words.append(word)
            
    c = Counter(words).most_common(30)
    c.sort(key=lambda x: x[1], reverse=True)
    return c

def tfidf_model():
    data = transform_text()
    tfidf = TfidfVectorizer()
    a = tfidf.fit_transform(data['transformed_text'])
    b = data['target']
    return a.toarray(),b

def split():
    X,y = tfidf_model()
    return train_test_split(X,y,test_size=0.2,random_state=2)

def gnb_model():
    X_train, X_test, y_train, y_test = split()
    gnb = GaussianNB()
    gnb.fit(X_train, y_train)
    y_pred = gnb.predict(X_test)
    return accuracy_score(y_test, y_pred), precision_score(y_test, y_pred)

def mnb_model():
    X_train, X_test, y_train, y_test = split()
    mnb = MultinomialNB()
    mnb.fit(X_train, y_train)
    y_pred = mnb.predict(X_test)
    return accuracy_score(y_test, y_pred), precision_score(y_test, y_pred)

def bnb_model():
    X_train, X_test, y_train, y_test = split()
    bnb = BernoulliNB()
    bnb.fit(X_train, y_train)
    y_pred = bnb.predict(X_test)
    return accuracy_score(y_test, y_pred), precision_score(y_test, y_pred)

a,b = gnb_model()
print('The GaussianNB model accuracy and precision are',a,'and',b)
a,b = mnb_model()
print('The MultinomialNB model accuracy and precision are',a,'and',b)
a,b = bnb_model()
print('The BernoulliNB model accuracy and precision are',a,'and',b)
