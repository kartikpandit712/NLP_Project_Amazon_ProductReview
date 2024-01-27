import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

model = None

def load_the_dataset():
    data = pd.read_csv("dataset.csv")
    print(data.head())
    print(data.shape)
    return data

def check_null_values():
    data =load_the_dataset()
    return data.isnull().sum()

def languages_present():
    data = load_the_dataset()
    return data["language"].value_counts()

def train_and_test_data():
    data = load_the_dataset()
    global X_train, X_test, y_train, y_test, cv
    x = np.array(data["Text"])
    y = np.array(data["language"])

    cv = CountVectorizer()
    X = cv.fit_transform(x)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
    return X_train, X_test, y_train, y_test, cv, "Data Trained Successfully"

def model_training():
    output = train_and_test_data()
    X_train, X_test, y_train, y_test=output[0],output[1],output[2],output[3]
    global model
    model = MultinomialNB()
    model.fit(X_train,y_train)
    score=model.score(X_test,y_test)
    print(score)
    return score

def detect_the_language():
    model_training()
    global model
    output= train_and_test_data()
    cv=output[4]
    user = input("Enter a Text: ")
    data = cv.transform([user]).toarray()
    output = model.predict(data)
    print(output)
    return output

load_the_dataset()
print(check_null_values())
print(languages_present())
print(train_and_test_data())
model_training()
detect_the_language()