import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import warnings
nltk.download('vader_lexicon')
warnings.filterwarnings('ignore')

#Load the dataset
def load_dataset():
    data = pd.read_csv('Reviews.csv')
    return data

#Drop the missing columns
def pre_processing():
    data = load_dataset().dropna()
    return data

#Finding the distribution of ratings
def distribution_of_ratings():
    data = pre_processing()
    ratings = data['Score'].value_counts()
    return ratings

#Calculating positive, negative and neutral values in the dataframe and merging with the original dataframe
def add_new_columns():
    data = pre_processing()
    analyzer = SentimentIntensityAnalyzer()

    pos = []
    neg = []
    neu = []

    for i in range(0,len(data)):
        scores = analyzer.polarity_scores(data['Text'][i])
        pos.append(scores['pos'])
        neg.append(scores['neg'])
        neu.append(scores['neu'])
    
    data['Positive'] = pos
    data['Negative'] = neg
    data['Neutral'] = neu

    data = data[['Id','ProductId','UserId','ProfileName','HelpfulnessNumerator','HelpfulnessDenominator','Score','Time','Summary','Text','Positive','Negative','Neutral']]
    return data

#Finding the sum of positive, negative and neutral values
def sentiment_scores():
    data = add_new_columns()

    x = data['Positive'].sum()
    y = data['Negative'].sum()
    z = data['Neutral'].sum()

    max_value = max(x,y,z)
    
    if max_value == x:
        return "Positive"
    elif max_value == y:
        return "Negative"
    else:
        return "Neutral"

#Calling all functions
load_dataset()

data = pre_processing()
print(data)

ratings = distribution_of_ratings()
print('The rating distribution is\n',ratings)

sentiment = sentiment_scores()
print('The sentiment with maximum score is\n',sentiment)

add_new_columns()
x,y,z = sentiment_scores()
print(f'Positive Score: {x}, Negative Score: {y}, Neutral Score: {z}')
