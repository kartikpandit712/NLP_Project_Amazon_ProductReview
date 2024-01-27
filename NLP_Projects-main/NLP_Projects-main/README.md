# Projects related to Natural Language Processing

1) Flipkart_reviews: This dataset consists of approx. 2300 Flipkart user reviews about the product. Using Nltk sentiment analysis, I found the positive, negative, and neutral scores of the reviews in this dataset. The scores are "Positive Score: 835.6700000000001, Negative Score: 104.917, Neutral Score: 1363.413"

2) Amazon_reviews: This dataset consists of approx. 15000 Amazon user reviews about the product. Using Nltk sentiment analysis, I found the positive, negative, and neutral scores of the reviews in this dataset. The scores are "Positive Score: 2879.8810000000003, Negative Score: 639.613, Neutral Score: 11480.474"

3) Spam Mail Detection: This dataset consists of 5.5k Email messages classified as either spam or ham. Data preprocessing and model building is done to predict whether the mail message is spam or ham. The highest precision and accuracy is 0.97 which is achieved by the Bernoulli Naive Bayes model.

To run code 1, 2 and 3 install the Nltk library and use command "python main.py" to run this code.

4) Language Prediction: This dataset has 22 languages with each language having 1000 records. The goal is to identify the type of language in the dataset. A count vectorizer is used to preprocess the data for modeling. Since it is a multi-classification problem, the Multinomial Naive Bayes algorithm is used to build the model giving an accuracy of around 95%.

5) Covid-19_Tweet_Classify (https://www.kaggle.com/datasets/datatattle/covid-19-nlp-text-classification):
Conducted sentiment analysis on a dataset comprising 41,157 training tweets and 3,798 test tweets, categorized into five sentiment types: Extremely Negative, Extremely Positive, Negative, Neutral, and Positive. Initially employed the Multinomial Naive Bayes algorithm with grid search hyperparameter tuning, achieving a preliminary accuracy of 48.05%. Subsequently, transitioned to leveraging the BERT (bert-base-uncased) model for sequence classification. With optimized tokenization, attention masks, and fine-tuning over 10 epochs, achieved a remarkable accuracy of approximately 84% on the test data, underscoring the efficacy of transfer learning with contextual embeddings in NLP tasks.

6) Text Summarization: The code defines functions to generate abstractive summaries using the Hugging Face's Transformers library. The load_model_and_tokenizer function retrieves a specified pre-trained model and its tokenizer. The abstractive_summary function takes in a text, the desired model name, and optional arguments for input and output length, then uses the model to generate a summarized version of the input text. When executed, the script prompts the user to specify a model from options like t5-small, facebook/bart-large-cnn, etc., and then input the text they want to summarize. After processing, the script outputs the generated summary to the console.


